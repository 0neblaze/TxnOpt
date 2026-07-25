"""Fail-fast platform primitives used by the Stage 5.2 evidence pipeline."""

from __future__ import annotations

import ctypes
import json
import os
import platform
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

MOVEFILE_REPLACE_EXISTING = 0x00000001
MOVEFILE_WRITE_THROUGH = 0x00000008

_WINDOWS_POWER_STATUS_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
$source = @'
using System;
using System.Runtime.InteropServices;
public static class Stage052NativePower {
    [StructLayout(LayoutKind.Sequential)]
    public struct SYSTEM_POWER_STATUS {
        public byte ACLineStatus;
        public byte BatteryFlag;
        public byte BatteryLifePercent;
        public byte SystemStatusFlag;
        public uint BatteryLifeTime;
        public uint BatteryFullLifeTime;
    }
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern bool GetSystemPowerStatus(out SYSTEM_POWER_STATUS status);
}
'@
Add-Type -TypeDefinition $source
$status = New-Object Stage052NativePower+SYSTEM_POWER_STATUS
if (-not [Stage052NativePower]::GetSystemPowerStatus([ref]$status)) {
    throw 'GetSystemPowerStatus failed'
}
$scheme = Get-ItemPropertyValue `
    -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\Power\User\PowerSchemes' `
    -Name ActivePowerScheme
[ordered]@{
    ac_line_status = [int]$status.ACLineStatus
    battery_flag = [int]$status.BatteryFlag
    battery_life_percent = [int]$status.BatteryLifePercent
    battery_saver = [int]$status.SystemStatusFlag
    active_power_scheme = [string]$scheme
} | ConvertTo-Json -Compress
""".strip()


def posix_file_cache_drop_is_safe(
    *,
    platform_name: str | None = None,
    kernel_release: str | None = None,
) -> bool:
    """Return whether POSIX_FADV_DONTNEED is safe for evidence files."""

    current_platform = sys.platform if platform_name is None else platform_name
    current_release = platform.release() if kernel_release is None else kernel_release
    return (
        current_platform != "win32"
        and "microsoft-standard-wsl" not in current_release.casefold()
    )


@dataclass(frozen=True, slots=True)
class WindowsWslPowerStatus:
    """Non-GUI Windows/WSL2 power state used by performance evidence."""

    ac_online: bool
    battery_saver: bool
    battery_life_percent: int
    battery_flag: int
    active_power_scheme: str


def read_windows_wsl_power_status(
    *,
    ac_online_path: Path = Path("/sys/class/power_supply/AC1/online"),
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> WindowsWslPowerStatus:
    """Read WSL AC state and Windows native power flags without opening a GUI."""

    wsl_ac_online = read_wsl_ac_power_online(ac_online_path)
    completed = run(
        (
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            _WINDOWS_POWER_STATUS_SCRIPT,
        ),
        check=True,
        capture_output=True,
        text=True,
        timeout=10.0,
    )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("cannot decode Windows native power status") from error
    if not isinstance(payload, dict):
        raise RuntimeError("Windows native power status must be a JSON object")
    ac_line_status = payload.get("ac_line_status")
    battery_saver = payload.get("battery_saver")
    battery_life_percent = payload.get("battery_life_percent")
    battery_flag = payload.get("battery_flag")
    active_power_scheme = payload.get("active_power_scheme")
    if ac_line_status not in {0, 1} or bool(ac_line_status) is not wsl_ac_online:
        raise RuntimeError("Windows and WSL2 AC power states do not agree")
    if battery_saver not in {0, 1}:
        raise RuntimeError("cannot determine Windows battery saver state")
    if (
        isinstance(battery_life_percent, bool)
        or not isinstance(battery_life_percent, int)
        or battery_life_percent not in range(0, 256)
    ):
        raise RuntimeError("Windows battery life percentage is invalid")
    if isinstance(battery_flag, bool) or not isinstance(battery_flag, int):
        raise RuntimeError("Windows battery flag is invalid")
    if not isinstance(active_power_scheme, str) or not active_power_scheme.strip():
        raise RuntimeError("Windows active power scheme is unavailable")
    return WindowsWslPowerStatus(
        ac_online=wsl_ac_online,
        battery_saver=bool(battery_saver),
        battery_life_percent=battery_life_percent,
        battery_flag=battery_flag,
        active_power_scheme=active_power_scheme.strip(),
    )


def read_wsl_ac_power_online(
    path: Path = Path("/sys/class/power_supply/AC1/online"),
) -> bool:
    """Read the WSL2 kernel AC flag without launching a Windows process."""

    try:
        value = int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError) as error:
        raise RuntimeError(f"cannot read WSL2 AC power state: {path}") from error
    if value not in {0, 1}:
        raise RuntimeError("WSL2 AC power state must be zero or one")
    return bool(value)


class _ProcessFactory(Protocol):
    def __call__(self) -> object: ...


class _MoveFileEx(Protocol):
    def __call__(self, source: str, destination: str, flags: int) -> object: ...


def peak_rss_bytes(
    *,
    platform_name: str | None = None,
    getrusage: Callable[[], int] | None = None,
    process_factory: _ProcessFactory | None = None,
) -> int:
    """Return process peak RSS with explicit Linux and Windows semantics."""

    platform = sys.platform if platform_name is None else platform_name
    if platform == "win32":
        if process_factory is None:
            import psutil  # type: ignore[import-untyped]

            process_factory = psutil.Process
        memory_info = getattr(process_factory(), "memory_info", None)
        if not callable(memory_info):
            raise RuntimeError("Windows peak RSS requires psutil Process.memory_info")
        peak_wset = getattr(memory_info(), "peak_wset", None)
        if isinstance(peak_wset, bool) or not isinstance(peak_wset, int) or peak_wset <= 0:
            raise RuntimeError("Windows peak RSS requires a positive psutil peak_wset")
        return peak_wset

    if getrusage is None:
        import resource

        def read_rusage() -> int:
            return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

        getrusage = read_rusage
    value = getrusage()
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeError("POSIX getrusage returned an invalid peak RSS")
    return value if platform == "darwin" else value * 1024


def durable_replace(
    source: Path,
    destination: Path,
    *,
    platform_name: str | None = None,
    windows_move_file_ex: _MoveFileEx | None = None,
) -> None:
    """Replace one path durably without weakening native Windows semantics."""

    platform = sys.platform if platform_name is None else platform_name
    if platform != "win32":
        os.replace(source, destination)
        return
    move_file_ex = windows_move_file_ex or _native_move_file_ex()
    flags = MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH
    if not bool(move_file_ex(str(source), str(destination), flags)):
        get_last_error = getattr(ctypes, "get_last_error", lambda: 0)
        error_code = int(get_last_error())
        raise OSError(error_code, f"MoveFileExW failed: {source} -> {destination}")


def sync_directory(path: Path, *, platform_name: str | None = None) -> None:
    """Persist POSIX directory metadata; Windows durability is in MoveFileExW."""

    platform = sys.platform if platform_name is None else platform_name
    if platform == "win32":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _native_move_file_ex() -> _MoveFileEx:
    loader = getattr(ctypes, "WinDLL", None)
    if loader is None:
        raise RuntimeError("MoveFileExW is unavailable outside native Windows")
    function = loader("kernel32", use_last_error=True).MoveFileExW
    function.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32]
    function.restype = ctypes.c_int
    return cast(_MoveFileEx, function)
