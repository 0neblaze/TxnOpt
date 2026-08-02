from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "evrptw" / "stage052_platform.py"
_SPEC = importlib.util.spec_from_file_location("stage052_platform_under_test", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_PLATFORM = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _PLATFORM
_SPEC.loader.exec_module(_PLATFORM)

MOVEFILE_REPLACE_EXISTING = _PLATFORM.MOVEFILE_REPLACE_EXISTING
MOVEFILE_WRITE_THROUGH = _PLATFORM.MOVEFILE_WRITE_THROUGH
durable_replace = _PLATFORM.durable_replace
peak_rss_bytes = _PLATFORM.peak_rss_bytes
posix_file_cache_drop_is_safe = _PLATFORM.posix_file_cache_drop_is_safe
read_windows_wsl_power_status = _PLATFORM.read_windows_wsl_power_status


def test_linux_peak_rss_uses_getrusage_kib_units() -> None:
    assert peak_rss_bytes(
        platform_name="linux",
        getrusage=lambda: 2048,
    ) == 2 * 1024 * 1024


def test_posix_file_cache_drop_is_limited_to_native_ext4_on_wsl() -> None:
    assert (
        posix_file_cache_drop_is_safe(
            platform_name="linux",
            kernel_release="6.18.35.2-microsoft-standard-WSL2",
        )
        is False
    )
    assert (
        posix_file_cache_drop_is_safe(
            platform_name="linux",
            kernel_release="6.8.0-71-generic",
        )
        is True
    )
    assert (
        posix_file_cache_drop_is_safe(
            platform_name="win32",
            kernel_release="10",
        )
        is False
    )
    assert (
        posix_file_cache_drop_is_safe(
            platform_name="linux",
            kernel_release="6.18.35.2-microsoft-standard-WSL2",
            filesystem_name="ext4",
        )
        is True
    )
    assert (
        posix_file_cache_drop_is_safe(
            platform_name="linux",
            kernel_release="6.18.35.2-microsoft-standard-WSL2",
            filesystem_name="9p",
        )
        is False
    )


def test_windows_peak_rss_requires_peak_wset() -> None:
    assert peak_rss_bytes(
        platform_name="win32",
        process_factory=lambda: SimpleNamespace(
            memory_info=lambda: SimpleNamespace(peak_wset=123_456)
        ),
    ) == 123_456

    with pytest.raises(RuntimeError, match="peak_wset"):
        peak_rss_bytes(
            platform_name="win32",
            process_factory=lambda: SimpleNamespace(
                memory_info=lambda: SimpleNamespace(rss=123_456)
            ),
        )


def test_windows_durable_replace_uses_write_through_movefileex(tmp_path: Path) -> None:
    source = tmp_path / "source.tmp"
    destination = tmp_path / "destination.json"
    source.write_text("payload", encoding="utf-8")
    calls: list[tuple[str, str, int]] = []

    def move_file_ex(source_path: str, destination_path: str, flags: int) -> bool:
        calls.append((source_path, destination_path, flags))
        Path(source_path).replace(destination_path)
        return True

    durable_replace(
        source,
        destination,
        platform_name="win32",
        windows_move_file_ex=move_file_ex,
    )

    assert calls == [
        (
            str(source),
            str(destination),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
        )
    ]
    assert destination.read_text(encoding="utf-8") == "payload"


def test_windows_power_status_uses_non_gui_native_api_and_wsl_ac(
    tmp_path: Path,
) -> None:
    ac_online = tmp_path / "online"
    ac_online.write_text("1\n", encoding="utf-8")
    calls: list[tuple[str, ...]] = []

    def run(
        arguments: tuple[str, ...],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        calls.append(arguments)
        assert kwargs["check"] is True
        return subprocess.CompletedProcess(
            arguments,
            0,
            stdout=(
                '{"ac_line_status":1,"battery_flag":8,'
                '"battery_life_percent":76,"battery_saver":0,'
                '"active_power_scheme":"balanced-guid"}'
            ),
            stderr="",
        )

    observed = read_windows_wsl_power_status(ac_online_path=ac_online, run=run)

    assert observed.ac_online is True
    assert observed.battery_saver is False
    assert observed.active_power_scheme == "balanced-guid"
    command = " ".join(calls[0])
    assert "GetSystemPowerStatus" in command
    assert "powercfg" not in command.casefold()
    assert "control.exe" not in command.casefold()
    assert "powercfg.cpl" not in command.casefold()


def test_current_stage052_source_does_not_invoke_power_control_panel() -> None:
    repository = _MODULE_PATH.parents[2]
    current_sources = (
        repository / "src" / "evrptw" / "stage052_platform.py",
        repository / "src" / "evrptw" / "stage052_evidence.py",
        repository / "src" / "evrptw" / "stage052_campaign_runner.py",
    )
    forbidden = ("powercfg.exe", "control.exe", "powercfg.cpl")

    for source in current_sources:
        text = source.read_text(encoding="utf-8").casefold()
        for token in forbidden:
            assert token not in text
