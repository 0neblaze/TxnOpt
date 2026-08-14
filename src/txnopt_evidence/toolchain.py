"""Exact, signed toolchain identity for the Tencent pre-cloud build."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from txnopt_evidence.codec import read_signed_json, write_signed_json

_REQUIRED_TOOLS = frozenset(
    {
        "uv",
        "cpython",
        "temurin-jdk",
        "tla2tools",
        "tlc",
        "pluscal",
        "jq",
        "numactl",
        "lsof",
        "fuser",
        "ipcs",
        "gcc",
        "cmake",
        "ninja",
        "cos-python-sdk-v5",
        "tencentcloud-sdk-python-common",
        "tencentcloud-sdk-python-cvm",
    }
)


@dataclass(frozen=True, slots=True)
class ToolchainEntry:
    """One exact executable or downloaded distribution in the build toolchain."""

    name: str
    version: str
    executable_path: str
    sha256: str
    validation_exit_code: int
    source_url: str | None = None
    download_sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.version or not self.executable_path:
            raise ValueError("toolchain identity fields cannot be empty")
        _validate_sha256(self.sha256, field="tool sha256")
        if self.validation_exit_code < 0:
            raise ValueError("tool validation exit code must be non-negative")
        if (self.source_url is None) != (self.download_sha256 is None):
            raise ValueError("download URL and digest must be recorded together")
        if self.download_sha256 is not None:
            _validate_sha256(self.download_sha256, field="download sha256")


def write_toolchain_lock(path: Path, entries: list[ToolchainEntry]) -> str:
    """Write the complete canonical pre-cloud toolchain lock once."""

    names = [entry.name for entry in entries]
    if len(names) != len(set(names)) or set(names) != _REQUIRED_TOOLS:
        raise ValueError("toolchain lock tool set differs from the required set")
    ordered = sorted(entries, key=lambda entry: entry.name)
    complete = all(entry.validation_exit_code == 0 for entry in ordered)
    payload = {
        "schema_version": "txnopt-toolchain-lock-v1",
        "status": "TOOLCHAIN_COMPLETE" if complete else "BLOCKED_TOOLCHAIN",
        "all_validation_exit_codes_zero": complete,
        "repository_state_written": False,
        "tools": [asdict(entry) for entry in ordered],
    }
    return write_signed_json(path, payload)


def read_toolchain_lock(
    path: Path,
    *,
    require_complete: bool = True,
) -> dict[str, Any]:
    """Read and fail closed on an incomplete or self-inconsistent lock."""

    payload = read_signed_json(path)
    if set(payload) != {
        "schema_version",
        "status",
        "all_validation_exit_codes_zero",
        "repository_state_written",
        "tools",
    }:
        raise ValueError("toolchain lock field set differs")
    if payload["schema_version"] != "txnopt-toolchain-lock-v1" or payload[
        "repository_state_written"
    ] is not False:
        raise ValueError("toolchain lock status differs")
    raw_tools = payload["tools"]
    if not isinstance(raw_tools, list):
        raise ValueError("toolchain lock tools must be a list")
    entries: list[ToolchainEntry] = []
    for raw in raw_tools:
        if not isinstance(raw, dict) or set(raw) != {
            "name",
            "version",
            "executable_path",
            "sha256",
            "validation_exit_code",
            "source_url",
            "download_sha256",
        }:
            raise ValueError("toolchain entry field set differs")
        entries.append(
            ToolchainEntry(
                name=_required_string(raw, "name"),
                version=_required_string(raw, "version"),
                executable_path=_required_string(raw, "executable_path"),
                sha256=_required_string(raw, "sha256"),
                validation_exit_code=_required_int(raw, "validation_exit_code"),
                source_url=_optional_string(raw, "source_url"),
                download_sha256=_optional_string(raw, "download_sha256"),
            )
        )
    names = [entry.name for entry in entries]
    if names != sorted(names) or set(names) != _REQUIRED_TOOLS:
        raise ValueError("toolchain lock tool set differs from the required set")
    complete = all(entry.validation_exit_code == 0 for entry in entries)
    expected_status = "TOOLCHAIN_COMPLETE" if complete else "BLOCKED_TOOLCHAIN"
    if (
        payload["status"] != expected_status
        or payload["all_validation_exit_codes_zero"] is not complete
    ):
        raise ValueError("toolchain lock status differs from its exit codes")
    if require_complete and not complete:
        raise ValueError("tool validation exit code is nonzero")
    return payload


def _required_string(payload: dict[object, object], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"toolchain {field} must be a nonempty string")
    return value


def _optional_string(payload: dict[object, object], field: str) -> str | None:
    value = payload.get(field)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"toolchain {field} must be null or a nonempty string")
    return value


def _required_int(payload: dict[object, object], field: str) -> int:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"toolchain {field} must be an integer")
    return value


def _validate_sha256(value: str, *, field: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field} must be lowercase SHA-256")


__all__ = ["ToolchainEntry", "read_toolchain_lock", "write_toolchain_lock"]
