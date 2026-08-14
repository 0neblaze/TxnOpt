from __future__ import annotations

import json
from pathlib import Path

import pytest

from txnopt_evidence.toolchain import (
    ToolchainEntry,
    read_toolchain_lock,
    write_toolchain_lock,
)


def _entry(name: str) -> ToolchainEntry:
    return ToolchainEntry(
        name=name,
        version="locked-version",
        executable_path=f"/external/{name}",
        sha256="a" * 64,
        validation_exit_code=0,
        source_url="https://example.invalid/tool",
        download_sha256="b" * 64,
    )


def test_toolchain_lock_requires_complete_exact_tool_set(tmp_path: Path) -> None:
    output = tmp_path / "toolchain-lock.json"
    entries = [_entry(name) for name in (
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
    )]

    digest = write_toolchain_lock(output, entries)
    payload = read_toolchain_lock(output)

    assert len(digest) == 64
    assert payload["schema_version"] == "txnopt-toolchain-lock-v1"
    assert payload["status"] == "TOOLCHAIN_COMPLETE"
    assert payload["all_validation_exit_codes_zero"] is True


def test_toolchain_lock_rejects_missing_or_failed_tools(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="tool set"):
        write_toolchain_lock(tmp_path / "missing.json", [_entry("uv")])

    failed = [_entry(name) for name in (
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
    )]
    failed[0] = ToolchainEntry(
        name="uv",
        version="0.12.4",
        executable_path="/external/uv",
        sha256="a" * 64,
        validation_exit_code=1,
        source_url="https://example.invalid/uv",
        download_sha256="b" * 64,
    )
    failed_path = tmp_path / "failed.json"
    write_toolchain_lock(failed_path, failed)
    blocked = read_toolchain_lock(failed_path, require_complete=False)
    assert blocked["status"] == "BLOCKED_TOOLCHAIN"
    assert blocked["all_validation_exit_codes_zero"] is False
    with pytest.raises(ValueError, match="exit code"):
        read_toolchain_lock(failed_path)


def test_toolchain_lock_rejects_resigned_tamper(tmp_path: Path) -> None:
    output = tmp_path / "toolchain-lock.json"
    entries = [_entry(name) for name in (
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
    )]
    write_toolchain_lock(output, entries)
    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["tools"][0]["version"] = "forged"
    output.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="sidecar"):
        read_toolchain_lock(output)
