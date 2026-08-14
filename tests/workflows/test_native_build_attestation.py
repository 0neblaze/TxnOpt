from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tools.native_build_attestation import (
    committed_source_attestation,
    committed_wheel_project_entry_sha256,
    inspect_source,
    materialize_committed_cpp,
    validate_scheduler_build_attestation,
    verify_committed_cpp,
)


def _git(root: Path, *arguments: str) -> None:
    subprocess.run(("git", *arguments), cwd=root, check=True, capture_output=True)


def _repository(tmp_path: Path) -> Path:
    root = tmp_path / "source"
    root.mkdir()
    _git(root, "init")
    _git(root, "config", "user.email", "native-attestation@example.invalid")
    _git(root, "config", "user.name", "Native Attestation Test")
    cpp = root / "cpp"
    cpp.mkdir()
    (cpp / "producer.cpp").write_text("int value = 1;\n", encoding="utf-8")
    tools = root / "tools"
    tools.mkdir()
    (tools / "__init__.py").write_text("", encoding="utf-8")
    package = root / "src" / "txnopt"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    _git(
        root,
        "add",
        "cpp/producer.cpp",
        "tools/__init__.py",
        "src/txnopt/__init__.py",
    )
    _git(root, "commit", "-m", "initial")
    return root


def test_native_build_attestation_reads_bytes_despite_assume_unchanged(
    tmp_path: Path,
) -> None:
    root = _repository(tmp_path)
    clean = inspect_source(root, development_override=False)
    committed = committed_source_attestation(root, "HEAD")
    committed_wheel_bytes = committed_wheel_project_entry_sha256(root, "HEAD")
    assert clean["source_dirty"] is False
    assert clean["development_override"] is False
    assert clean["source_manifest_sha256"] == committed["source_manifest_sha256"]
    assert clean["tracked_file_count"] == committed["tracked_file_count"]

    _git(root, "update-index", "--assume-unchanged", "cpp/producer.cpp")
    (root / "cpp" / "producer.cpp").write_text("int value = 2;\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="producer.cpp"):
        inspect_source(root, development_override=False)
    development = inspect_source(root, development_override=True)
    assert development["source_dirty"] is True
    assert development["development_override"] is True
    assert development["source_manifest_sha256"] != clean["source_manifest_sha256"]
    assert committed_source_attestation(root, "HEAD") == committed
    assert committed_wheel_project_entry_sha256(root, "HEAD") == committed_wheel_bytes


def test_native_cpp_snapshot_uses_committed_blobs_and_detects_tampering(
    tmp_path: Path,
) -> None:
    root = _repository(tmp_path)
    snapshot = tmp_path / "snapshot"
    materialize_committed_cpp(root, "HEAD", snapshot)
    snapshotted_source = snapshot / "cpp" / "producer.cpp"
    assert snapshotted_source.read_text(encoding="utf-8") == "int value = 1;\n"
    verify_committed_cpp(root, "HEAD", snapshot)

    (root / "cpp" / "producer.cpp").write_text("int value = 2;\n", encoding="utf-8")
    verify_committed_cpp(root, "HEAD", snapshot)

    snapshotted_source.write_text("int value = 3;\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="snapshot bytes do not match Git"):
        verify_committed_cpp(root, "HEAD", snapshot)


def test_native_build_attestation_rejects_parent_repository_identity(
    tmp_path: Path,
) -> None:
    root = _repository(tmp_path)
    nested = root / "untracked-source-archive"
    nested.mkdir()
    (nested / "producer.cpp").write_text("int value = 1;\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="exact Git top-level"):
        inspect_source(nested, development_override=True)


def test_scheduler_build_attestation_rejects_json_type_confusion() -> None:
    expected: dict[str, object] = {
        "schema_version": 1,
        "revision": "a" * 40,
        "git_tree": "b" * 40,
        "source_manifest_sha256": "c" * 64,
        "tracked_file_count": 919,
        "source_dirty": False,
        "development_override": False,
        "cpp_source_kind": "git_blob_snapshot",
    }
    validate_scheduler_build_attestation(
        expected,
        revision="a" * 40,
        git_tree="b" * 40,
        source_manifest_sha256="c" * 64,
        tracked_file_count=919,
    )
    for field, invalid in (
        ("schema_version", True),
        ("tracked_file_count", 919.0),
        ("source_dirty", 0),
        ("development_override", 0),
    ):
        payload = {**expected, field: invalid}
        with pytest.raises(RuntimeError, match="type is invalid"):
            validate_scheduler_build_attestation(
                payload,
                revision="a" * 40,
                git_tree="b" * 40,
                source_manifest_sha256="c" * 64,
                tracked_file_count=919,
            )


def test_scheduler_build_attestation_binds_performance_profile() -> None:
    expected: dict[str, object] = {
        "schema_version": 2,
        "revision": "a" * 40,
        "git_tree": "b" * 40,
        "source_manifest_sha256": "c" * 64,
        "tracked_file_count": 919,
        "source_dirty": False,
        "development_override": False,
        "cpp_source_kind": "git_blob_snapshot",
        "performance_profile": "host-native-lto",
        "compiler_id": "GNU",
        "compiler_version": "14.2.0",
        "interprocedural_optimization": True,
        "host_native": True,
    }
    validate_scheduler_build_attestation(
        expected,
        revision="a" * 40,
        git_tree="b" * 40,
        source_manifest_sha256="c" * 64,
        tracked_file_count=919,
        performance_profile="host-native-lto",
        compiler_id="GNU",
        compiler_version="14.2.0",
        interprocedural_optimization=True,
        host_native=True,
    )
    with pytest.raises(RuntimeError, match="contradict"):
        validate_scheduler_build_attestation(
            {**expected, "interprocedural_optimization": False},
            revision="a" * 40,
            git_tree="b" * 40,
            source_manifest_sha256="c" * 64,
            tracked_file_count=919,
        )
    with pytest.raises(RuntimeError, match="does not reconcile"):
        validate_scheduler_build_attestation(
            expected,
            revision="a" * 40,
            git_tree="b" * 40,
            source_manifest_sha256="c" * 64,
            tracked_file_count=919,
            performance_profile="portable-lto",
        )
