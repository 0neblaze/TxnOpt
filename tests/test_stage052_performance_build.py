from __future__ import annotations

import hashlib
import json
import os
import subprocess
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from evrptw.experiments.stage052_performance_build import (
    PerformanceBuildError,
    _atomic_signed_json,
    _publish_verified_profile_receipt,
    _safe_extract_wheel,
    _validated_python_executable,
    _verify_profile_compile_commands,
    _verify_wheel_source_inventory,
    main,
    probe_host_native_lto_support,
    reject_ambient_build_flags,
    require_clean_revision,
)
from evrptw.experiments.stage052_performance_calibration import WheelReceipt


def test_build_interpreter_preserves_virtual_environment_symlink(
    tmp_path: Path,
) -> None:
    base_python = tmp_path / "base-python"
    base_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    base_python.chmod(0o755)
    venv_python = tmp_path / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(base_python)

    validated = _validated_python_executable(venv_python)

    assert validated == venv_python.absolute()
    assert validated.is_symlink()


@pytest.mark.parametrize(
    ("profile", "flags"),
    (
        ("portable-o3", "-O3"),
        ("portable-lto", "-O3 -flto=auto"),
        ("host-native-lto", "-O3 -flto=auto -march=native"),
    ),
)
def test_compile_command_profile_surface_is_explicit(
    profile: str,
    flags: str,
) -> None:
    _verify_profile_compile_commands(
        profile,
        {"commands": [f"/usr/bin/g++ {flags} -c source.cpp"]},
    )


@pytest.mark.parametrize(
    ("profile", "flags"),
    (
        ("portable-o3", "-O3 -flto=auto"),
        ("portable-lto", "-O3"),
        ("portable-lto", "-O3 -flto=auto -march=native"),
        ("host-native-lto", "-O3 -flto=auto"),
    ),
)
def test_compile_command_profile_surface_rejects_implicit_or_missing_flags(
    profile: str,
    flags: str,
) -> None:
    with pytest.raises(PerformanceBuildError, match="compile command"):
        _verify_profile_compile_commands(
            profile,
            {"commands": [f"/usr/bin/g++ {flags} -c source.cpp"]},
        )


def test_invalid_compile_surface_is_rejected_before_receipt_publication(
    tmp_path: Path,
) -> None:
    receipt_path = tmp_path / "wheel_receipt.json"

    with pytest.raises(PerformanceBuildError, match="invalid LTO surface"):
        _publish_verified_profile_receipt(
            profile="portable-o3",
            receipt_path=receipt_path,
            receipt=cast(WheelReceipt, SimpleNamespace()),
            compile_commands={"commands": ["/usr/bin/g++ -O3 -flto=auto -c source.cpp"]},
        )

    assert not receipt_path.exists()
    assert not receipt_path.with_suffix(".json.sha256").exists()


def test_performance_build_cli_requires_repository_root(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as raised:
        main(["--output-root", str(tmp_path / "builds")])
    assert raised.value.code == 2


def test_clean_revision_identity_rejects_untracked_source(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(("git", "init", "-q"), cwd=repository, check=True)
    subprocess.run(
        ("git", "config", "user.email", "stage052@example.invalid"),
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ("git", "config", "user.name", "Stage 5.2 Test"),
        cwd=repository,
        check=True,
    )
    (repository / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    subprocess.run(("git", "add", "tracked.txt"), cwd=repository, check=True)
    environment = dict(os.environ)
    environment.update(
        {
            "GIT_AUTHOR_DATE": "2026-08-09T00:00:00+00:00",
            "GIT_COMMITTER_DATE": "2026-08-09T00:00:00+00:00",
        }
    )
    subprocess.run(
        ("git", "commit", "-q", "-m", "fixture"),
        cwd=repository,
        env=environment,
        check=True,
    )

    revision, tree, epoch = require_clean_revision(repository)
    assert len(revision) == 40
    assert len(tree) == 40
    assert epoch > 0

    (repository / "untracked.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(PerformanceBuildError, match="clean Git commit"):
        require_clean_revision(repository)


def test_ambient_build_flags_are_fail_closed() -> None:
    reject_ambient_build_flags({})
    with pytest.raises(PerformanceBuildError, match="freeze their own flags"):
        reject_ambient_build_flags({"CXXFLAGS": "-g"})
    with pytest.raises(PerformanceBuildError, match="fast-math"):
        reject_ambient_build_flags({"CMAKE_ARGS": "-DCMAKE_CXX_FLAGS=-Ofast"})


def test_wheel_extraction_and_clean_source_inventory(tmp_path: Path) -> None:
    wheel = tmp_path / "candidate.whl"
    source = b"source\n"
    native = b"native"
    scheduler = b"scheduler"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("evrptw/__init__.py", source)
        archive.writestr("_ignored.dist-info/METADATA", b"metadata")
        archive.writestr(
            "evrptw/_core.cpython-313-x86_64-linux-gnu.so",
            native,
        )
        archive.writestr("evrptw/_native_host_scheduler", scheduler)
    installed = tmp_path / "installed"
    native_member, scheduler_member = _safe_extract_wheel(wheel, installed)
    assert native_member == "evrptw/_core.cpython-313-x86_64-linux-gnu.so"
    assert scheduler_member == "evrptw/_native_host_scheduler"
    assert (installed / native_member).read_bytes() == native
    assert os.access(installed / scheduler_member, os.X_OK)
    _verify_wheel_source_inventory(
        wheel,
        expected_entries={
            "evrptw/__init__.py": hashlib.sha256(source).hexdigest(),
        },
        native_member=native_member,
        scheduler_member=scheduler_member,
    )


def test_signed_build_receipt_hashes_published_bytes(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    digest = _atomic_signed_json(path, {"schema_version": "fixture", "value": 1})
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert path.with_suffix(".json.sha256").read_text(encoding="ascii").strip() == digest
    assert json.loads(path.read_text(encoding="utf-8"))["value"] == 1


def test_host_native_lto_probe_is_capability_driven(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "evrptw.experiments.stage052_performance_build.shutil.which",
        lambda *_args, **_kwargs: None,
    )
    unavailable = probe_host_native_lto_support({"PATH": ""})
    assert unavailable["supported"] is False
    assert unavailable["reason"] == "compiler_not_found"

    def supported(command: tuple[str, ...], **_kwargs: object) -> SimpleNamespace:
        Path(command[-1]).write_bytes(b"binary")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(
        "evrptw.experiments.stage052_performance_build.subprocess.run",
        supported,
    )
    available = probe_host_native_lto_support(
        {"PATH": ""},
        compiler_command=("/usr/bin/c++",),
    )
    assert available["supported"] is True
    flags = available["flags"]
    assert isinstance(flags, list)
    assert "-march=native" in flags
