"""Build and attest the three Stage 5.2 native performance candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Final, cast

from evrptw.experiments.stage052_performance_calibration import WheelReceipt
from evrptw.stage052_atomic import publish_no_replace
from evrptw.stage052_performance import (
    BuildArtifactIdentity,
    HostPerformanceEnvelope,
    build_profile_candidates,
    detect_host_performance,
    require_clean_repository_root,
)
from tools.native_build_attestation import (
    committed_source_attestation,
    committed_wheel_project_entry_sha256,
    validate_scheduler_build_attestation,
)

BUILD_MANIFEST_SCHEMA_VERSION: Final = "stage05.2-performance-build-v1"
BUILD_FAILURE_SCHEMA_VERSION: Final = "stage05.2-performance-build-failure-v1"
_PROFILES: Final = ("portable-o3", "portable-lto", "host-native-lto")
_FORBIDDEN_FLAG_ENV: Final = (
    "CFLAGS",
    "CXXFLAGS",
    "CPPFLAGS",
    "LDFLAGS",
    "CMAKE_ARGS",
    "SKBUILD_CONFIGURE_OPTIONS",
)
_FORBIDDEN_FLAG_PATTERNS: Final = (
    "fast-math",
    "ffast-math",
    "-ofast",
    "/fp:fast",
)
_GIT_SHA1: Final = re.compile(r"^[0-9a-f]{40}$")


class PerformanceBuildError(RuntimeError):
    """Fail-closed build or attestation error."""


def probe_host_native_lto_support(
    environment: Mapping[str, str] | None = None,
    *,
    compiler_command: Sequence[str] | None = None,
) -> dict[str, object]:
    """Compile and link a minimal C++20 program with the exact native LTO flags."""

    selected_environment = dict(os.environ if environment is None else environment)
    if compiler_command is None:
        configured = selected_environment.get("CXX")
        if configured:
            command_prefix = tuple(shlex.split(configured))
        else:
            compiler = shutil.which("c++", path=selected_environment.get("PATH"))
            command_prefix = () if compiler is None else (compiler,)
    else:
        command_prefix = tuple(compiler_command)
    flags = ("-std=c++20", "-O3", "-flto", "-march=native")
    if not command_prefix:
        return {
            "supported": False,
            "reason": "compiler_not_found",
            "compiler_command": [],
            "flags": list(flags),
            "returncode": None,
            "stderr_sha256": None,
        }
    with tempfile.TemporaryDirectory(prefix="evrptw-host-native-probe-") as directory:
        output = Path(directory) / "probe"
        command = (*command_prefix, *flags, "-x", "c++", "-", "-o", str(output))
        try:
            completed = subprocess.run(
                command,
                input="int main() { return 0; }\n",
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
                env=selected_environment,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            return {
                "supported": False,
                "reason": type(error).__name__,
                "compiler_command": list(command_prefix),
                "flags": list(flags),
                "returncode": None,
                "stderr_sha256": None,
            }
        supported = completed.returncode == 0 and output.is_file()
        return {
            "supported": supported,
            "reason": "supported" if supported else "compile_or_link_rejected",
            "compiler_command": list(command_prefix),
            "flags": list(flags),
            "returncode": completed.returncode,
            "stderr_sha256": hashlib.sha256(completed.stderr.encode()).hexdigest(),
        }


def _canonical_bytes(payload: Mapping[str, object]) -> bytes:
    try:
        return (
            json.dumps(
                dict(payload),
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise PerformanceBuildError("build receipt is not canonical JSON") from error


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise PerformanceBuildError(f"cannot hash build artifact: {path}") from error
    return digest.hexdigest()


def _atomic_signed_json(path: Path, payload: Mapping[str, object]) -> str:
    data = _canonical_bytes(payload)
    digest = hashlib.sha256(data).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    sidecar = path.with_suffix(path.suffix + ".sha256")
    sidecar_temporary = sidecar.with_name(f".{sidecar.name}.{os.getpid()}.tmp")
    path_published = False
    sidecar_published = False
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        with sidecar_temporary.open("x", encoding="ascii", newline="\n") as handle:
            handle.write(digest + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        publish_no_replace(sidecar_temporary, sidecar)
        sidecar_published = True
        publish_no_replace(temporary, path)
        path_published = True
        directory_descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except OSError as error:
        temporary.unlink(missing_ok=True)
        sidecar_temporary.unlink(missing_ok=True)
        if not path_published and sidecar_published:
            sidecar.unlink(missing_ok=True)
        raise PerformanceBuildError(f"cannot publish signed receipt: {path}") from error
    return digest


def _run_text(
    command: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str] | None = None,
) -> str:
    try:
        result = subprocess.run(
            tuple(command),
            cwd=cwd,
            env=None if environment is None else dict(environment),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        detail = ""
        if isinstance(error, subprocess.CalledProcessError):
            detail = (error.stderr or error.stdout or "").strip()
        raise PerformanceBuildError(
            f"command failed: {shlex.join(command)}" + (f": {detail}" if detail else "")
        ) from error
    return result.stdout.strip()


def require_clean_revision(repository: Path) -> tuple[str, str, int]:
    root = repository.resolve(strict=True)
    revision = _run_text(("git", "rev-parse", "HEAD"), cwd=root)
    tree = _run_text(("git", "rev-parse", "HEAD^{tree}"), cwd=root)
    status = _run_text(
        ("git", "status", "--porcelain=v1", "--untracked-files=all"),
        cwd=root,
    )
    if _GIT_SHA1.fullmatch(revision) is None or _GIT_SHA1.fullmatch(tree) is None:
        raise PerformanceBuildError("Git revision/tree identity is invalid")
    if status:
        raise PerformanceBuildError("performance wheels require one clean Git commit")
    commit_epoch_raw = _run_text(("git", "show", "-s", "--format=%ct", "HEAD"), cwd=root)
    try:
        commit_epoch = int(commit_epoch_raw)
    except ValueError as error:
        raise PerformanceBuildError("Git commit timestamp is invalid") from error
    if commit_epoch <= 0:
        raise PerformanceBuildError("Git commit timestamp must be positive")
    return revision, tree, commit_epoch


def reject_ambient_build_flags(environment: Mapping[str, str]) -> None:
    for name in _FORBIDDEN_FLAG_ENV:
        raw = environment.get(name, "").strip()
        if not raw:
            continue
        lowered = raw.casefold()
        if any(pattern in lowered for pattern in _FORBIDDEN_FLAG_PATTERNS):
            raise PerformanceBuildError(f"{name} requests forbidden fast-math behavior")
        raise PerformanceBuildError(
            f"{name} must be empty; performance builds freeze their own flags"
        )


def _safe_extract_wheel(wheel: Path, destination: Path) -> tuple[str, str]:
    try:
        archive = zipfile.ZipFile(wheel)
    except (OSError, zipfile.BadZipFile) as error:
        raise PerformanceBuildError(f"invalid performance wheel: {wheel}") from error
    native_members: list[str] = []
    scheduler_members: list[str] = []
    with archive:
        for information in archive.infolist():
            member = PurePosixPath(information.filename)
            if (
                member.is_absolute()
                or ".." in member.parts
                or str(member) != information.filename.rstrip("/")
            ):
                raise PerformanceBuildError("wheel contains a non-canonical path")
            if information.is_dir():
                continue
            target = destination.joinpath(*member.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                with target.open("xb") as handle:
                    handle.write(archive.read(information))
            except OSError as error:
                raise PerformanceBuildError(
                    f"cannot extract performance wheel member: {member}"
                ) from error
            mode = (information.external_attr >> 16) & 0o777
            if mode:
                target.chmod(mode)
            if (
                member.parent == PurePosixPath("evrptw")
                and member.name.startswith("_core.")
                and member.suffix == ".so"
            ):
                native_members.append(str(member))
            if str(member) == "evrptw/_native_host_scheduler":
                scheduler_members.append(str(member))
    if len(native_members) != 1 or scheduler_members != ["evrptw/_native_host_scheduler"]:
        raise PerformanceBuildError("wheel native/scheduler inventory is invalid")
    scheduler_path = destination / scheduler_members[0]
    scheduler_path.chmod(scheduler_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return native_members[0], scheduler_members[0]


def _verify_wheel_source_inventory(
    wheel: Path,
    *,
    expected_entries: Mapping[str, str],
    native_member: str,
    scheduler_member: str,
) -> None:
    try:
        with zipfile.ZipFile(wheel) as archive:
            names = {
                name
                for name in archive.namelist()
                if not name.endswith("/") and name.startswith(("evrptw/", "tools/"))
            }
            expected_names = set(expected_entries) | {native_member, scheduler_member}
            if names != expected_names:
                raise PerformanceBuildError(
                    "wheel project inventory differs from the clean Git tree"
                )
            for name, expected_sha256 in expected_entries.items():
                observed = hashlib.sha256(archive.read(name)).hexdigest()
                if observed != expected_sha256:
                    raise PerformanceBuildError(f"wheel source member differs from Git: {name}")
    except (OSError, KeyError, zipfile.BadZipFile) as error:
        if isinstance(error, PerformanceBuildError):
            raise
        raise PerformanceBuildError("cannot verify wheel project inventory") from error


def _query_native_identity(
    *,
    python_executable: Path,
    installed_root: Path,
) -> dict[str, object]:
    script = (
        "import json, sys\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "from evrptw import _core\n"
        "print(json.dumps({\n"
        "'git_revision': _core.__build_git_revision__,\n"
        "'git_tree': _core.__build_git_tree__,\n"
        "'source_manifest_sha256': _core.__build_source_manifest_sha256__,\n"
        "'tracked_file_count': _core.__build_tracked_file_count__,\n"
        "'source_dirty': _core.__build_source_dirty__,\n"
        "'development_override': _core.__build_development_override__,\n"
        "'cpp_source_kind': _core.__build_cpp_source_kind__,\n"
        "'source_attestation_version': _core.__build_source_attestation_version__,\n"
        "'performance_profile': _core.__build_performance_profile__,\n"
        "'compiler_id': _core.__build_compiler_id__,\n"
        "'compiler_version': _core.__build_compiler_version__,\n"
        "'interprocedural_optimization': "
        "_core.__build_interprocedural_optimization__,\n"
        "'host_native': _core.__build_host_native__,\n"
        "}, sort_keys=True))\n"
    )
    output = _run_text(
        (
            str(python_executable),
            "-I",
            "-c",
            script,
            str(installed_root),
        ),
        cwd=installed_root,
    )
    try:
        payload = json.loads(output)
    except json.JSONDecodeError as error:
        raise PerformanceBuildError("native build identity is not valid JSON") from error
    if not isinstance(payload, dict):
        raise PerformanceBuildError("native build identity must be an object")
    return cast(dict[str, object], payload)


def _query_scheduler_identity(scheduler: Path) -> dict[str, object]:
    output = _run_text((str(scheduler), "--build-attestation"), cwd=scheduler.parent)
    try:
        payload = json.loads(output)
    except json.JSONDecodeError as error:
        raise PerformanceBuildError("scheduler build identity is not valid JSON") from error
    if not isinstance(payload, dict):
        raise PerformanceBuildError("scheduler build identity must be an object")
    return cast(dict[str, object], payload)


def _compile_commands_receipt(build_directory: Path) -> dict[str, object]:
    path = build_directory / "compile_commands.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PerformanceBuildError("compile_commands.json is unavailable") from error
    if not isinstance(raw, list) or not raw:
        raise PerformanceBuildError("compile_commands.json is empty")
    commands: list[str] = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise PerformanceBuildError(f"compile_commands.json entry {index} is invalid")
        command = entry.get("command")
        arguments = entry.get("arguments")
        if isinstance(command, str) and command:
            normalized = command
        elif (
            isinstance(arguments, list)
            and arguments
            and all(isinstance(item, str) for item in arguments)
        ):
            normalized = shlex.join(cast(list[str], arguments))
        else:
            raise PerformanceBuildError(f"compile_commands.json entry {index} has no command")
        lowered = normalized.casefold()
        if any(pattern in lowered for pattern in _FORBIDDEN_FLAG_PATTERNS):
            raise PerformanceBuildError("compile command enables forbidden fast-math")
        commands.append(normalized)
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "command_count": len(commands),
        "commands": commands,
    }


def _build_profile(
    *,
    repository: Path,
    output_root: Path,
    python_executable: Path,
    profile: str,
    host: HostPerformanceEnvelope,
    revision: str,
    tree: str,
    source_attestation: Mapping[str, object],
    source_entries: Mapping[str, str],
    commit_epoch: int,
    base_environment: Mapping[str, str],
    host_native_supported: bool,
) -> tuple[WheelReceipt, dict[str, object]]:
    profile_root = output_root / profile
    wheel_directory = profile_root / "wheel"
    build_directory = profile_root / "cmake-build"
    installed_root = profile_root / "installed"
    wheel_directory.mkdir(parents=True)
    installed_root.mkdir(parents=True)
    environment = dict(base_environment)
    environment.update(
        {
            "CMAKE_ARGS": " ".join(
                (
                    f"-DEVRPTW_PERFORMANCE_PROFILE={profile}",
                    "-DEVRPTW_SANITIZER=none",
                    "-DEVRPTW_ALLOW_DIRTY_BUILD=OFF",
                    "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
                )
            ),
            "CMAKE_BUILD_PARALLEL_LEVEL": str(len(host.allowed_cpu_ids)),
            "PIP_NO_CACHE_DIR": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "SOURCE_DATE_EPOCH": str(commit_epoch),
        }
    )
    command = (
        str(python_executable),
        "-m",
        "pip",
        "wheel",
        "--no-deps",
        "--no-build-isolation",
        "--no-cache-dir",
        "--wheel-dir",
        str(wheel_directory),
        "--config-settings",
        f"build-dir={build_directory}",
        str(repository),
    )
    log_path = profile_root / "build.log"
    started = time.perf_counter()
    try:
        with log_path.open("x", encoding="utf-8", newline="\n") as log:
            process = subprocess.run(
                command,
                cwd=repository,
                env=environment,
                check=False,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
    except OSError as error:
        raise PerformanceBuildError(f"cannot start {profile} wheel build") from error
    build_seconds = time.perf_counter() - started
    if process.returncode != 0:
        raise PerformanceBuildError(
            f"{profile} wheel build failed with exit code {process.returncode}"
        )
    wheels = tuple(wheel_directory.glob("*.whl"))
    if len(wheels) != 1:
        raise PerformanceBuildError(f"{profile} did not produce exactly one wheel")
    wheel = wheels[0].resolve(strict=True)
    native_member, scheduler_member = _safe_extract_wheel(wheel, installed_root)
    _verify_wheel_source_inventory(
        wheel,
        expected_entries=source_entries,
        native_member=native_member,
        scheduler_member=scheduler_member,
    )
    native = (installed_root / native_member).resolve(strict=True)
    scheduler = (installed_root / scheduler_member).resolve(strict=True)
    native_identity = _query_native_identity(
        python_executable=python_executable,
        installed_root=installed_root,
    )
    expected_profile_flags = {
        "portable-o3": (False, False),
        "portable-lto": (True, False),
        "host-native-lto": (True, True),
    }
    expected_native_identity = {
        "git_revision": revision,
        "git_tree": tree,
        "source_manifest_sha256": source_attestation["source_manifest_sha256"],
        "tracked_file_count": source_attestation["tracked_file_count"],
        "source_dirty": False,
        "development_override": False,
        "cpp_source_kind": "git_blob_snapshot",
        "source_attestation_version": 1,
        "performance_profile": profile,
        "compiler_id": native_identity.get("compiler_id"),
        "compiler_version": native_identity.get("compiler_version"),
        "interprocedural_optimization": expected_profile_flags[profile][0],
        "host_native": expected_profile_flags[profile][1],
    }
    if (
        not isinstance(native_identity.get("compiler_id"), str)
        or not native_identity["compiler_id"]
        or not isinstance(native_identity.get("compiler_version"), str)
        or not native_identity["compiler_version"]
        or native_identity != expected_native_identity
    ):
        raise PerformanceBuildError(f"{profile} native build identity does not reconcile")
    scheduler_identity = _query_scheduler_identity(scheduler)
    validate_scheduler_build_attestation(
        scheduler_identity,
        revision=revision,
        git_tree=tree,
        source_manifest_sha256=cast(
            str,
            source_attestation["source_manifest_sha256"],
        ),
        tracked_file_count=cast(int, source_attestation["tracked_file_count"]),
        performance_profile=profile,
        compiler_id=cast(str, native_identity["compiler_id"]),
        compiler_version=cast(str, native_identity["compiler_version"]),
        interprocedural_optimization=expected_profile_flags[profile][0],
        host_native=expected_profile_flags[profile][1],
    )
    candidate = next(
        item
        for item in build_profile_candidates(
            compiler=cast(str, native_identity["compiler_id"]),
            cpu_features=host.cpu_features,
            host_native_supported=host_native_supported,
        )
        if item.name == profile
    )
    identity = BuildArtifactIdentity(
        git_revision=revision,
        git_tree=tree,
        source_manifest_sha256=cast(
            str,
            source_attestation["source_manifest_sha256"],
        ),
        wheel_sha256=_sha256_file(wheel),
        native_sha256=_sha256_file(native),
        scheduler_sha256=_sha256_file(scheduler),
        compiler_version=cast(str, native_identity["compiler_version"]),
        flags=candidate.flags,
        cpu_feature_mask=host.cpu_features,
    )
    receipt_path = profile_root / "wheel_receipt.json"
    receipt = WheelReceipt(
        build_profile=profile,
        artifact_identity=identity,
        wheel_path=wheel,
        native_path=native,
        scheduler_path=scheduler,
        compiler_id=cast(str, native_identity["compiler_id"]),
        source_receipt_path=receipt_path,
    )
    receipt.verify_files()
    receipt_sha256 = _atomic_signed_json(receipt_path, receipt.to_dict())
    compile_commands = _compile_commands_receipt(build_directory)
    profile_manifest = {
        "build_profile": profile,
        "build_seconds": build_seconds,
        "command": list(command),
        "environment": {
            name: environment[name]
            for name in (
                "CMAKE_ARGS",
                "CMAKE_BUILD_PARALLEL_LEVEL",
                "PIP_NO_CACHE_DIR",
                "PYTHONDONTWRITEBYTECODE",
                "SOURCE_DATE_EPOCH",
            )
        },
        "build_log_path": str(log_path),
        "build_log_sha256": _sha256_file(log_path),
        "compile_commands": compile_commands,
        "wheel_receipt_path": str(receipt_path),
        "wheel_receipt_sha256": receipt_sha256,
        "native_build_identity": native_identity,
        "scheduler_build_identity": scheduler_identity,
    }
    return receipt, profile_manifest


def build_performance_wheels(
    *,
    repository: Path,
    output_root: Path,
    python_executable: Path = Path(sys.executable),
    environment: Mapping[str, str] | None = None,
    host: HostPerformanceEnvelope | None = None,
) -> tuple[Path, ...]:
    root = repository.resolve(strict=True)
    output = output_root.resolve()
    if output.exists():
        raise PerformanceBuildError("performance build output namespace already exists")
    try:
        output.relative_to(root)
    except ValueError:
        pass
    else:
        raise PerformanceBuildError(
            "performance build output must be outside the clean source repository"
        )
    revision, tree, commit_epoch = require_clean_revision(root)
    base_environment = dict(os.environ if environment is None else environment)
    reject_ambient_build_flags(base_environment)
    detected_host = detect_host_performance() if host is None else host
    host_native_probe = probe_host_native_lto_support(base_environment)
    host_native_supported = host_native_probe["supported"] is True
    candidates = build_profile_candidates(
        detected_host,
        host_native_supported=host_native_supported,
    )
    profiles = tuple(item.name for item in candidates)
    expected_profiles = _PROFILES if host_native_supported else _PROFILES[:2]
    if profiles != expected_profiles:
        raise PerformanceBuildError("performance build profile capability selection is invalid")
    source_attestation = committed_source_attestation(root, revision)
    source_entries = committed_wheel_project_entry_sha256(root, revision)
    output.mkdir(parents=True)
    receipts: list[Path] = []
    profile_manifests: list[dict[str, object]] = []
    try:
        for profile in profiles:
            _, profile_manifest = _build_profile(
                repository=root,
                output_root=output,
                python_executable=python_executable.resolve(strict=True),
                profile=profile,
                host=detected_host,
                revision=revision,
                tree=tree,
                source_attestation=source_attestation,
                source_entries=source_entries,
                commit_epoch=commit_epoch,
                base_environment=base_environment,
                host_native_supported=host_native_supported,
            )
            receipt_path = output / profile / "wheel_receipt.json"
            receipts.append(receipt_path)
            profile_manifests.append(profile_manifest)
        manifest = {
            "schema_version": BUILD_MANIFEST_SCHEMA_VERSION,
            "revision": revision,
            "git_tree": tree,
            "source_manifest_sha256": source_attestation["source_manifest_sha256"],
            "tracked_file_count": source_attestation["tracked_file_count"],
            "host": detected_host.to_dict(),
            "host_native_lto_probe": host_native_probe,
            "selected_profiles": list(profiles),
            "profiles": profile_manifests,
            "formal_started": False,
            "cuda_started": False,
            "attempt08_started": False,
        }
        _atomic_signed_json(output / "build_manifest.json", manifest)
    except BaseException as error:
        failure = {
            "schema_version": BUILD_FAILURE_SCHEMA_VERSION,
            "revision": revision,
            "git_tree": tree,
            "error_type": type(error).__name__,
            "error": str(error),
            "completed_receipts": [str(path) for path in receipts],
            "formal_started": False,
            "cuda_started": False,
            "attempt08_started": False,
        }
        _atomic_signed_json(output / "build_failure.json", failure)
        raise
    return tuple(receipts)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the supported clean Stage 5.2 performance wheels."
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        required=True,
        help="Explicit clean ext4 Git worktree used for source identity",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    arguments = parser.parse_args(argv)
    resolved_repository = require_clean_repository_root(arguments.repository_root)
    receipts = build_performance_wheels(
        repository=resolved_repository,
        output_root=arguments.output_root,
        python_executable=arguments.python,
    )
    print(
        json.dumps(
            {"wheel_receipts": [str(path) for path in receipts]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "PerformanceBuildError",
    "build_performance_wheels",
    "main",
    "probe_host_native_lto_support",
    "reject_ambient_build_flags",
    "require_clean_revision",
)
