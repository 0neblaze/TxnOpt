"""Attest the exact tracked bytes used by a native TxnOpt build."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
import tarfile
from collections.abc import Mapping
from pathlib import Path

SCHEMA_VERSION = "txnopt-native-build-attestation-v1"
_ACTIVE_PACKAGE_ROOTS = (
    "src/txnopt/",
    "src/txnopt_cases/",
    "src/txnopt_evidence/",
    "src/txnopt_legacy/",
)


def validate_scheduler_build_attestation(
    payload: Mapping[str, object],
    *,
    revision: str,
    git_tree: str,
    source_manifest_sha256: str,
    tracked_file_count: int,
    performance_profile: str | None = None,
    compiler_id: str | None = None,
    compiler_version: str | None = None,
    interprocedural_optimization: bool | None = None,
    host_native: bool | None = None,
) -> None:
    if type(payload.get("schema_version")) is not int:
        raise RuntimeError("native scheduler attestation schema type is invalid")
    if type(payload.get("tracked_file_count")) is not int:
        raise RuntimeError("native scheduler tracked-file count type is invalid")
    if type(payload.get("source_dirty")) is not bool:
        raise RuntimeError("native scheduler dirty-source flag type is invalid")
    if type(payload.get("development_override")) is not bool:
        raise RuntimeError("native scheduler development override type is invalid")
    schema_version = payload.get("schema_version")
    if schema_version not in {1, 2}:
        raise RuntimeError("native scheduler attestation schema is unsupported")
    expected: dict[str, object] = {
        "schema_version": schema_version,
        "revision": revision,
        "git_tree": git_tree,
        "source_manifest_sha256": source_manifest_sha256,
        "tracked_file_count": tracked_file_count,
        "source_dirty": False,
        "development_override": False,
        "cpp_source_kind": "git_blob_snapshot",
    }
    if schema_version == 2:
        observed_profile = payload.get("performance_profile")
        observed_compiler_id = payload.get("compiler_id")
        observed_compiler_version = payload.get("compiler_version")
        observed_ipo = payload.get("interprocedural_optimization")
        observed_host_native = payload.get("host_native")
        profile_flags = {
            "portable-o3": (False, False),
            "portable-lto": (True, False),
            "host-native-lto": (True, True),
        }
        if observed_profile not in profile_flags:
            raise RuntimeError("native scheduler performance profile is invalid")
        if not isinstance(observed_compiler_id, str) or not observed_compiler_id:
            raise RuntimeError("native scheduler compiler identity is invalid")
        if not isinstance(observed_compiler_version, str) or not observed_compiler_version:
            raise RuntimeError("native scheduler compiler version is invalid")
        if type(observed_ipo) is not bool or type(observed_host_native) is not bool:
            raise RuntimeError("native scheduler build flags are invalid")
        if (observed_ipo, observed_host_native) != profile_flags[observed_profile]:
            raise RuntimeError("native scheduler build flags contradict its profile")
        expected.update(
            {
                "performance_profile": (
                    observed_profile if performance_profile is None else performance_profile
                ),
                "compiler_id": (observed_compiler_id if compiler_id is None else compiler_id),
                "compiler_version": (
                    observed_compiler_version if compiler_version is None else compiler_version
                ),
                "interprocedural_optimization": (
                    observed_ipo
                    if interprocedural_optimization is None
                    else interprocedural_optimization
                ),
                "host_native": (observed_host_native if host_native is None else host_native),
            }
        )
    elif any(
        value is not None
        for value in (
            performance_profile,
            compiler_id,
            compiler_version,
            interprocedural_optimization,
            host_native,
        )
    ):
        raise RuntimeError("legacy scheduler attestation lacks performance identity")
    if dict(payload) != expected:
        raise RuntimeError("native scheduler build attestation does not reconcile")


def _git(root: Path, *arguments: str) -> bytes:
    return subprocess.run(
        ("git", *arguments),
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout


def _git_blob_sha1(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data, usedforsecurity=False).hexdigest()


def _working_bytes(path: Path, mode: str) -> bytes:
    if mode == "120000":
        return os.readlink(path).encode("utf-8")
    if mode == "160000":
        raise RuntimeError("native build attestation does not support submodules")
    return path.read_bytes()


def _manifest_sha256(records: list[dict[str, object]]) -> str:
    manifest_bytes = json.dumps(
        records,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(manifest_bytes).hexdigest()


def committed_source_attestation(root: Path, revision: str) -> dict[str, object]:
    """Recompute the clean source manifest from committed Git objects only."""

    root = root.resolve()
    top_level = Path(_git(root, "rev-parse", "--show-toplevel").decode("utf-8").strip()).resolve()
    if top_level != root:
        raise RuntimeError("native build source is not the exact Git top-level")
    raw_tree = _git(root, "ls-tree", "-r", "-z", revision)
    archive_bytes = _git(root, "archive", "--format=tar", revision)
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:") as archive:
        archive_members = {member.name: member for member in archive.getmembers()}
        records: list[dict[str, object]] = []
        for item in raw_tree.rstrip(b"\0").split(b"\0"):
            metadata, raw_path = item.split(b"\t", 1)
            mode, object_type, head_oid = metadata.decode("ascii").split()
            if object_type != "blob" or mode == "160000":
                raise RuntimeError("native build attestation encountered an unsupported Git entry")
            relative_path = raw_path.decode("utf-8")
            member = archive_members.get(relative_path)
            if member is None:
                raise RuntimeError("Git archive omitted a tracked source file")
            if mode == "120000":
                data = member.linkname.encode("utf-8")
            else:
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise RuntimeError("Git archive source file is unreadable")
                data = extracted.read()
            if _git_blob_sha1(data) != head_oid:
                raise RuntimeError("Git archive source bytes do not match Git blobs")
            records.append(
                {
                    "path": relative_path,
                    "mode": mode,
                    "head_oid": head_oid,
                    "working_oid": head_oid,
                    "working_sha256": hashlib.sha256(data).hexdigest(),
                }
            )
    revision_sha = _git(root, "rev-parse", revision).decode("ascii").strip()
    tree = _git(root, "rev-parse", f"{revision}^{{tree}}").decode("ascii").strip()
    return {
        "schema_version": SCHEMA_VERSION,
        "revision": revision_sha,
        "git_tree": tree,
        "source_manifest_sha256": _manifest_sha256(records),
        "tracked_file_count": len(records),
        "source_dirty": False,
        "development_override": False,
        "dirty_paths": [],
    }


def committed_wheel_project_entries(root: Path, revision: str) -> set[str]:
    """Map committed package/tool paths to their required wheel entry names."""

    raw_paths = _git(root.resolve(), "ls-tree", "-r", "--name-only", "-z", revision)
    entries: set[str] = set()
    for raw_path in raw_paths.rstrip(b"\0").split(b"\0"):
        path = raw_path.decode("utf-8")
        if path.startswith(_ACTIVE_PACKAGE_ROOTS):
            entries.add(path.removeprefix("src/"))
    if not entries:
        raise RuntimeError("committed wheel project inventory is empty")
    return entries


def committed_wheel_project_entry_sha256(root: Path, revision: str) -> dict[str, str]:
    """Return the committed Git-blob bytes required for every wheel source entry."""

    root = root.resolve()
    raw_tree = _git(
        root,
        "ls-tree",
        "-r",
        "-z",
        revision,
        "--",
        "src/txnopt",
        "src/txnopt_cases",
        "src/txnopt_evidence",
        "src/txnopt_legacy",
    )
    tracked_paths = [
        item.split(b"\t", 1)[1].decode("utf-8")
        for item in raw_tree.rstrip(b"\0").split(b"\0")
        if item
    ]
    if not tracked_paths:
        raise RuntimeError("committed wheel project inventory is empty")
    archive_bytes = _git(
        root,
        "archive",
        "--format=tar",
        revision,
        *tracked_paths,
    )
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:") as archive:
        members = {member.name: member for member in archive.getmembers()}
        entries: dict[str, str] = {}
        for item in raw_tree.rstrip(b"\0").split(b"\0"):
            if not item:
                continue
            metadata, raw_path = item.split(b"\t", 1)
            mode, object_type, head_oid = metadata.decode("ascii").split()
            if object_type != "blob" or mode == "160000":
                raise RuntimeError("wheel source inventory has an unsupported Git entry")
            path = raw_path.decode("utf-8")
            member = members.get(path)
            if member is None:
                raise RuntimeError("Git archive omitted a wheel project source file")
            if mode == "120000":
                data = member.linkname.encode("utf-8")
            else:
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise RuntimeError("Git archive wheel source file is unreadable")
                data = extracted.read()
            if _git_blob_sha1(data) != head_oid:
                raise RuntimeError("Git archive wheel source bytes do not match Git")
            wheel_path = path.removeprefix("src/") if path.startswith("src/") else path
            entries[wheel_path] = hashlib.sha256(data).hexdigest()
    if set(entries) != committed_wheel_project_entries(root, revision):
        raise RuntimeError("wheel source byte inventory does not match Git paths")
    return dict(sorted(entries.items()))


def _committed_cpp_files(root: Path, revision: str) -> dict[str, tuple[str, bytes]]:
    raw_tree = _git(root, "ls-tree", "-r", "-z", revision, "--", "cpp")
    archive_bytes = _git(root, "archive", "--format=tar", revision, "cpp")
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:") as archive:
        archive_members = {member.name: member for member in archive.getmembers()}
        files: dict[str, tuple[str, bytes]] = {}
        for item in raw_tree.rstrip(b"\0").split(b"\0"):
            if not item:
                continue
            metadata, raw_path = item.split(b"\t", 1)
            mode, object_type, head_oid = metadata.decode("ascii").split()
            if object_type != "blob" or mode == "160000":
                raise RuntimeError("native C++ snapshot has an unsupported Git entry")
            relative_path = raw_path.decode("utf-8")
            member = archive_members.get(relative_path)
            if member is None:
                raise RuntimeError("Git archive omitted a native C++ source file")
            if mode == "120000":
                data = member.linkname.encode("utf-8")
            else:
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise RuntimeError("Git archive native source file is unreadable")
                data = extracted.read()
            if _git_blob_sha1(data) != head_oid:
                raise RuntimeError("Git archive native source bytes do not match Git")
            files[relative_path] = (mode, data)
    if not files:
        raise RuntimeError("native C++ snapshot is empty")
    return files


def _snapshot_path(destination: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts or relative.parts[:1] != ("cpp",):
        raise RuntimeError("native C++ snapshot path is unsafe")
    output = destination.joinpath(*relative.parts)
    if not output.resolve(strict=False).is_relative_to(destination.resolve()):
        raise RuntimeError("native C++ snapshot path escapes its destination")
    return output


def materialize_committed_cpp(root: Path, revision: str, destination: Path) -> None:
    """Materialize compiler inputs from immutable Git blobs, never working bytes."""

    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    committed = _committed_cpp_files(root.resolve(), revision)
    expected_paths: set[Path] = set()
    for relative_path, (mode, data) in committed.items():
        output = _snapshot_path(destination, relative_path)
        expected_paths.add(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists() or output.is_symlink():
            observed = _working_bytes(output, mode)
            if observed != data:
                raise RuntimeError("existing native C++ snapshot bytes have changed")
            continue
        if mode == "120000":
            output.symlink_to(data.decode("utf-8"))
        else:
            output.write_bytes(data)
            output.chmod(0o755 if mode == "100755" else 0o644)
    observed_paths = {
        path for path in destination.rglob("*") if path.is_file() or path.is_symlink()
    }
    if observed_paths != expected_paths:
        raise RuntimeError("native C++ snapshot contains missing or unexpected files")


def verify_committed_cpp(root: Path, revision: str, destination: Path) -> None:
    destination = destination.resolve()
    committed = _committed_cpp_files(root.resolve(), revision)
    expected_paths: set[Path] = set()
    for relative_path, (mode, data) in committed.items():
        output = _snapshot_path(destination, relative_path)
        expected_paths.add(output)
        if not output.exists() and not output.is_symlink():
            raise RuntimeError("native C++ snapshot file is missing")
        if _working_bytes(output, mode) != data:
            raise RuntimeError("native C++ snapshot bytes do not match Git")
    observed_paths = {
        path for path in destination.rglob("*") if path.is_file() or path.is_symlink()
    }
    if observed_paths != expected_paths:
        raise RuntimeError("native C++ snapshot contains missing or unexpected files")


def inspect_source(root: Path, *, development_override: bool) -> dict[str, object]:
    root = root.resolve()
    top_level = Path(_git(root, "rev-parse", "--show-toplevel").decode("utf-8").strip()).resolve()
    if top_level != root:
        raise RuntimeError("native build source is not the exact Git top-level")
    object_format = _git(root, "rev-parse", "--show-object-format").decode().strip()
    if object_format != "sha1":
        raise RuntimeError("native build attestation requires a SHA-1 Git repository")
    revision = _git(root, "rev-parse", "HEAD").decode("ascii").strip()
    tree = _git(root, "rev-parse", "HEAD^{tree}").decode("ascii").strip()
    raw_tree = _git(root, "ls-tree", "-r", "-z", "HEAD")
    records: list[dict[str, object]] = []
    dirty_paths: list[str] = []
    for item in raw_tree.rstrip(b"\0").split(b"\0"):
        metadata, raw_path = item.split(b"\t", 1)
        mode, object_type, head_oid = metadata.decode("ascii").split()
        if object_type != "blob":
            raise RuntimeError("native build attestation encountered a non-blob entry")
        relative_path = raw_path.decode("utf-8")
        absolute_path = root / relative_path
        if not absolute_path.exists() and not absolute_path.is_symlink():
            dirty_paths.append(relative_path)
            working_sha256 = None
            working_oid = None
        else:
            data = _working_bytes(absolute_path, mode)
            working_sha256 = hashlib.sha256(data).hexdigest()
            working_oid = _git_blob_sha1(data)
            working_mode = (
                "120000"
                if absolute_path.is_symlink()
                else "100755"
                if os.access(absolute_path, os.X_OK)
                else "100644"
            )
            if working_oid != head_oid or working_mode != mode:
                dirty_paths.append(relative_path)
        records.append(
            {
                "path": relative_path,
                "mode": mode,
                "head_oid": head_oid,
                "working_oid": working_oid,
                "working_sha256": working_sha256,
            }
        )
    head_paths = {str(record["path"]) for record in records}
    index_paths = {
        value.decode("utf-8")
        for value in _git(root, "ls-files", "-z").rstrip(b"\0").split(b"\0")
        if value
    }
    dirty_paths.extend(sorted(index_paths ^ head_paths))
    dirty_paths = sorted(set(dirty_paths))
    if dirty_paths and not development_override:
        raise RuntimeError(
            "native build source has tracked byte changes: " + ", ".join(dirty_paths[:8])
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "revision": revision,
        "git_tree": tree,
        "source_manifest_sha256": _manifest_sha256(records),
        "tracked_file_count": len(records),
        "source_dirty": bool(dirty_paths),
        "development_override": development_override,
        "dirty_paths": dirty_paths,
    }


def _expect(payload: dict[str, object], name: str, expected: str) -> None:
    observed = payload[name]
    if str(observed).lower() != expected.lower():
        raise RuntimeError(
            f"native build attestation changed for {name}: {observed!r} != {expected!r}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("inspect", "verify", "materialize-cpp", "verify-materialized-cpp"),
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--development-override", action="store_true")
    parser.add_argument("--expected-revision")
    parser.add_argument("--expected-tree")
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--expected-source-dirty")
    parser.add_argument("--expected-development-override")
    parser.add_argument("--revision")
    parser.add_argument("--destination", type=Path)
    arguments = parser.parse_args()
    try:
        if arguments.command in ("materialize-cpp", "verify-materialized-cpp"):
            if arguments.revision is None or arguments.destination is None:
                raise RuntimeError("native C++ snapshot arguments are incomplete")
            if arguments.command == "materialize-cpp":
                materialize_committed_cpp(arguments.root, arguments.revision, arguments.destination)
            else:
                verify_committed_cpp(arguments.root, arguments.revision, arguments.destination)
            print(json.dumps({"verified": True}, sort_keys=True))
            return 0
        payload = inspect_source(
            arguments.root,
            development_override=arguments.development_override,
        )
        if arguments.command == "verify":
            required = {
                "revision": arguments.expected_revision,
                "git_tree": arguments.expected_tree,
                "source_manifest_sha256": arguments.expected_manifest_sha256,
                "source_dirty": arguments.expected_source_dirty,
                "development_override": arguments.expected_development_override,
            }
            if any(value is None for value in required.values()):
                raise RuntimeError("native build verification expectations are incomplete")
            for name, expected in required.items():
                assert expected is not None
                _expect(payload, name, expected)
    except (OSError, subprocess.CalledProcessError, RuntimeError) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
