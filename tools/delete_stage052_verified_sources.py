"""Delete only independently verified Stage 5.2 migration source trees."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Final

from hash_retention_tree_windows import verify_mappings  # type: ignore[import-not-found]

_CONFIRMATION: Final = "确认"
_MANIFEST_SCHEMA: Final = "stage052-source-deletion-candidates-v1"
_EMPTY_SHA256: Final = hashlib.sha256(b"").hexdigest()


@dataclass(frozen=True, slots=True)
class Candidate:
    logical_id: str
    root_alias: str
    root: Path
    path: Path
    relative_path: PurePosixPath
    file_count: int
    byte_count: int
    tree_sha256: str


def _canonical_json(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_signed_json(path: Path, expected_sha256: str) -> dict[str, object]:
    sidecar = path.with_suffix(f"{path.suffix}.sha256")
    if not sidecar.is_file():
        raise RuntimeError(f"signed sidecar is missing: {sidecar}")
    sidecar_digest = sidecar.read_text(encoding="ascii").split()[0]
    observed = _sha256(path)
    if observed != expected_sha256 or observed != sidecar_digest:
        raise RuntimeError("deletion manifest SHA-256 does not match")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("deletion manifest must be an object")
    return payload


def _write_signed_json(path: Path, payload: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(_canonical_json(payload))
    digest = _sha256(temporary)
    os.replace(temporary, path)
    sidecar = path.with_suffix(f"{path.suffix}.sha256")
    sidecar_temporary = sidecar.with_name(
        f".{sidecar.name}.{os.getpid()}.tmp"
    )
    sidecar_temporary.write_text(
        f"{digest}  {path.name}\n",
        encoding="ascii",
    )
    os.replace(sidecar_temporary, sidecar)
    return digest


def _safe_candidate(
    raw: object,
    *,
    roots: dict[str, Path],
) -> Candidate:
    if not isinstance(raw, dict):
        raise RuntimeError("deletion candidate is invalid")
    logical_id = raw.get("logical_id")
    root_alias = raw.get("source_root_alias")
    source_path = raw.get("source_path")
    file_count = raw.get("file_count")
    byte_count = raw.get("byte_count")
    tree_sha256 = raw.get("tree_sha256")
    if (
        not isinstance(logical_id, str)
        or not logical_id
        or not isinstance(root_alias, str)
        or root_alias not in roots
        or not isinstance(source_path, str)
        or not source_path
        or isinstance(file_count, bool)
        or not isinstance(file_count, int)
        or file_count < 0
        or isinstance(byte_count, bool)
        or not isinstance(byte_count, int)
        or byte_count < 0
        or not isinstance(tree_sha256, str)
        or len(tree_sha256) != 64
        or raw.get("deletion_authorized") is not False
    ):
        raise RuntimeError("deletion candidate identity is invalid")
    root = roots[root_alias].resolve(strict=True)
    path = Path(source_path)
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise RuntimeError(
            f"deletion candidate escapes its root: {source_path}"
        ) from error
    if not relative.parts or ".." in relative.parts:
        raise RuntimeError(f"deletion candidate is unsafe: {source_path}")
    rebuilt = root.joinpath(*relative.parts)
    if rebuilt != path:
        raise RuntimeError(f"deletion candidate is not canonical: {source_path}")
    return Candidate(
        logical_id=logical_id,
        root_alias=root_alias,
        root=root,
        path=path,
        relative_path=PurePosixPath(*relative.parts),
        file_count=file_count,
        byte_count=byte_count,
        tree_sha256=tree_sha256,
    )


def load_candidates(
    manifest_path: Path,
    *,
    expected_sha256: str,
    roots: dict[str, Path],
) -> tuple[dict[str, object], tuple[Candidate, ...]]:
    payload = _load_signed_json(manifest_path, expected_sha256)
    raw_candidates = payload.get("candidates")
    if (
        payload.get("schema_version") != _MANIFEST_SCHEMA
        or payload.get("source_deletion_authorized") is not False
        or payload.get("requires_literal_confirmation") != _CONFIRMATION
        or payload.get("rollback_after_source_deletion")
        != "none_single_media_archive"
        or not isinstance(raw_candidates, list)
        or not raw_candidates
    ):
        raise RuntimeError("deletion manifest governance fields are invalid")
    candidates = tuple(
        _safe_candidate(raw, roots=roots) for raw in raw_candidates
    )
    paths = tuple(candidate.path for candidate in candidates)
    logical_ids = tuple(candidate.logical_id for candidate in candidates)
    if (
        len(set(paths)) != len(paths)
        or len(set(logical_ids)) != len(logical_ids)
        or payload.get("candidate_count") != len(candidates)
        or payload.get("candidate_bytes")
        != sum(candidate.byte_count for candidate in candidates)
    ):
        raise RuntimeError("deletion manifest totals or identities differ")
    sorted_paths = sorted(path.as_posix().rstrip("/") for path in paths)
    for index, first in enumerate(sorted_paths):
        for second in sorted_paths[index + 1 :]:
            if second.startswith(f"{first}/"):
                raise RuntimeError("deletion candidates overlap")
    return payload, candidates


def _windows_path(path: Path) -> str:
    resolved = path.resolve()
    parts = resolved.parts
    if (
        len(parts) < 4
        or parts[0] != "/"
        or parts[1] != "mnt"
        or len(parts[2]) != 1
        or not parts[2].isalpha()
    ):
        raise RuntimeError(f"path is not a mounted Windows drive: {path}")
    return f"{parts[2].upper()}:\\" + "\\".join(parts[3:])


def _native_identities(
    *,
    root: Path,
    candidates: tuple[Candidate, ...],
    workers: int,
) -> dict[str, tuple[int, int, str]]:
    helper = Path(__file__).with_name("hash_retention_tree_windows.py")
    python_windows = shutil.which("python.exe")
    if python_windows is None or not helper.is_file():
        raise RuntimeError("Windows-native deletion preverification unavailable")
    request = {
        "root": _windows_path(root),
        "workers": workers,
        "mappings": [
            {
                "logical_id": candidate.logical_id,
                "relative_path": candidate.relative_path.as_posix(),
            }
            for candidate in candidates
        ],
    }
    completed = subprocess.run(
        (
            python_windows,
            _windows_path(helper),
            "--mapping-stdin",
        ),
        input=json.dumps(request),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Windows-native deletion preverification failed: "
            f"{completed.stderr.strip()}"
        )
    payload = json.loads(completed.stdout)
    raw_mappings = payload.get("mappings")
    if not isinstance(raw_mappings, dict):
        raise RuntimeError("Windows-native deletion identities are invalid")
    result: dict[str, tuple[int, int, str]] = {}
    for candidate in candidates:
        raw = raw_mappings.get(candidate.logical_id)
        result[candidate.logical_id] = _parse_identity(
            raw,
            description=f"Windows-native identity: {candidate.logical_id}",
        )
    return result


def _parse_identity(
    raw: object,
    *,
    description: str,
) -> tuple[int, int, str]:
    if not isinstance(raw, dict):
        raise RuntimeError(f"{description} is missing")
    file_count = raw.get("file_count")
    byte_count = raw.get("byte_count")
    tree_sha256 = raw.get("tree_sha256")
    if (
        isinstance(file_count, bool)
        or not isinstance(file_count, int)
        or file_count < 0
        or isinstance(byte_count, bool)
        or not isinstance(byte_count, int)
        or byte_count < 0
        or not isinstance(tree_sha256, str)
        or len(tree_sha256) != 64
    ):
        raise RuntimeError(f"{description} is invalid")
    return file_count, byte_count, tree_sha256


def _posix_identities(
    *,
    root: Path,
    candidates: tuple[Candidate, ...],
    workers: int,
) -> dict[str, tuple[int, int, str]]:
    payload = verify_mappings(
        root,
        workers=workers,
        mappings=tuple(
            (candidate.logical_id, candidate.relative_path)
            for candidate in candidates
        ),
    )
    raw_mappings = payload.get("mappings")
    if not isinstance(raw_mappings, dict):
        raise RuntimeError("POSIX deletion identities are invalid")
    result: dict[str, tuple[int, int, str]] = {}
    for candidate in candidates:
        raw = raw_mappings.get(candidate.logical_id)
        result[candidate.logical_id] = _parse_identity(
            raw,
            description=f"POSIX identity: {candidate.logical_id}",
        )
    return result


def verify_candidates(
    candidates: tuple[Candidate, ...],
    *,
    workers: int,
) -> None:
    grouped = {
        alias: tuple(
            candidate
            for candidate in candidates
            if candidate.root_alias == alias
        )
        for alias in ("d_archive", "wsl_staging")
    }
    with ThreadPoolExecutor(max_workers=2) as executor:
        d_future = executor.submit(
            _native_identities,
            root=grouped["d_archive"][0].root,
            candidates=grouped["d_archive"],
            workers=workers,
        )
        wsl_future = executor.submit(
            _posix_identities,
            root=grouped["wsl_staging"][0].root,
            candidates=grouped["wsl_staging"],
            workers=workers,
        )
        identities = {
            **d_future.result(),
            **wsl_future.result(),
        }
    for candidate in candidates:
        expected = (
            candidate.file_count,
            candidate.byte_count,
            candidate.tree_sha256,
        )
        if identities.get(candidate.logical_id) != expected:
            raise RuntimeError(
                "source changed after deletion manifest verification: "
                f"{candidate.logical_id}"
            )


def delete_candidates(
    candidates: tuple[Candidate, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    deleted: list[str] = []
    preexisting_absent: list[str] = []
    for index, candidate in enumerate(candidates, start=1):
        if not candidate.path.exists():
            if (
                candidate.file_count != 0
                or candidate.byte_count != 0
                or candidate.tree_sha256 != _EMPTY_SHA256
            ):
                raise RuntimeError(
                    f"nonempty deletion source disappeared: {candidate.path}"
                )
            preexisting_absent.append(str(candidate.path))
        else:
            if not candidate.path.is_dir() or candidate.path.is_symlink():
                raise RuntimeError(
                    f"deletion source is no longer a regular directory: "
                    f"{candidate.path}"
                )
            shutil.rmtree(candidate.path)
            if candidate.path.exists():
                raise RuntimeError(
                    f"deletion source still exists: {candidate.path}"
                )
            deleted.append(str(candidate.path))
        if index % 25 == 0 or index == len(candidates):
            print(
                json.dumps(
                    {
                        "event": "source_deletion_progress",
                        "completed": index,
                        "total": len(candidates),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return tuple(deleted), tuple(preexisting_absent)


def _active_writer_commands() -> tuple[str, ...]:
    blocked_tokens = (
        "migrate_stage052_retention_v2.py",
        "stage052_campaign",
        "stage052_producer",
        "stage052_reviewer",
    )
    commands: list[str] = []
    for raw_pid in Path("/proc").iterdir():
        if not raw_pid.name.isdigit() or int(raw_pid.name) == os.getpid():
            continue
        try:
            command = (
                (raw_pid / "cmdline")
                .read_bytes()
                .replace(b"\0", b" ")
                .decode(errors="replace")
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if any(token in command for token in blocked_tokens):
            commands.append(command)
    return tuple(commands)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--confirm-literal", required=True)
    parser.add_argument("--d-root", type=Path, required=True)
    parser.add_argument("--wsl-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=range(1, 33), default=32)
    arguments = parser.parse_args()
    if arguments.confirm_literal != _CONFIRMATION:
        raise RuntimeError("literal deletion confirmation is missing")
    roots = {
        "d_archive": arguments.d_root,
        "wsl_staging": arguments.wsl_root,
    }
    payload, candidates = load_candidates(
        arguments.manifest,
        expected_sha256=arguments.expected_sha256,
        roots=roots,
    )
    writers = _active_writer_commands()
    if writers:
        raise RuntimeError(f"active Stage 5.2 writers detected: {writers}")
    print(
        json.dumps(
            {
                "event": "source_deletion_preverification_started",
                "candidate_count": len(candidates),
                "candidate_bytes": sum(
                    candidate.byte_count for candidate in candidates
                ),
                "workers_per_volume": arguments.workers,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    started = time.perf_counter()
    verify_candidates(candidates, workers=arguments.workers)
    verified_at = datetime.now(UTC).isoformat()
    print(
        json.dumps(
            {
                "event": "source_deletion_preverification_passed",
                "candidate_count": len(candidates),
                "elapsed_seconds": time.perf_counter() - started,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    deleted, preexisting_absent = delete_candidates(candidates)
    remaining = tuple(
        str(candidate.path)
        for candidate in candidates
        if candidate.path.exists()
    )
    if remaining:
        raise RuntimeError(f"approved deletion sources remain: {remaining}")
    receipt_path = arguments.manifest.with_name(
        "source-deletion-execution-receipt.json"
    )
    receipt_sha256 = _write_signed_json(
        receipt_path,
        {
            "schema_version": "stage052-source-deletion-execution-v1",
            "migration_id": payload.get("migration_id"),
            "created_at_utc": datetime.now(UTC).isoformat(),
            "manifest_path": str(arguments.manifest),
            "manifest_sha256": arguments.expected_sha256,
            "literal_confirmation": _CONFIRMATION,
            "predelete_verification_passed_at_utc": verified_at,
            "candidate_count": len(candidates),
            "candidate_bytes": sum(
                candidate.byte_count for candidate in candidates
            ),
            "deleted_path_count": len(deleted),
            "preexisting_absent_path_count": len(preexisting_absent),
            "postdelete_absent_path_count": len(candidates),
            "source_deletion_executed": True,
            "rollback_after_source_deletion": "none_single_media_archive",
            "status": "passed",
        },
    )
    print(
        json.dumps(
            {
                "event": "source_deletion_completed",
                "receipt": str(receipt_path),
                "receipt_sha256": receipt_sha256,
                "deleted_path_count": len(deleted),
                "preexisting_absent_path_count": len(preexisting_absent),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
