"""Windows-native parallel SHA-256 verifier for one retained logical tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

_SEGMENT_ID = re.compile(r"[a-z][a-z0-9_]*")


@dataclass(frozen=True, slots=True)
class _Identity:
    relative_path: str
    byte_count: int
    sha256: str

    def to_dict(self) -> dict[str, int | str]:
        return {
            "relative_path": self.relative_path,
            "byte_count": self.byte_count,
            "sha256": self.sha256,
        }


def _canonical_json(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def _snapshot(path: Path) -> tuple[int, int, int, int, int]:
    stat = path.stat()
    return (
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
        stat.st_ino,
        stat.st_dev,
    )


def _hash_file(task: tuple[str, Path]) -> _Identity:
    relative, path = task
    before = _snapshot(path)
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
            byte_count += len(chunk)
    if byte_count != before[0] or _snapshot(path) != before:
        raise RuntimeError(f"retention target changed during verification: {relative}")
    return _Identity(relative, byte_count, digest.hexdigest())


def _hash_mapping_file(
    task: tuple[str, str, Path],
) -> tuple[str, _Identity]:
    logical_id, relative, path = task
    return logical_id, _hash_file((relative, path))


def _tree_identity(
    identities: dict[str, _Identity],
) -> tuple[int, int, str]:
    digest = hashlib.sha256()
    for relative in sorted(identities):
        digest.update(_canonical_json(identities[relative].to_dict()))
    return (
        len(identities),
        sum(identity.byte_count for identity in identities.values()),
        digest.hexdigest(),
    )


def _parse_segment(raw: str) -> tuple[str, PurePosixPath]:
    segment_id, separator, raw_prefix = raw.partition("=")
    prefix = PurePosixPath(raw_prefix)
    if (
        not separator
        or _SEGMENT_ID.fullmatch(segment_id) is None
        or prefix.is_absolute()
        or ".." in prefix.parts
    ):
        raise ValueError(f"invalid retention segment: {raw}")
    return segment_id, prefix


def _is_junction(path: Path) -> bool:
    checker = getattr(path, "is_junction", None)
    return bool(checker()) if checker is not None else False


def verify_tree(
    root: Path,
    *,
    workers: int,
    segments: tuple[tuple[str, PurePosixPath], ...],
) -> dict[str, object]:
    resolved = root.resolve(strict=True)
    if not resolved.is_dir() or resolved.is_symlink() or _is_junction(resolved):
        raise RuntimeError("retention target root is not a regular directory")
    files: dict[str, Path] = {}
    for path in sorted(resolved.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink() or _is_junction(path):
            raise RuntimeError(f"retention target contains a link: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise RuntimeError(f"retention target entry is unsupported: {path}")
        files[path.relative_to(resolved).as_posix()] = path
    with ThreadPoolExecutor(max_workers=workers) as executor:
        identities = {
            identity.relative_path: identity
            for identity in executor.map(_hash_file, tuple(files.items()))
        }
    segmented: dict[str, dict[str, _Identity]] = {
        segment_id: {} for segment_id, _prefix in segments
    }
    for logical_path, identity in identities.items():
        logical_parts = PurePosixPath(logical_path).parts
        matches: list[tuple[str, str]] = []
        for segment_id, prefix in segments:
            prefix_parts = () if str(prefix) == "." else prefix.parts
            if logical_parts[: len(prefix_parts)] != prefix_parts:
                continue
            relative_parts = logical_parts[len(prefix_parts) :]
            if relative_parts:
                matches.append(
                    (segment_id, PurePosixPath(*relative_parts).as_posix())
                )
        if len(matches) != 1:
            raise RuntimeError(
                "retained file does not map to exactly one segment: "
                f"{logical_path}"
            )
        segment_id, segment_relative = matches[0]
        segmented[segment_id][segment_relative] = _Identity(
            segment_relative,
            identity.byte_count,
            identity.sha256,
        )
    file_count, byte_count, tree_sha256 = _tree_identity(identities)
    return {
        "schema_version": "experiment-retention-native-tree-verification-v1",
        "verifier_pid": os.getpid(),
        "root": str(resolved),
        "workers": workers,
        "file_count": file_count,
        "byte_count": byte_count,
        "tree_sha256": tree_sha256,
        "segments": {
            segment_id: {
                "file_count": segment_identity[0],
                "byte_count": segment_identity[1],
                "tree_sha256": segment_identity[2],
            }
            for segment_id, segment_files in sorted(segmented.items())
            if (
                segment_identity := _tree_identity(segment_files)
            )
        },
    }


def verify_mappings(
    root: Path,
    *,
    workers: int,
    mappings: tuple[tuple[str, PurePosixPath], ...],
) -> dict[str, object]:
    """Hash many disjoint retained trees through one native worker pool."""

    resolved = root.resolve(strict=True)
    if not resolved.is_dir() or resolved.is_symlink() or _is_junction(resolved):
        raise RuntimeError("retention target root is not a regular directory")
    mapping_roots: dict[str, Path | None] = {}
    relative_roots: dict[str, PurePosixPath] = {}
    for logical_id, relative in mappings:
        candidate = resolved
        missing = False
        for part in relative.parts:
            candidate /= part
            if candidate.is_symlink() or _is_junction(candidate):
                raise RuntimeError(
                    f"retention target mapping is a link: {logical_id}"
                )
            if not candidate.exists():
                missing = True
                break
        if missing:
            mapping_roots[logical_id] = None
            relative_roots[logical_id] = relative
            continue
        mapping_root = candidate.resolve(strict=True)
        if (
            not mapping_root.is_dir()
            or mapping_root == resolved
            or resolved not in mapping_root.parents
        ):
            raise RuntimeError(
                f"retention target mapping escapes its root: {logical_id}"
            )
        mapping_roots[logical_id] = mapping_root
        relative_roots[logical_id] = relative
    relative_values = tuple(relative_roots.values())
    for index, first in enumerate(relative_values):
        for second in relative_values[index + 1 :]:
            if (
                first.parts == second.parts[: len(first.parts)]
                or second.parts == first.parts[: len(second.parts)]
            ):
                raise RuntimeError("retention target mappings overlap")

    tasks: list[tuple[str, str, Path]] = []
    for logical_id, current_root in mapping_roots.items():
        if current_root is None:
            continue
        for path in sorted(
            current_root.rglob("*"),
            key=lambda item: item.as_posix(),
        ):
            if path.is_symlink() or _is_junction(path):
                raise RuntimeError(
                    f"retention target contains a link: {path}"
                )
            if path.is_dir():
                continue
            if not path.is_file():
                raise RuntimeError(
                    f"retention target entry is unsupported: {path}"
                )
            tasks.append(
                (
                    logical_id,
                    path.relative_to(current_root).as_posix(),
                    path,
                )
            )
    identities: dict[str, dict[str, _Identity]] = {
        logical_id: {} for logical_id in mapping_roots
    }
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for logical_id, identity in executor.map(
            _hash_mapping_file,
            tuple(tasks),
        ):
            identities[logical_id][identity.relative_path] = identity
    return {
        "schema_version": (
            "experiment-retention-native-mapping-verification-v1"
        ),
        "verifier_pid": os.getpid(),
        "root": str(resolved),
        "workers": workers,
        "mappings": {
            logical_id: {
                "file_count": mapping_identity[0],
                "byte_count": mapping_identity[1],
                "tree_sha256": mapping_identity[2],
            }
            for logical_id, files in sorted(identities.items())
            if (mapping_identity := _tree_identity(files))
        },
    }


def _parse_mapping_request() -> tuple[
    Path,
    int,
    tuple[tuple[str, PurePosixPath], ...],
]:
    payload = json.load(sys.stdin)
    if not isinstance(payload, dict):
        raise ValueError("mapping verification request must be an object")
    raw_root = payload.get("root")
    workers = payload.get("workers")
    raw_mappings = payload.get("mappings")
    if (
        not isinstance(raw_root, str)
        or isinstance(workers, bool)
        or not isinstance(workers, int)
        or not 1 <= workers <= 32
        or not isinstance(raw_mappings, list)
        or not raw_mappings
    ):
        raise ValueError("mapping verification request is invalid")
    mappings: list[tuple[str, PurePosixPath]] = []
    for raw in raw_mappings:
        if not isinstance(raw, dict):
            raise ValueError("mapping verification entry is invalid")
        logical_id = raw.get("logical_id")
        raw_relative = raw.get("relative_path")
        if (
            not isinstance(logical_id, str)
            or not logical_id
            or not isinstance(raw_relative, str)
            or not raw_relative
            or "\\" in raw_relative
            or ":" in raw_relative
        ):
            raise ValueError("mapping verification entry is invalid")
        relative = PurePosixPath(raw_relative)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or str(relative) == "."
        ):
            raise ValueError("mapping verification entry is invalid")
        mappings.append((logical_id, relative))
    if len({logical_id for logical_id, _relative in mappings}) != len(mappings):
        raise ValueError("mapping verification logical IDs must be unique")
    return Path(raw_root), workers, tuple(mappings)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path)
    parser.add_argument("--workers", type=int, choices=range(1, 33))
    parser.add_argument("--segment", action="append")
    parser.add_argument("--mapping-stdin", action="store_true")
    arguments = parser.parse_args()
    if arguments.mapping_stdin:
        if (
            arguments.root is not None
            or arguments.workers is not None
            or arguments.segment is not None
        ):
            parser.error("--mapping-stdin cannot be combined with tree options")
        root, workers, mappings = _parse_mapping_request()
        print(
            json.dumps(
                verify_mappings(
                    root,
                    workers=workers,
                    mappings=mappings,
                ),
                sort_keys=True,
            )
        )
        return
    if (
        arguments.root is None
        or arguments.workers is None
        or arguments.segment is None
    ):
        parser.error("--root, --workers, and --segment are required")
    segments = tuple(_parse_segment(raw) for raw in arguments.segment)
    if len({segment_id for segment_id, _prefix in segments}) != len(segments):
        raise ValueError("retention segment IDs must be unique")
    print(
        json.dumps(
            verify_tree(
                arguments.root,
                workers=arguments.workers,
                segments=segments,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
