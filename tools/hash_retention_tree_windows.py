"""Windows-native parallel SHA-256 verifier for one retained logical tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=range(1, 33), required=True)
    parser.add_argument("--segment", action="append", required=True)
    arguments = parser.parse_args()
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
