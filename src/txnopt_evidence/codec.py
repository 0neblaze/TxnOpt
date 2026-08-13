"""Canonical bytes, sidecars, and path-safe raw bundle verification."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def canonical_json_bytes(payload: object, *, pretty: bool = False) -> bytes:
    options: dict[str, Any] = {
        "allow_nan": False,
        "sort_keys": True,
    }
    if pretty:
        options["indent"] = 2
    else:
        options["separators"] = (",", ":")
    return (json.dumps(payload, **options) + ("\n" if pretty else "")).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_exclusive(path: Path, data: bytes) -> str:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite artifact: {path}")
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
    return sha256_bytes(data)


def write_sidecar(path: Path, digest: str) -> Path:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    write_exclusive(sidecar, f"{digest}  {path.name}\n".encode())
    return sidecar


def write_signed_json(path: Path, payload: object) -> str:
    data = canonical_json_bytes(payload, pretty=True)
    digest = write_exclusive(path, data)
    write_sidecar(path, digest)
    return digest


def verify_sidecar(path: Path) -> str:
    if path.is_symlink():
        raise ValueError(f"artifact cannot be a symlink: {path}")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"artifact must be a regular file: {path}")
    digest = sha256_file(resolved)
    sidecar = resolved.with_suffix(resolved.suffix + ".sha256")
    if not sidecar.is_file() or sidecar.is_symlink():
        raise ValueError(f"signed sidecar is missing: {sidecar}")
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    if len(fields) != 2 or fields[0] != digest or fields[1] != resolved.name:
        raise ValueError(f"signed sidecar differs for {resolved.name}")
    return digest


def read_signed_json(path: Path) -> dict[str, Any]:
    verify_sidecar(path)
    payload: object = json.loads(path.read_bytes())
    if not isinstance(payload, dict):
        raise ValueError("signed JSON must contain an object")
    return payload


def write_event_stream(
    path: Path,
    events: Sequence[Mapping[str, object]],
) -> tuple[str, str]:
    previous = "0" * 64
    lines: list[bytes] = []
    for event_id, payload in enumerate(events):
        unsigned = {
            "event_id": event_id,
            "previous_event_sha256": previous,
            "payload": dict(payload),
        }
        event_digest = sha256_bytes(canonical_json_bytes(unsigned))
        record = {**unsigned, "event_sha256": event_digest}
        lines.append(canonical_json_bytes(record) + b"\n")
        previous = event_digest
    digest = write_exclusive(path, b"".join(lines))
    write_sidecar(path, digest)
    return digest, previous


def read_event_stream(path: Path) -> tuple[tuple[dict[str, Any], ...], str]:
    verify_sidecar(path)
    previous = "0" * 64
    events: list[dict[str, Any]] = []
    for expected_id, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        record: object = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError("event record must be an object")
        event_id = record.get("event_id")
        prior = record.get("previous_event_sha256")
        payload = record.get("payload")
        claimed = record.get("event_sha256")
        if event_id != expected_id or prior != previous or not isinstance(payload, dict):
            raise ValueError(f"event chain differs at event {expected_id}")
        unsigned = {
            "event_id": event_id,
            "previous_event_sha256": prior,
            "payload": payload,
        }
        actual = sha256_bytes(canonical_json_bytes(unsigned))
        if claimed != actual:
            raise ValueError(f"event digest differs at event {expected_id}")
        events.append(payload)
        previous = actual
    if not events:
        raise ValueError("semantic event stream cannot be empty")
    return tuple(events), previous


def resolve_bundle_file(bundle: Path, relative_path: str) -> Path:
    if not relative_path or Path(relative_path).is_absolute():
        raise ValueError("manifest artifact path must be relative")
    resolved_bundle = bundle.resolve(strict=True)
    candidate = resolved_bundle / relative_path
    if candidate.is_symlink():
        raise ValueError("manifest artifact cannot be a symlink")
    resolved = candidate.resolve(strict=True)
    if resolved.parent != resolved_bundle:
        raise ValueError("manifest artifact escapes the raw bundle")
    return resolved


__all__ = [
    "canonical_json_bytes",
    "read_event_stream",
    "read_signed_json",
    "resolve_bundle_file",
    "sha256_bytes",
    "sha256_file",
    "verify_sidecar",
    "write_event_stream",
    "write_exclusive",
    "write_sidecar",
    "write_signed_json",
]
