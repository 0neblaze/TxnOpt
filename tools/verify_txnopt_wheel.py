"""Fail-closed verifier for the internal TxnOpt 0.1.0a1 wheel surface."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

_ALLOWED_ROOTS = {
    "txnopt",
    "txnopt_cases",
    "txnopt_evidence",
    "txnopt_legacy",
    "txnopt-0.1.0a1.dist-info",
}
_DIST_INFO = "txnopt-0.1.0a1.dist-info"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_wheel(path: Path) -> dict[str, object]:
    wheel = path.resolve(strict=True)
    with ZipFile(wheel) as archive:
        names = tuple(archive.namelist())
        unexpected = sorted(
            name for name in names if PurePosixPath(name).parts[0] not in _ALLOWED_ROOTS
        )
        if unexpected:
            raise RuntimeError(f"wheel contains unexpected roots: {unexpected}")
        metadata = archive.read(f"{_DIST_INFO}/METADATA").decode("utf-8")
        if "\nName: txnopt\n" not in metadata or "\nVersion: 0.1.0a1\n" not in metadata:
            raise RuntimeError("wheel metadata identity is not txnopt 0.1.0a1")
        entry_points = archive.read(f"{_DIST_INFO}/entry_points.txt").decode("utf-8")
        if entry_points.strip() != "[console_scripts]\ntxnopt = txnopt_evidence.cli:main":
            raise RuntimeError("wheel console scripts differ from the TxnOpt CLI")
        native_entries = [
            name for name in names if re.fullmatch(r"txnopt/_native[^/]*\.(?:so|pyd)", name)
        ]
        if len(native_entries) != 1:
            raise RuntimeError("wheel must contain exactly one txnopt._native extension")
        if any("/_core" in name or "native_host_scheduler" in name for name in names):
            raise RuntimeError("wheel contains a frozen native ABI artifact")
        for name in names:
            if not name.endswith((".py", ".pyi")):
                continue
            source = archive.read(name).decode("utf-8")
            if re.search(r"(?:from|import)\s+evrptw(?:\.|\s|$)", source):
                raise RuntimeError(f"active wheel imports frozen namespace: {name}")
            if "stage05.2" in source:
                raise RuntimeError(f"active wheel contains a frozen schema identity: {name}")
    return {
        "schema_version": "txnopt-wheel-verification-v1",
        "status": "PASS",
        "wheel": str(wheel),
        "wheel_sha256": _sha256(wheel),
        "entry_count": len(names),
        "native_extension": native_entries[0],
        "fallback_count": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    result = verify_wheel(parser.parse_args().wheel)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
