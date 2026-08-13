"""Strict read-only reader for historical signed JSON receipts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class LegacyReceiptReader:
    """Verify a frozen JSON receipt and its conventional SHA-256 sidecar."""

    def read(self, path: Path) -> dict[str, Any]:
        resolved = path.resolve(strict=True)
        if not resolved.is_file() or resolved.is_symlink():
            raise ValueError("legacy receipt must be a regular non-symlink file")
        data = resolved.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        sidecar = resolved.with_suffix(resolved.suffix + ".sha256")
        if not sidecar.is_file() or sidecar.is_symlink():
            raise ValueError("legacy receipt SHA-256 sidecar is missing")
        fields = sidecar.read_text(encoding="utf-8").strip().split()
        if len(fields) != 2 or fields[0] != digest or fields[1] != resolved.name:
            raise ValueError("legacy receipt SHA-256 sidecar differs")
        payload = json.loads(data)
        if not isinstance(payload, dict):
            raise ValueError("legacy receipt must contain a JSON object")
        return payload
