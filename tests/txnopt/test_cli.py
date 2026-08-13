from __future__ import annotations

import hashlib
import json
from pathlib import Path

from txnopt_evidence.cli import main


def _signed_json(path: Path, payload: object) -> None:
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    path.write_bytes(data)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{hashlib.sha256(data).hexdigest()}  {path.name}\n",
        encoding="utf-8",
    )


def test_cli_verify_and_legacy_verify(tmp_path: Path, capsys: object) -> None:
    manifest = tmp_path / "manifest.json"
    _signed_json(manifest, {"schema_version": "txnopt-test-v1"})
    assert main(["verify", str(manifest)]) == 0
    assert main(["legacy", "verify", str(manifest)]) == 0


def test_cli_run_and_replay_fail_closed_without_fallback(
    tmp_path: Path,
    capsys: object,
) -> None:
    path = tmp_path / "config.json"
    path.write_text("{}\n", encoding="utf-8")
    assert main(["run", "--config", str(path)]) == 2
    assert main(["replay", str(path)]) == 2
