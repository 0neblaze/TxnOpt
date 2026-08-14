from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEGACY_COMMIT = "3b0cf371759f3465c7264b85894d090004f3cf43"
LEGACY_POLICY_SHA256 = "4ceb354c6448282f621a08091a2f2e97882b54c515c1194043c0f505e04b4b7d"


def test_legacy_policy_is_the_exact_pre_txnopt_root_policy() -> None:
    frozen = ROOT / "legacy/governance/AGENTS.stage052.md"
    assert hashlib.sha256(frozen.read_bytes()).hexdigest() == LEGACY_POLICY_SHA256
    historical = subprocess.run(
        ["git", "show", f"{LEGACY_COMMIT}:AGENTS.md"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    assert frozen.read_bytes() == historical


def test_legacy_freeze_manifest_is_signed_and_honest() -> None:
    manifest = ROOT / "legacy/legacy-freeze-manifest.json"
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    sidecar = manifest.with_suffix(".json.sha256").read_text(encoding="utf-8").split()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert sidecar == [digest, manifest.name]
    assert payload["legacy_commit"] == LEGACY_COMMIT
    assert payload["verification_status"] == "freeze_not_correctness"
    assert payload["source_disposition_matches_active_source"] is False
    assert payload["protected_historical_paths_mutated"] is False
