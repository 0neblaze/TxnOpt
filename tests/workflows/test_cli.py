from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from txnopt_evidence.cli import main


def _signed_json(path: Path, payload: object) -> None:
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    path.write_bytes(data)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{hashlib.sha256(data).hexdigest()}  {path.name}\n",
        encoding="utf-8",
    )


def test_cli_legacy_compatibility_rejects_unregistered_signed_json(
    tmp_path: Path,
    capsys: object,
) -> None:
    manifest = tmp_path / "manifest.json"
    _signed_json(manifest, {"schema_version": "txnopt-test-v1"})
    assert main(["verify", str(manifest), "--legacy-compatibility"]) == 2


def test_cli_run_fails_closed_without_fallback(
    tmp_path: Path,
    capsys: object,
) -> None:
    path = tmp_path / "config.json"
    path.write_text("{}\n", encoding="utf-8")
    assert main(["run", "--config", str(path)]) == 2


def test_cli_level1_plan_dispatches_to_packaged_materializer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import txnopt_evidence.campaign as campaign

    destination = tmp_path / "plan"
    expected_manifest = destination / "manifest.json"
    calls: dict[str, object] = {}

    def fake_materialize(protocol: Path, catalog: Path, **kwargs: object) -> Path:
        calls.update(protocol=protocol, catalog=catalog, **kwargs)
        return expected_manifest

    monkeypatch.setattr(campaign, "materialize_level1_plan", fake_materialize)
    assert main(
        [
            "plan",
            "--protocol",
            str(tmp_path / "protocol.json"),
            "--catalog",
            str(tmp_path / "catalog.json"),
            "--destination",
            str(destination),
            "--raw-output-root",
            str(tmp_path / "raw"),
            "--build-manifest",
            str(tmp_path / "build.json"),
            "--fixed-work",
            "1200",
            "--fixed-time-seconds",
            "3",
            "--max-rounds",
            "10",
            "--evrptw-max-candidates",
            "64",
            "--rcpsp-max-candidates",
            "64",
            "--attempt",
            "24",
        ]
    ) == 0
    output = json.loads(capsys.readouterr().out)
    assert output == {
        "schema_version": "txnopt-level1-plan-command-v1",
        "status": "planned",
        "manifest_path": str(expected_manifest),
    }
    assert calls["fixed_work"] == 1200
    assert calls["attempt"] == 24


def test_cli_level1_preflight_and_review_dispatch_to_packaged_facades(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import txnopt_evidence.level1_campaign_reviewer as reviewer
    import txnopt_evidence.level1_campaign_runner as runner

    plan = tmp_path / "plan.json"
    analysis = tmp_path / "analysis.json"
    wheel = tmp_path / "txnopt.whl"
    python = tmp_path / "python"
    execution = tmp_path / "execution.json"
    review_root = tmp_path / "review"
    review_receipt = tmp_path / "review-receipt.json"
    preflight_calls: dict[str, object] = {}
    review_calls: dict[str, object] = {}

    def fake_preflight(
        plan_path: Path,
        analysis_path: Path,
        **kwargs: object,
    ) -> dict[str, object]:
        preflight_calls.update(plan_path=plan_path, analysis_path=analysis_path, **kwargs)
        return {"status": "PASS_NOT_AUTHORIZED_TO_EXECUTE", "config_count": 1}

    def fake_review(
        plan_path: Path,
        analysis_path: Path,
        **kwargs: object,
    ) -> dict[str, object]:
        review_calls.update(plan_path=plan_path, analysis_path=analysis_path, **kwargs)
        return {"status": "NOT_READY", "review_pass_count": 0}

    monkeypatch.setattr(runner, "preflight_campaign", fake_preflight)
    monkeypatch.setattr(reviewer, "review_campaign", fake_review)

    assert main(
        [
            "preflight",
            "--plan-manifest",
            str(plan),
            "--analysis-protocol",
            str(analysis),
            "--python",
            str(python),
            "--wheel",
            str(wheel),
        ]
    ) == 0
    preflight_output = json.loads(capsys.readouterr().out)
    assert preflight_output["status"] == "PASS_NOT_AUTHORIZED_TO_EXECUTE"
    assert preflight_calls["python"] == python
    assert preflight_calls["wheel"] == wheel

    assert main(
        [
            "review",
            "--plan-manifest",
            str(plan),
            "--analysis-protocol",
            str(analysis),
            "--execution-receipt",
            str(execution),
            "--python",
            str(python),
            "--wheel",
            str(wheel),
            "--review-root",
            str(review_root),
            "--review-receipt",
            str(review_receipt),
            "--review-workers",
            "2",
            "--per-run-timeout-seconds",
            "12",
        ]
    ) == 0
    review_output = json.loads(capsys.readouterr().out)
    assert review_output["status"] == "NOT_READY"
    assert review_calls["review_workers"] == 2
    assert review_calls["per_run_timeout_seconds"] == 12.0
