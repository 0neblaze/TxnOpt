from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from txnopt_evidence.codec import canonical_json_bytes, read_signed_json, write_signed_json
from txnopt_evidence.level1_calibration import (
    CalibrationSelection,
    _execution_python_path,
    _load_calibration_plan,
    _materialize_calibration_plan,
    _protocol_calibration_schema,
    _run,
    materialize_attempt27_v2_correction,
)


def _build_manifest(tmp_path: Path, number: int) -> Path:
    statuses = {
        16: "BUILD_COMPLETE_TENCENT_CLOUD_CUTOVER_NOT_LEVEL1_READY",
        18: "BUILD_COMPLETE_TENCENT_PRE_CLOUD_SUCCESSOR_NOT_LEVEL1_READY",
    }
    build = tmp_path / f"build{number}.json"
    write_signed_json(
        build,
        {
            "schema_version": "txnopt-level1-build-manifest-v1",
            "run_label": f"txnopt_level1_build_attempt{number}",
            "status": statuses[number],
            "formal_successor": {
                "prior_review_binding_status": "PRIOR_SOURCE_ONLY",
                "successor_status": f"REVIEW_PENDING_BUILD{number}",
                "independent_successor_review_completed": False,
                "level1_formal_gate_passed": False,
            },
            "producer": {
                "revision": "a" * 40,
                "git_tree": "b" * 40,
                "source_manifest_sha256": "c" * 64,
                "tracked_file_count": 1,
                "source_dirty": False,
                "development_override": False,
            },
            "artifacts": {
                "wheel": {"sha256": "d" * 64},
                "native_extension": {
                    "sha256": "e" * 64,
                    "protocol": "txnopt-native-round-v1",
                },
            },
        },
    )
    return build


def test_calibration_command_observes_peak_rss() -> None:
    payload, elapsed, peak_rss_bytes = _run(
        [
            sys.executable,
            "-c",
            "import json; data=bytearray(8_000_000); print(json.dumps({'ok': bool(data)}))",
        ]
    )

    assert payload == {"ok": True}
    assert elapsed > 0.0
    assert peak_rss_bytes >= 8_000_000


def test_calibration_subprocess_does_not_inherit_pythonpath(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A source-tree PYTHONPATH would make the bound Build16 interpreter import
    # orchestration sources instead of its installed producer wheel.
    monkeypatch.setenv("PYTHONPATH", "/forged/source")
    payload, _elapsed, _peak = _run(
        [
            sys.executable,
            "-c",
            "import json,os; print(json.dumps({'pythonpath': os.getenv('PYTHONPATH')}))",
        ]
    )

    assert payload == {"pythonpath": None}


def test_calibration_preserves_virtualenv_python_launcher(tmp_path: Path) -> None:
    launcher = tmp_path / "venv" / "bin" / "python"
    launcher.parent.mkdir(parents=True)
    launcher.symlink_to(Path(sys.executable))

    selected = _execution_python_path(launcher)

    assert selected == launcher.absolute()
    assert selected.is_symlink()


def test_protocol_v2_selects_attempt27_calibration_schema(tmp_path: Path) -> None:
    protocol = tmp_path / "protocol-v2.json"
    protocol.write_text(
        json.dumps({"schema_version": "txnopt-level1-protocol-v2"}),
        encoding="utf-8",
    )

    assert _protocol_calibration_schema(protocol) == "txnopt-local-runtime-calibration-v2"


def test_attempt27_v1_receipt_is_corrected_without_overwrite(tmp_path: Path) -> None:
    protocol = tmp_path / "protocol-v2.json"
    protocol.write_text(
        json.dumps({"schema_version": "txnopt-level1-protocol-v2"}),
        encoding="utf-8",
    )
    plan = tmp_path / "formal-plan.json"
    write_signed_json(plan, {"protocol_path": str(protocol)})
    source = tmp_path / "attempt27-v1.json"
    samples = [
        {
            "run_peak_rss_bytes": 1_000_000 + ordinal,
            "review_peak_rss_bytes": 2_000_000 + ordinal,
            "prefix_safety": "PASS",
            "aggregate_refinement_replay": "PASS",
            "fallback_count": 0,
        }
        for ordinal in range(96)
    ]
    write_signed_json(
        source,
        {
            "schema_version": "txnopt-local-runtime-calibration-v1",
            "status": "LOCAL_CALIBRATION_COMPLETE_NOT_LEVEL1_EVIDENCE",
            "build": "Build16",
            "attempt": 27,
            "plan_manifest_path": str(plan),
            "samples": samples,
            "cloud_purchase_performed": False,
            "formal_matrix_started": False,
            "holdout_opened": False,
        },
    )
    corrected_path = tmp_path / "attempt27-v2.json"

    materialize_attempt27_v2_correction(source, corrected_path)

    assert source.exists()
    corrected = read_signed_json(corrected_path)
    assert corrected["schema_version"] == "txnopt-local-runtime-calibration-v2"
    assert corrected["peak_rss_bytes"] == 2_000_095
    assert corrected["supersedes_calibration_receipt_sha256"] == hashlib.sha256(
        source.read_bytes()
    ).hexdigest()


def test_attempt27_materialization_cannot_write_attempt26_raw_root(
    tmp_path: Path,
) -> None:
    build = _build_manifest(tmp_path, 16)
    formal_raw = tmp_path / "formal-attempt26-raw"
    source = tmp_path / "attempt26.json"
    source.write_bytes(
        canonical_json_bytes(
            {
                "schema_version": "txnopt-run-config-v1",
                "run_label": "txnopt_level1_rcpsp_j1201_1_2014_serial_1_fixed_work_attempt26",
                "output_root": str(formal_raw),
                "build_manifest": {
                    "path": str(build),
                    "sha256": hashlib.sha256(build.read_bytes()).hexdigest(),
                },
                "run_config": {
                    "seed": 2014,
                    "workers": 1,
                    "execution_mode": "serial",
                    "fixed_work": 1200,
                    "deadline_seconds": None,
                    "speculation_window": 0,
                    "trace_policy": "semantic_and_physical",
                    "max_rounds": 10,
                },
                "case": {"domain": "rcpsp"},
            },
            pretty=True,
        )
    )
    calibration_raw = tmp_path / "calibration-attempt27-raw"
    plan_path, execution = _materialize_calibration_plan(
        source_plan_sha256="f" * 64,
        build_manifest_path=build,
        selections=(
            CalibrationSelection(
                source_config_path=source,
                domain="rcpsp",
                case_id="j1201_1",
                axis="serial_1",
                seed=2014,
                budget="fixed_work",
            ),
        ),
        destination=tmp_path / "attempt27-plan",
        raw_root=calibration_raw,
        attempt=27,
    )

    assert not formal_raw.exists()
    assert not calibration_raw.exists()
    assert len(execution) == 1
    derived = json.loads(execution[0].config_path.read_bytes())
    assert derived["output_root"] == str(calibration_raw.resolve())
    assert derived["run_label"].endswith("_attempt27")
    identity = read_signed_json(execution[0].expected_identity_path)
    assert identity["run_label"] == derived["run_label"]
    plan = read_signed_json(plan_path)
    assert plan["source_formal_plan_sha256"] == "f" * 64
    assert plan["formal_matrix_started"] is False
    loaded_path, loaded_execution = _load_calibration_plan(
        plan_path,
        source_plan_sha256="f" * 64,
        build_manifest_path=build,
        raw_root=calibration_raw,
        attempt=27,
    )
    assert loaded_path == plan_path
    assert loaded_execution == execution


def test_build18_calibration_is_closed_to_attempt29(tmp_path: Path) -> None:
    build = _build_manifest(tmp_path, 18)
    formal_raw = tmp_path / "formal-attempt28-raw"
    source = tmp_path / "attempt28.json"
    source.write_bytes(
        canonical_json_bytes(
            {
                "schema_version": "txnopt-run-config-v1",
                "run_label": (
                    "txnopt_level1_rcpsp_j1201_1_2014_serial_1_"
                    "fixed_work_attempt28"
                ),
                "output_root": str(formal_raw),
                "build_manifest": {
                    "path": str(build),
                    "sha256": hashlib.sha256(build.read_bytes()).hexdigest(),
                },
                "run_config": {
                    "seed": 2014,
                    "workers": 1,
                    "execution_mode": "serial",
                    "fixed_work": 1200,
                    "deadline_seconds": None,
                    "speculation_window": 0,
                    "trace_policy": "semantic_and_physical",
                    "max_rounds": 10,
                },
                "case": {"domain": "rcpsp"},
            },
            pretty=True,
        )
    )
    selection = CalibrationSelection(
        source_config_path=source,
        domain="rcpsp",
        case_id="j1201_1",
        axis="serial_1",
        seed=2014,
        budget="fixed_work",
    )

    plan_path, execution = _materialize_calibration_plan(
        source_plan_sha256="f" * 64,
        build_manifest_path=build,
        selections=(selection,),
        destination=tmp_path / "attempt29-plan",
        raw_root=tmp_path / "attempt29-raw",
        attempt=29,
    )

    assert read_signed_json(plan_path)["build"] == "Build18"
    assert execution[0].config_path.stem.endswith("_attempt29")
    with pytest.raises(ValueError, match="calibration attempt"):
        _materialize_calibration_plan(
            source_plan_sha256="f" * 64,
            build_manifest_path=build,
            selections=(selection,),
            destination=tmp_path / "attempt30-plan",
            raw_root=tmp_path / "attempt30-raw",
            attempt=30,
        )
