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
    _run,
)


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


def test_attempt27_materialization_cannot_write_attempt26_raw_root(
    tmp_path: Path,
) -> None:
    build = tmp_path / "build16.json"
    write_signed_json(
        build,
        {
            "schema_version": "txnopt-level1-build-manifest-v1",
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
