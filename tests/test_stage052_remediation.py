from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest

import evrptw.experiments.stage052_performance_review as performance_review
from evrptw.artifacts import (
    ArtifactBundleWriter,
    ArtifactIntegrityError,
    ArtifactReader,
    ArtifactRunContext,
    ArtifactStorageConfig,
)
from evrptw.experiments.stage052_performance import _record_remediation_child
from evrptw.experiments.stage052_performance_review import _validate_c05_remediation
from evrptw.stage052_evidence import Stage052PrerequisiteIdentity
from evrptw.stage052_remediation import (
    Stage052RemediationConfig,
    _validate_source_storage,
    remediate_stage052_artifacts,
    replay_stage052_bundle_semantics,
)

SOURCE_RUN_LABEL = "stage05.2_native_kernels_attempt03"
CHILD_RUN_LABEL = "stage05.2_artifact_streaming_attempt05"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_signed_source(
    tmp_path: Path,
    *,
    status: str = "NOT_READY",
    solution_distance: float = 2.0,
) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    source_dir = tmp_path / "results" / SOURCE_RUN_LABEL
    config_path = tmp_path / "stage052.toml"
    config_path.write_text("[artifact_storage]\nenabled = true\n", encoding="utf-8")
    writer = ArtifactBundleWriter(
        source_dir,
        ArtifactRunContext("stage05.2", "native_kernels", SOURCE_RUN_LABEL),
        ArtifactStorageConfig(storage_policy_version="artifact-storage-v2"),
    )
    writer.write_control(
        metadata={
            "run_label": SOURCE_RUN_LABEL,
            "component": "native_kernels",
            "scope": "performance",
            "repository_revision": "a" * 40,
            "repository_dirty": False,
            "configuration_sha256": _sha256(config_path),
        },
        configuration_path=config_path,
    )
    writer.write_instance_seed(
        instance="c101_21",
        seed=2014,
        raw_payload={
            "instance": "c101_21",
            "seed": 2014,
            "axes": {
                "fixed_work": {
                    "valid": True,
                    "started_calls": 2,
                    "completed_calls": 2,
                }
            },
        },
        solution_payload={
            "instance": "c101_21",
            "seed": 2014,
            "axes": {"fixed_work": {"objective_key": [1, solution_distance, 0.0, 0]}}
        },
        trace_payload={"axes": {"fixed_work": {"summary": {"started_calls": 2}}}},
        environment_payload={"python": "3.13"},
        route_dictionary={"route:2:C1": ("C1",)},
        critical_events=(
            {
                "event_type": "route_evaluation",
                "benchmark_axis": "fixed_work",
                "route_key": "route:2:C1",
                "lane": "fixed_work:legacy",
                "operator": "relocate",
                "iteration": 3,
                "exact_started": True,
                "exact_completed": True,
                "feasible": True,
            },
            {
                "event_type": "screening_decision",
                "benchmark_axis": "fixed_work",
                "route_key": "route:2:C1",
                "lane": "fixed_work:constraint",
                "operator": "station_pressure",
                "iteration": 4,
                "decision_id": 9,
                "status": "pass",
                "reason": "",
                "demand": 2.0,
                "distance_lower_bound": 3.5,
                "distance_increment_lower_bound": 0.5,
                "exact_call_blocked": False,
                "first_failed_check": "",
                "min_time_window_slack": 4.0,
                "negative_cache_hit": False,
                "single_segment_reachable": True,
                "structural_energy_lower_bound": 1.5,
                "started_at": 1.25,
                "completed_at": 1.5,
                "duration_seconds": 0.25,
                "checks": (
                    {
                        "check": "capacity",
                        "status": "pass",
                        "value": True,
                        "reason": "",
                    },
                ),
            },
        ),
        shard_ordinal=0,
        worker_identity="pid-0",
    )
    per_run = source_dir / "control" / f"{SOURCE_RUN_LABEL}_per_run_results.csv"
    with per_run.open("w", encoding="utf-8", newline="") as handle:
        csv_writer = csv.DictWriter(
            handle,
            fieldnames=("instance", "seed", "axis", "solver_seconds"),
        )
        csv_writer.writeheader()
        csv_writer.writerow(
            {
                "instance": "c101_21",
                "seed": 2014,
                "axis": "fixed_work",
                "solver_seconds": 7.0,
            }
        )
    writer.record_existing_file(
        per_run,
        artifact_type="per_run_results",
        storage_format="csv_control",
        row_count=1,
    )
    bundle = writer.finalize()

    review_dir = source_dir / "review"
    review_dir.mkdir()
    findings = review_dir / "review_findings.csv"
    report = review_dir / "review_report.md"
    findings.write_text("gate,passed\npersistence_ratio,false\n", encoding="utf-8")
    report.write_text("# Independent review\n", encoding="utf-8")
    review = {
        "schema_version": "stage05.2-review-v1",
        "run_label": SOURCE_RUN_LABEL,
        "component": "native_kernels",
        "scope": "performance",
        "status": status,
        "review_execution_required": True,
        "raw_manifest_sha256": _sha256(bundle.manifest_path),
        "gates": {"persistence_ratio": {"passed": status != "NOT_READY"}},
        "files": {
            findings.name: _sha256(findings),
            report.name: _sha256(report),
        },
    }
    review_manifest = review_dir / "review_manifest.json"
    review_manifest.write_text(
        json.dumps(review, sort_keys=True),
        encoding="utf-8",
    )
    (review_dir / "review_execution.json").write_text(
        json.dumps(
            {
                "run_label": SOURCE_RUN_LABEL,
                "finalized": True,
                "status": "completed",
                "systemd_service_result": "success",
                "cgroup_memory_peak_status": "verified",
                "raw_manifest_unchanged": True,
                "review_manifest_sha256": _sha256(review_manifest),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return source_dir


def _config(*, expected_event_count: int = 2) -> Stage052RemediationConfig:
    return Stage052RemediationConfig(
        child_run_label=CHILD_RUN_LABEL,
        expected_shard_count=1,
        expected_solver_row_count=1,
        expected_event_count=expected_event_count,
        expected_identities=(("c101_21", 2014),),
        expected_axes=("fixed_work",),
        batch_size=1,
    )


def test_signed_e03_not_ready_bundle_is_streamed_to_equal_v3_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_dir = _write_signed_source(tmp_path)
    source_manifest = ArtifactReader(source_dir).result.manifest_path
    source_manifest_before = _sha256(source_manifest)
    child_dir = tmp_path / "results" / "c05" / "remediation" / SOURCE_RUN_LABEL

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("remediation used a full-materialization API")

    monkeypatch.setattr(ArtifactReader, "read_events", forbidden)
    monkeypatch.setattr(ArtifactReader, "reconstruct_trace", forbidden)
    monkeypatch.setattr(pq, "read_table", forbidden)

    result = remediate_stage052_artifacts(
        source_dir=source_dir,
        child_dir=child_dir,
        config=_config(),
    )

    assert _sha256(source_manifest) == source_manifest_before
    assert result.source_event_count == result.child_event_count == 2
    assert result.source_semantic_digest == result.child_semantic_digest
    assert result.verified_solver_seconds == pytest.approx(7.0)
    assert result.persistence_ratio == pytest.approx(
        result.artifact_persistence_seconds
        / (result.verified_solver_seconds + result.artifact_persistence_seconds)
    )
    assert result.child_manifest_sha256 == _sha256(result.child_manifest_path)
    assert result.child_manifest_sidecar_path.read_text(encoding="utf-8").strip() == (
        result.child_manifest_sha256
    )
    assert result.child_manifest_sidecar_sha256 == _sha256(result.child_manifest_sidecar_path)

    child = ArtifactReader(child_dir)
    assert child.manifest["evidence_completeness"] == "complete"
    subtypes = {
        str(item.get("artifact_subtype", ""))
        for item in child.manifest["artifacts"]
        if isinstance(item, dict)
    }
    assert "screening_definitions_v3" in subtypes
    assert "screening_occurrences_v3" in subtypes
    assert "screening_decisions_v2" not in subtypes


def test_remediation_infers_legacy_v2_schema_from_signed_artifact_subtypes() -> None:
    source = type(
        "LegacyReader",
        (),
        {
            "manifest": {
                "storage_policy_version": "artifact-storage-v2",
                "storage_policy": {"storage_policy_version": "artifact-storage-v2"},
                "artifacts": [
                    {
                        "artifact_type": "events",
                        "artifact_subtype": "screening_decisions_v2",
                    }
                ],
            }
        },
    )()

    _validate_source_storage(source)


def test_remediation_reader_scratch_stays_on_child_staging_volume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_dir = _write_signed_source(tmp_path)
    child_dir = tmp_path / "external-staging" / "c05" / SOURCE_RUN_LABEL
    foreign_tmp = tmp_path / "foreign-system-tmp"
    foreign_tmp.mkdir()
    monkeypatch.setenv("TMPDIR", str(foreign_tmp))
    observed_scratch_roots: list[Path | None] = []
    real_iter_events = ArtifactReader.iter_events

    def tracked_iter_events(
        reader: ArtifactReader,
        relative_path: str | Path,
        **kwargs: Any,
    ) -> Any:
        raw_root = kwargs.get("scratch_root")
        observed_scratch_roots.append(raw_root if isinstance(raw_root, Path) else None)
        return real_iter_events(reader, relative_path, **kwargs)

    monkeypatch.setattr(ArtifactReader, "iter_events", tracked_iter_events)

    result = remediate_stage052_artifacts(
        source_dir=source_dir,
        child_dir=child_dir,
        config=_config(),
    )

    assert result.persistence_passed
    assert observed_scratch_roots
    assert all(root == child_dir for root in observed_scratch_roots)
    assert list(foreign_tmp.iterdir()) == []
    assert not any(
        path.name.startswith("evrptw-screening-definitions-")
        for path in child_dir.iterdir()
    )


def test_remediation_rejects_a_source_review_that_is_not_not_ready(
    tmp_path: Path,
) -> None:
    source_dir = _write_signed_source(tmp_path, status="READY_FOR_STAGE052_BENCHMARK")
    child_dir = tmp_path / "results" / "c05" / "remediation" / SOURCE_RUN_LABEL

    with pytest.raises(ArtifactIntegrityError, match="status mismatch"):
        remediate_stage052_artifacts(
            source_dir=source_dir,
            child_dir=child_dir,
            config=_config(),
        )

    assert not child_dir.exists()


def test_event_count_mismatch_retains_a_signed_partial_child_and_fails_fast(
    tmp_path: Path,
) -> None:
    source_dir = _write_signed_source(tmp_path)
    child_dir = tmp_path / "results" / "c05" / "remediation" / SOURCE_RUN_LABEL

    with pytest.raises(ArtifactIntegrityError, match="event count"):
        remediate_stage052_artifacts(
            source_dir=source_dir,
            child_dir=child_dir,
            config=_config(expected_event_count=3),
        )

    reader = ArtifactReader(child_dir)
    assert reader.manifest["status"] == "partial"
    assert reader.manifest["evidence_completeness"] == "partial"
    failures = [
        item
        for item in reader.manifest["artifacts"]
        if isinstance(item, dict) and item.get("artifact_type") == "failure"
    ]
    assert len(failures) == 1
    failure = reader.read_json(str(failures[0]["relative_path"]))
    assert failure["status"] == "partial"
    assert "event count" in str(failure["failure_reason"])
    sidecar_path = reader.result.manifest_path.with_suffix(".sha256")
    assert sidecar_path.read_text(encoding="utf-8").strip() == _sha256(reader.result.manifest_path)


def test_remediation_rejects_a_stale_signed_review_before_writing(
    tmp_path: Path,
) -> None:
    source_dir = _write_signed_source(tmp_path)
    review_manifest = source_dir / "review" / "review_manifest.json"
    review = json.loads(review_manifest.read_text(encoding="utf-8"))
    review["raw_manifest_sha256"] = "0" * 64
    review_manifest.write_text(json.dumps(review, sort_keys=True), encoding="utf-8")
    execution_path = source_dir / "review" / "review_execution.json"
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    execution["review_manifest_sha256"] = _sha256(review_manifest)
    execution_path.write_text(json.dumps(execution, sort_keys=True), encoding="utf-8")
    child_dir = tmp_path / "results" / "c05" / "remediation" / SOURCE_RUN_LABEL

    with pytest.raises(ArtifactIntegrityError, match="stale"):
        remediate_stage052_artifacts(
            source_dir=source_dir,
            child_dir=child_dir,
            config=_config(),
        )

    assert not child_dir.exists()


def test_remediation_result_records_full_child_manifest_identity(tmp_path: Path) -> None:
    source_dir = _write_signed_source(tmp_path)
    child_dir = tmp_path / "results" / "c05" / "remediation" / SOURCE_RUN_LABEL

    result = remediate_stage052_artifacts(
        source_dir=source_dir,
        child_dir=child_dir,
        config=_config(),
    )

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["source_raw_manifest_sha256"] == result.source_raw_manifest_sha256
    assert summary["source_review_manifest_sha256"] == result.source_review_manifest_sha256
    assert summary["source_semantic_digest"] == result.source_semantic_digest
    assert summary["child_semantic_digest"] == result.child_semantic_digest
    assert summary["source_event_count"] == summary["child_event_count"] == 2
    assert summary["screening_schema_version"] == "screening_decisions_v3"
    assert (
        pq.ParquetFile(
            next(child_dir.glob("*/*/*_screening_definitions_*.parquet"))
        ).metadata.num_rows
        == 1
    )


def test_c05_parent_binds_the_complete_nested_child_and_reviewer_accepts_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_dir = _write_signed_source(tmp_path / "source")
    parent_dir = tmp_path / "results" / CHILD_RUN_LABEL
    child_dir = parent_dir / "remediation" / SOURCE_RUN_LABEL
    result = remediate_stage052_artifacts(
        source_dir=source_dir,
        child_dir=child_dir,
        config=_config(),
    )
    parent = ArtifactBundleWriter(
        parent_dir,
        ArtifactRunContext("stage05.2", "artifact_streaming", CHILD_RUN_LABEL),
        ArtifactStorageConfig(
            storage_policy_version="artifact-storage-v2",
            screening_schema_version="screening_decisions_v3",
        ),
    )
    _record_remediation_child(parent, result)
    parent.finalize()

    verified_parent = ArtifactReader(parent_dir)
    nested = [
        item
        for item in verified_parent.manifest["artifacts"]
        if item["artifact_type"] == "remediation_child_payload"
    ]
    assert len(nested) == len(ArtifactReader(child_dir).manifest["artifacts"])

    source_reader = ArtifactReader(source_dir)
    identity = Stage052PrerequisiteIdentity(
        run_label=SOURCE_RUN_LABEL,
        component="native_kernels",
        status="NOT_READY",
        repository_revision="a" * 40,
        configuration_sha256="b" * 64,
        raw_manifest_sha256=_sha256(source_reader.result.manifest_path),
        review_manifest_sha256=_sha256(source_dir / "review/review_manifest.json"),
    )
    monkeypatch.setattr(performance_review, "E03_SOLVER_ROW_COUNT", 1)
    monkeypatch.setattr(performance_review, "E03_SHARD_COUNT", 1)
    monkeypatch.setattr(performance_review, "E03_EVENT_COUNT", 2)

    passed, detail = _validate_c05_remediation(
        parent_dir,
        source_dir=source_dir,
        source_identity=identity,
    )

    assert passed is True, detail

    nested_path = parent_dir / str(nested[0]["relative_path"])
    nested_path.unlink()
    with pytest.raises(ArtifactIntegrityError, match="missing"):
        ArtifactReader(parent_dir)


def test_canonical_digest_covers_non_event_objective_semantics(tmp_path: Path) -> None:
    original = _write_signed_source(tmp_path / "original", solution_distance=2.0)
    tampered = _write_signed_source(tmp_path / "tampered", solution_distance=999.0)

    original_summary = replay_stage052_bundle_semantics(original, batch_size=1)
    tampered_summary = replay_stage052_bundle_semantics(tampered, batch_size=1)

    assert original_summary.event_count == tampered_summary.event_count == 2
    assert original_summary.shard_count == tampered_summary.shard_count == 1
    assert original_summary.semantic_digest != tampered_summary.semantic_digest


def test_child_non_event_tampering_fails_semantic_replay_and_retains_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_dir = _write_signed_source(tmp_path)
    child_dir = tmp_path / "results" / "c05" / "remediation" / SOURCE_RUN_LABEL
    real_read_json = ArtifactReader.read_json

    def tampered_read_json(
        reader: ArtifactReader,
        relative_path: str | Path,
    ) -> dict[str, Any]:
        payload = real_read_json(reader, relative_path)
        if reader.run_dir == child_dir and "_solution_" in str(relative_path):
            changed: dict[str, Any] = json.loads(json.dumps(payload))
            changed["axes"]["fixed_work"]["objective_key"][1] = 999.0
            return changed
        return payload

    monkeypatch.setattr(ArtifactReader, "read_json", tampered_read_json)

    with pytest.raises(ArtifactIntegrityError, match="semantic digest"):
        remediate_stage052_artifacts(
            source_dir=source_dir,
            child_dir=child_dir,
            config=_config(),
        )

    assert ArtifactReader(child_dir).manifest["evidence_completeness"] == "partial"
