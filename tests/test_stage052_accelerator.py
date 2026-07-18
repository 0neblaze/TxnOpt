from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from evrptw.artifacts import (
    ArtifactBundleWriter,
    ArtifactRunContext,
    ArtifactStorageConfig,
)
from evrptw.experiments.stage052_performance import _accelerator_decision_inputs
from evrptw.stage052 import AcceleratorDecision, PerformanceObservation
from evrptw.stage052_accelerator import (
    InjectedMetalPilotExecutor,
    MetalPilotStatus,
    SubprocessMetalPilotExecutor,
    audit_metal_pilot_result,
    campaign_execution_adapter_gate,
    run_conditional_metal_pilot,
)


def _native_observations() -> tuple[PerformanceObservation, ...]:
    rows: list[PerformanceObservation] = []
    for family, instance in (("C", "c101_21"), ("R", "r101_21"), ("RC", "rc101_21")):
        del family
        for seed in (2014, 2015, 2016):
            rows.append(
                PerformanceObservation(
                    instance=instance,
                    seed=seed,
                    customer_count=100,
                    end_to_end_seconds=10.0,
                    semantic_digest=f"{instance}:{seed}",
                )
            )
    return tuple(rows)


def _write_high_occupancy_e_bundle(tmp_path: Path) -> Path:
    run_label = "stage05.2_native_kernels_attempt99"
    raw_dir = tmp_path / run_label
    writer = ArtifactBundleWriter(
        raw_dir,
        ArtifactRunContext("stage05.2", "native_kernels", run_label),
        ArtifactStorageConfig(storage_policy_version="artifact-storage-v2"),
    )
    writer.write_control(metadata={"run_label": run_label})
    per_run_rows: list[dict[str, object]] = []
    for instance in ("c101C5", "c101_21", "r101_21", "rc101_21"):
        for seed in (2014, 2015, 2016):
            shard = raw_dir / instance / str(seed)
            shard.mkdir(parents=True)
            raw_path = shard / f"{run_label}_raw_{instance}_{seed}.json"
            raw_path.write_text(
                json.dumps(
                    {
                        "instance": instance,
                        "seed": seed,
                        "component": "native_kernels",
                        "scope": "performance",
                        "axes": {
                            "fixed_work": {
                                "validator_passed": True,
                                "valid": True,
                                "end_to_end_seconds": 10.0,
                                "semantic_digest": f"{instance}:{seed}",
                                "backend_metrics": {
                                    "exact_calls": 32,
                                    "batch_launches": 1,
                                    "launch_occupancies": [32],
                                },
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            writer.record_existing_file(raw_path, artifact_type="raw")
            per_run_rows.append(
                {
                    "instance": instance,
                    "seed": seed,
                    "axis": "fixed_work",
                    "end_to_end_seconds": 10.0,
                    "semantic_digest": f"{instance}:{seed}",
                }
            )
    per_run_path = raw_dir / "control" / f"{run_label}_per_run.csv"
    per_run_path.parent.mkdir(parents=True, exist_ok=True)
    with per_run_path.open("w", newline="", encoding="utf-8") as handle:
        csv_writer = csv.DictWriter(handle, fieldnames=list(per_run_rows[0]))
        csv_writer.writeheader()
        csv_writer.writerows(per_run_rows)
    writer.record_existing_file(per_run_path, artifact_type="per_run_results")
    writer.finalize()
    return raw_dir


def test_below_threshold_is_decision_only_and_never_calls_metal() -> None:
    executor = InjectedMetalPilotExecutor(
        lambda rows: pytest.fail(f"Metal executor called for {rows!r}")
    )

    result = run_conditional_metal_pilot(
        median_batch_occupancy=31.0,
        native_observations=(),
        executor=executor,
    )

    assert result.status is MetalPilotStatus.COMPLETE
    assert result.decision is AcceleratorDecision.GPU_NOT_JUSTIFIED
    assert result.selected_backend == "native_cpu"
    assert result.metal_observations == ()


def test_above_threshold_without_real_runtime_is_explicit_partial_not_fallback() -> None:
    result = run_conditional_metal_pilot(
        median_batch_occupancy=32.0,
        native_observations=_native_observations(),
    )

    assert result.status is MetalPilotStatus.PARTIAL
    assert result.decision is None
    assert result.selected_backend is None
    assert result.fallback_used is False
    assert result.failure_code == "METAL_RUNTIME_UNAVAILABLE"


def test_runner_input_branch_persists_partial_payload_when_helper_is_missing(
    tmp_path: Path,
) -> None:
    raw_dir = _write_high_occupancy_e_bundle(tmp_path)

    payload = _accelerator_decision_inputs(
        raw_dir,
        prerequisite=SimpleNamespace(to_dict=lambda: {"run_label": raw_dir.name}),
    )

    assert payload["schema_version"] == "stage05.2-accelerator-pilot-artifact-v1"
    pilot = payload["pilot"]
    assert isinstance(pilot, dict)
    assert pilot["status"] == "partial"
    assert pilot["failure_code"] == "METAL_RUNTIME_UNAVAILABLE"
    assert pilot["fallback_used"] is False


def test_runner_input_branch_executes_configured_metal_helper(tmp_path: Path) -> None:
    raw_dir = _write_high_occupancy_e_bundle(tmp_path)
    helper = tmp_path / "metal-helper"
    helper.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        "request = json.load(sys.stdin)\n"
        "rows = request['rows']\n"
        "for row in rows:\n"
        "    row['end_to_end_seconds'] *= 0.8\n"
        "json.dump({'schema_version': 'stage05.2-metal-helper-v1', "
        "'execution_backend': 'metal', 'fallback_used': False, 'rows': rows}, sys.stdout)\n",
        encoding="utf-8",
    )
    helper.chmod(0o755)

    payload = _accelerator_decision_inputs(
        raw_dir,
        prerequisite=SimpleNamespace(to_dict=lambda: {"run_label": raw_dir.name}),
        metal_helper_path=helper,
    )

    pilot = payload["pilot"]
    assert isinstance(pilot, dict)
    assert pilot["status"] == "complete"
    assert pilot["decision"] == "ACCELERATOR_PROMOTED"
    assert pilot["selected_backend"] == "metal"


def test_injected_real_pilot_promotes_only_after_semantic_and_performance_gates() -> None:
    native = _native_observations()
    executor = InjectedMetalPilotExecutor(
        lambda rows: tuple(
            PerformanceObservation(
                instance=row.instance,
                seed=row.seed,
                customer_count=row.customer_count,
                end_to_end_seconds=row.end_to_end_seconds * 0.80,
                semantic_digest=row.semantic_digest,
            )
            for row in rows
        ),
        runtime_identity={"runtime": "injected-test-metal", "device": "test-gpu"},
    )

    result = run_conditional_metal_pilot(
        median_batch_occupancy=32.0,
        native_observations=native,
        executor=executor,
    )

    assert result.status is MetalPilotStatus.COMPLETE
    assert result.decision is AcceleratorDecision.ACCELERATOR_PROMOTED
    assert result.selected_backend == "metal"
    assert result.semantic_equality_passed
    assert result.aggregate_median_saving == pytest.approx(0.20)
    assert result.family_median_savings == pytest.approx({"C": 0.20, "R": 0.20, "RC": 0.20})
    campaign_ready, detail = campaign_execution_adapter_gate(result)
    assert not campaign_ready
    assert "F02 remains NOT_READY" in detail


def test_complete_but_slow_metal_pilot_retains_native_cpu() -> None:
    native = _native_observations()
    executor = InjectedMetalPilotExecutor(
        lambda rows: tuple(
            PerformanceObservation(
                instance=row.instance,
                seed=row.seed,
                customer_count=row.customer_count,
                end_to_end_seconds=row.end_to_end_seconds * 0.90,
                semantic_digest=row.semantic_digest,
            )
            for row in rows
        )
    )

    result = run_conditional_metal_pilot(
        median_batch_occupancy=100.0,
        native_observations=native,
        executor=executor,
    )

    assert result.status is MetalPilotStatus.COMPLETE
    assert result.decision is AcceleratorDecision.NATIVE_CPU_RETAINED
    assert result.selected_backend == "native_cpu"
    assert campaign_execution_adapter_gate(result)[0]


def test_family_regression_over_three_percent_blocks_metal_promotion() -> None:
    native = _native_observations()
    executor = InjectedMetalPilotExecutor(
        lambda rows: tuple(
            PerformanceObservation(
                instance=row.instance,
                seed=row.seed,
                customer_count=row.customer_count,
                end_to_end_seconds=row.end_to_end_seconds
                * (1.04 if row.instance.startswith("c") else 0.80),
                semantic_digest=row.semantic_digest,
            )
            for row in rows
        )
    )

    result = run_conditional_metal_pilot(
        median_batch_occupancy=32.0,
        native_observations=native,
        executor=executor,
    )

    assert result.aggregate_median_saving == pytest.approx(0.20)
    assert result.family_median_savings["C"] == pytest.approx(-0.04)
    assert result.decision is AcceleratorDecision.NATIVE_CPU_RETAINED
    assert result.selected_backend == "native_cpu"


def test_semantic_mismatch_can_never_promote_metal() -> None:
    native = _native_observations()
    executor = InjectedMetalPilotExecutor(
        lambda rows: tuple(
            PerformanceObservation(
                instance=row.instance,
                seed=row.seed,
                customer_count=row.customer_count,
                end_to_end_seconds=1.0,
                semantic_digest="wrong" if index == 0 else row.semantic_digest,
            )
            for index, row in enumerate(rows)
        )
    )

    result = run_conditional_metal_pilot(
        median_batch_occupancy=32.0,
        native_observations=native,
        executor=executor,
    )

    assert result.status is MetalPilotStatus.COMPLETE
    assert result.decision is AcceleratorDecision.NATIVE_CPU_RETAINED
    assert result.selected_backend == "native_cpu"
    assert not result.semantic_equality_passed


def test_subprocess_executor_rejects_reported_fallback(tmp_path: Path) -> None:
    helper = tmp_path / "metal-helper"
    helper.write_text(
        "#!/bin/sh\n"
        "python3 -c 'import json,sys; p=json.load(sys.stdin); "
        'json.dump({"schema_version":"stage05.2-metal-helper-v1",'
        '"execution_backend":"metal","fallback_used":True,"rows":[]},sys.stdout)\'\n',
        encoding="utf-8",
    )
    helper.chmod(0o755)
    executor = SubprocessMetalPilotExecutor(helper)

    with pytest.raises(ValueError, match="fallback"):
        executor.execute(_native_observations())


def test_subprocess_executor_runs_strict_real_helper_protocol(tmp_path: Path) -> None:
    helper = tmp_path / "metal-helper"
    helper.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        "request = json.load(sys.stdin)\n"
        "rows = request['rows']\n"
        "for row in rows:\n"
        "    row['end_to_end_seconds'] *= 0.8\n"
        "json.dump({'schema_version': 'stage05.2-metal-helper-v1', "
        "'execution_backend': 'metal', 'fallback_used': False, 'rows': rows}, sys.stdout)\n",
        encoding="utf-8",
    )
    helper.chmod(0o755)

    result = run_conditional_metal_pilot(
        median_batch_occupancy=32.0,
        native_observations=_native_observations(),
        executor=SubprocessMetalPilotExecutor(helper),
    )

    assert result.status is MetalPilotStatus.COMPLETE
    assert result.decision is AcceleratorDecision.ACCELERATOR_PROMOTED
    assert result.selected_backend == "metal"
    assert result.runtime_identity is not None
    assert result.runtime_identity["production_evidence_eligible"] is True


def test_pilot_payload_round_trips_without_losing_gate_evidence() -> None:
    native = _native_observations()
    result = run_conditional_metal_pilot(
        median_batch_occupancy=32.0,
        native_observations=native,
        executor=InjectedMetalPilotExecutor(lambda rows: rows),
    )

    payload = json.loads(json.dumps(result.to_dict()))

    assert payload["schema_version"] == "stage05.2-metal-pilot-v1"
    assert payload["decision"] == "NATIVE_CPU_RETAINED"
    assert payload["selected_backend"] == "native_cpu"
    assert payload["native_rows"] == payload["metal_rows"]
    assert payload["fallback_used"] is False

    audited = audit_metal_pilot_result(payload, expected_native=native)
    assert audited.decision is AcceleratorDecision.NATIVE_CPU_RETAINED


def test_independent_pilot_audit_rejects_selected_backend_tampering() -> None:
    native = _native_observations()
    result = run_conditional_metal_pilot(
        median_batch_occupancy=32.0,
        native_observations=native,
        executor=InjectedMetalPilotExecutor(lambda rows: rows),
    )
    payload = result.to_dict()
    payload["selected_backend"] = "metal"

    with pytest.raises(ValueError, match="selected backend"):
        audit_metal_pilot_result(payload, expected_native=native)
