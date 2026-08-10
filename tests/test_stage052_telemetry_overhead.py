from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from evrptw.candidate_control import stable_candidate_payload_hash
from evrptw.experiments.stage052_telemetry_overhead import (
    TelemetryWorkloadSample,
    load_telemetry_overhead_receipt,
    measure_representative_telemetry_overhead,
    measure_telemetry_overhead,
    write_telemetry_overhead_receipt,
)
from evrptw.objective import SolutionObjective
from evrptw.parser import parse_schneider
from evrptw.stage052_performance import TelemetryOverheadReceipt
from evrptw.validation import validate_routes


def _write_signed_json(path: Path, payload: object) -> tuple[str, str]:
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    sidecar = Path(f"{path}.sha256")
    sidecar.write_text(digest + "\n", encoding="ascii")
    return digest, hashlib.sha256(sidecar.read_bytes()).hexdigest()


def test_telemetry_overhead_measurement_is_paired_signed_and_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    tick = iter(
        (
            0.0,
            1.0,
            1.0,
            2.01,
            2.01,
            3.01,
            3.01,
            4.02,
            4.02,
            5.02,
            5.02,
            6.03,
            6.03,
            7.03,
            7.03,
            8.04,
            8.04,
            9.04,
            9.04,
            10.05,
        )
    )

    def workload(_iterations: int) -> bytes:
        nonlocal calls
        calls += 1
        return b"a" * 64

    class Monitor:
        def __init__(self, *, sample_interval_seconds: float) -> None:
            assert sample_interval_seconds == 0.05

        def __enter__(self) -> Monitor:
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def statistics(
            self,
            *,
            elapsed_seconds: float,
            compute_thread_limit: int,
        ) -> dict[str, object]:
            assert elapsed_seconds > 0.0
            assert compute_thread_limit > 0
            return {"sample_count": 10}

    monkeypatch.setattr(
        "evrptw.experiments.stage052_telemetry_overhead.ProcessTreeMonitor",
        Monitor,
    )
    monkeypatch.setattr(
        "evrptw.experiments.stage052_telemetry_overhead.time.perf_counter",
        lambda: next(tick),
    )
    receipt = measure_telemetry_overhead(
        iterations=1,
        repeat_count=5,
        minimum_unmonitored_seconds=0.0,
        workload=workload,
    )
    assert calls == 11
    assert receipt.passed
    assert receipt.pair_orders == (
        "off-on",
        "on-off",
        "off-on",
        "on-off",
        "off-on",
    )
    path = tmp_path / "telemetry-overhead.json"
    write_telemetry_overhead_receipt(path, receipt)
    assert load_telemetry_overhead_receipt(path) == receipt
    assert (
        Path(f"{path}.sha256").read_text().strip() == hashlib.sha256(path.read_bytes()).hexdigest()
    )
    with pytest.raises(FileExistsError):
        write_telemetry_overhead_receipt(path, receipt)


def test_representative_telemetry_gate_covers_complete_fixed_work_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ticks = iter(float(index) for index in range(28))
    monkeypatch.setattr(
        "evrptw.experiments.stage052_telemetry_overhead.time.perf_counter",
        lambda: next(ticks),
    )
    evidence = {
        "kind": "representative-fixed-work-axis",
        "mode": "current_stage052",
        "axis": "fixed_work",
        "instance": "c101C5",
        "seed": 2014,
        "exact_calls": 20,
        "iterations": 200,
        "batch_size": 128,
        "minimal_validator_replay": True,
        "fingerprints_identical": True,
    }

    def sample(enabled: bool, index: int) -> TelemetryWorkloadSample:
        return TelemetryWorkloadSample(
            b"a" * 64,
            {"enabled": enabled, "sample_index": index},
            {
                **evidence,
                "semantic_telemetry": enabled,
                "physical_telemetry": enabled,
                "persistence": enabled,
                "independent_replay": enabled,
            },
        )

    receipt = measure_representative_telemetry_overhead(
        run_sample=sample,
        minimum_unmonitored_seconds=0.0,
    )
    receipt.require_representative_fixed_work()
    assert receipt.passed
    assert receipt.p95_overhead_fraction == pytest.approx(0.0)


def test_independent_reviewer_replays_every_raw_telemetry_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from evrptw.experiments import stage052_performance_calibration_review as review

    benchmark_dir = Path(__file__).resolve().parents[1] / "data/schneider"
    instance = parse_schneider(benchmark_dir / "c101C5.txt")
    routes = [
        ["D0", "C30", "D0"],
        ["D0", "C12", "D0"],
        ["D0", "C100", "D0"],
        ["D0", "C85", "D0"],
        ["D0", "C64", "D0"],
    ]
    report = validate_routes(instance, routes)
    assert report.feasible
    objective = list(SolutionObjective.from_report(instance, report).key)
    candidate_work_events: tuple[dict[str, object], ...] = ()
    route_result_events: tuple[dict[str, object], ...] = ()
    candidate_work_hash = stable_candidate_payload_hash(candidate_work_events)
    route_result_hash = stable_candidate_payload_hash(route_result_events)
    empty_row_receipt = {
        "count": 0,
        "sha256": hashlib.sha256(b"stage05.2-row-evidence-v1\0").hexdigest(),
    }
    semantic_trajectory_receipt = {
        "schema_version": "stage05.2-external-semantic-trajectory-v1",
        "source": "canonical_semantic_journal:operator",
        **empty_row_receipt,
    }
    axis_payload = {
        "objective": objective,
        "routes": routes,
        "candidate_work_hash": candidate_work_hash,
        "route_result_hash": route_result_hash,
        "effective_iterations": 20,
        "termination_reason": "fixed_work_exhausted",
        "accepted_moves": 2,
        "rejected_moves": 18,
        "exact_started_calls": 20,
        "exact_completed_calls": 20,
        "exact_interrupted_calls": 0,
        "semantic_trajectory": semantic_trajectory_receipt,
        "trajectory": empty_row_receipt,
        "stage04_events": empty_row_receipt,
        "candidate_transaction_events": empty_row_receipt,
    }
    fingerprint = hashlib.sha256(
        json.dumps(axis_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    axis_path = tmp_path / "raw-axis.json"
    axis_sha256, axis_sidecar_sha256 = _write_signed_json(axis_path, axis_payload)
    persistence_path = tmp_path / "raw-axis.persistence.json"
    persistence_sha256, persistence_sidecar_sha256 = _write_signed_json(
        persistence_path,
        {"schema_version": "test-axis-persistence-v1"},
    )
    persistence_sidecar = Path(f"{persistence_path}.sha256")
    axis_inventory = [
        {
            "storage_alias": "stage052-performance-calibration-run",
            "relative_path": axis_path.relative_to(tmp_path).as_posix(),
            "sha256": axis_sha256,
            "relative_sidecar_path": Path(f"{axis_path}.sha256").relative_to(tmp_path).as_posix(),
            "sidecar_sha256": axis_sidecar_sha256,
            "role": "representative-telemetry-axis",
            "supporting_artifacts": [
                {
                    "relative_path": persistence_path.relative_to(tmp_path).as_posix(),
                    "sha256": persistence_sha256,
                    "role": "axis-persistence-receipt",
                },
                {
                    "relative_path": persistence_sidecar.relative_to(tmp_path).as_posix(),
                    "sha256": persistence_sidecar_sha256,
                    "role": "axis-persistence-receipt-sidecar",
                },
            ],
        }
    ]
    sample_base: dict[str, object] = {
        "kind": "representative-fixed-work-axis",
        "mode": "current_stage052",
        "axis": "fixed_work",
        "instance": "c101C5",
        "seed": 2014,
        "exact_calls": 20,
        "iterations": 200,
        "batch_size": 128,
        "minimal_validator_replay": True,
        "fingerprints_identical": True,
    }

    def sample(enabled: bool, index: int, elapsed: float) -> dict[str, object]:
        sample_path = tmp_path / f"sample-{index}.json"
        workload_evidence = {
            **sample_base,
            "semantic_telemetry": enabled,
            "physical_telemetry": enabled,
            "persistence": enabled,
            "independent_replay": enabled,
        }
        raw_inventory = axis_inventory if enabled else []
        minimal_replay_receipt = (
            None
            if enabled
            else {
                "schema_version": "stage05.2-telemetry-off-minimal-replay-v1",
                "instance": "c101C5",
                "seed": 2014,
                "routes": routes,
                "objective": objective,
                "candidate_work_events": candidate_work_events,
                "route_result_events": route_result_events,
                "candidate_work_hash": candidate_work_hash,
                "route_result_hash": route_result_hash,
                "trajectory_rows": {
                    "semantic_trajectory": (),
                    "trajectory": (),
                    "stage04_events": (),
                    "candidate_transaction_events": (),
                },
            }
        )
        payload = {
            "schema_version": "stage05.2-representative-telemetry-sample-v3",
            "enabled": enabled,
            "sample_index": index,
            "elapsed_seconds": elapsed,
            "fingerprint": fingerprint,
            "fingerprint_payload": axis_payload,
            "workload_evidence": workload_evidence,
            "raw_axis_inventory": raw_inventory,
            "minimal_replay_receipt": minimal_replay_receipt,
        }
        sample_sha256, sample_sidecar_sha256 = _write_signed_json(sample_path, payload)
        return {
            "enabled": enabled,
            "sample_index": index,
            "elapsed_seconds": elapsed,
            "fingerprint": fingerprint,
            "resource_summary": {
                "sample_storage_alias": "stage052-performance-calibration-run",
                "sample_relative_path": sample_path.relative_to(tmp_path).as_posix(),
                "sample_sidecar_relative_path": Path(f"{sample_path}.sha256")
                .relative_to(tmp_path)
                .as_posix(),
                "sample_sha256": sample_sha256,
                "sample_sidecar_sha256": sample_sidecar_sha256,
                "child_elapsed_seconds": elapsed,
                "raw_axis_inventory": raw_inventory,
            },
            "workload_evidence": workload_evidence,
        }

    orders = ("off-on", "on-off", "off-on", "on-off", "off-on")
    unmonitored = (1.0,) * 5
    monitored = (1.01,) * 5
    evidence = {
        **sample_base,
        "semantic_telemetry": True,
        "physical_telemetry": True,
        "persistence": True,
        "independent_replay": True,
        "unmonitored_telemetry_surface": {
            "semantic_telemetry": False,
            "physical_telemetry": False,
            "persistence": False,
            "independent_replay": False,
        },
        "monitored_telemetry_surface": {
            "semantic_telemetry": True,
            "physical_telemetry": True,
            "persistence": True,
            "independent_replay": True,
        },
    }
    evidence["warm_sample_evidence"] = (
        sample(False, -2, 1.0),
        sample(True, -1, 1.0),
    )
    evidence["paired_sample_evidence"] = tuple(
        {
            "pair_index": index,
            "order": order,
            "unmonitored": sample(
                False,
                index * 2 if order == "off-on" else index * 2 + 1,
                unmonitored[index],
            ),
            "monitored": sample(
                True,
                index * 2 + 1 if order == "off-on" else index * 2,
                monitored[index],
            ),
        }
        for index, order in enumerate(orders)
    )
    receipt = TelemetryOverheadReceipt(
        unmonitored_seconds=unmonitored,
        monitored_seconds=monitored,
        pair_orders=orders,
        sample_interval_seconds=0.05,
        workload_output_sha256=hashlib.sha256(fingerprint.encode("ascii")).hexdigest(),
        monitored_resource_summaries=tuple({"sample_count": 1} for _ in range(5)),
        workload_evidence=evidence,
    )
    monkeypatch.setattr(review, "_review_raw_axis", lambda *_args, **_kwargs: None)

    replayed = review._replay_telemetry_children(
        receipt,
        run_root=tmp_path,
        benchmark_dir=benchmark_dir,
        build_identity={},
    )

    assert replayed == 6

    warm_samples = evidence["warm_sample_evidence"]
    assert isinstance(warm_samples, tuple)
    off_row = warm_samples[0]
    assert isinstance(off_row, dict)
    resources = off_row["resource_summary"]
    assert isinstance(resources, dict)
    off_path = tmp_path / str(resources["sample_relative_path"])
    off_payload = json.loads(off_path.read_text(encoding="utf-8"))
    off_payload["minimal_replay_receipt"]["routes"] = []
    sample_sha256, sidecar_sha256 = _write_signed_json(off_path, off_payload)
    resources["sample_sha256"] = sample_sha256
    resources["sample_sidecar_sha256"] = sidecar_sha256
    tampered_receipt = TelemetryOverheadReceipt(
        unmonitored_seconds=unmonitored,
        monitored_seconds=monitored,
        pair_orders=orders,
        sample_interval_seconds=0.05,
        workload_output_sha256=hashlib.sha256(fingerprint.encode("ascii")).hexdigest(),
        monitored_resource_summaries=tuple({"sample_count": 1} for _ in range(5)),
        workload_evidence=evidence,
    )
    with pytest.raises(review.CalibrationReviewError, match="validator replay failed"):
        review._replay_telemetry_children(
            tampered_receipt,
            run_root=tmp_path,
            benchmark_dir=benchmark_dir,
            build_identity={},
        )

    off_payload["minimal_replay_receipt"]["routes"] = routes
    off_payload["minimal_replay_receipt"]["trajectory_rows"]["candidate_transaction_events"] = [
        {"status": "accepted"}
    ]
    sample_sha256, sidecar_sha256 = _write_signed_json(off_path, off_payload)
    resources["sample_sha256"] = sample_sha256
    resources["sample_sidecar_sha256"] = sidecar_sha256
    trajectory_tampered_receipt = TelemetryOverheadReceipt(
        unmonitored_seconds=unmonitored,
        monitored_seconds=monitored,
        pair_orders=orders,
        sample_interval_seconds=0.05,
        workload_output_sha256=hashlib.sha256(fingerprint.encode("ascii")).hexdigest(),
        monitored_resource_summaries=tuple({"sample_count": 1} for _ in range(5)),
        workload_evidence=evidence,
    )
    with pytest.raises(
        review.CalibrationReviewError,
        match="candidate_transaction_events digest does not replay",
    ):
        review._replay_telemetry_children(
            trajectory_tampered_receipt,
            run_root=tmp_path,
            benchmark_dir=benchmark_dir,
            build_identity={},
        )
