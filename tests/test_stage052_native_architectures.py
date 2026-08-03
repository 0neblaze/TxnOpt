from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from evrptw.charging import solve_exact_charging
from evrptw.experiments.stage052_native_architecture_review import (
    ReviewRecord,
    _canonical_semantic_events,
    _common_prefix,
    _describe_first_divergence,
    _raw_axis_inventory,
    _replay_record,
    _scheduler_screening_occupancy,
    _semantic_trajectory,
    load_records,
    render_report,
    review_records,
    write_review,
)
from evrptw.experiments.stage052_native_architectures import (
    MODES,
    PAIRED_INSTANCES,
    SCHEMA_VERSION,
    SEEDS,
    WARM_START_SCHEMA_VERSION,
    ArchitectureAxisTask,
    _canonical_trace_event,
    _require_campaign_identity,
    _require_native_architecture_capabilities,
    _run_group,
    _write_signed_json,
    build_axis_plan,
    expected_axis_count,
    load_warm_start_bundle,
    rotated_modes,
    run_labels_for_scope,
)
from evrptw.native_scheduler import NativeHostScheduler
from evrptw.objective import SolutionObjective
from evrptw.parser import parse_schneider
from evrptw.validation import validate_routes
from evrptw.warm_start import canonical_customer_sequences_sha256


def test_native_campaign_gate_names_every_incomplete_architecture_capability() -> None:
    with pytest.raises(
        RuntimeError,
        match=(
            "host_candidate_transaction_scheduler, "
            "whole_search_gil_released, single_host_24_thread_compute_pool"
        ),
    ):
        _require_native_architecture_capabilities()


def test_campaign_identity_revalidates_lease_revision_and_clean_tree(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    lease_calls: list[tuple[Path, str, frozenset[str]]] = []

    def fake_require_owned(
        root: Path,
        *,
        token: str,
        allowed_phases: frozenset[str],
    ) -> None:
        lease_calls.append((root, token, allowed_phases))

    class Receipt:
        def __init__(self, stdout: str) -> None:
            self.stdout = stdout

    def fake_run(command: list[str], **_: object) -> Receipt:
        return Receipt("a" * 40 + "\n" if command[1] == "rev-parse" else "")

    monkeypatch.setattr(
        "evrptw.experiments.stage052_native_architectures.require_owned",
        fake_require_owned,
    )
    monkeypatch.setattr(
        "evrptw.experiments.stage052_native_architectures.subprocess.run",
        fake_run,
    )

    _require_campaign_identity(
        tmp_path,
        continuity_lease_token="lease-token",
        expected_revision="a" * 40,
    )

    assert lease_calls == [
        (
            tmp_path,
            "lease-token",
            frozenset({"paired-campaign", "pilot-campaign"}),
        )
    ]


@pytest.mark.parametrize(
    ("revision", "status", "message"),
    [
        ("b" * 40, "", "Git revision changed"),
        ("a" * 40, " M source.py\n", "worktree changed"),
    ],
)
def test_campaign_identity_rejects_checkout_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    revision: str,
    status: str,
    message: str,
) -> None:
    class Receipt:
        def __init__(self, stdout: str) -> None:
            self.stdout = stdout

    monkeypatch.setattr(
        "evrptw.experiments.stage052_native_architectures.require_owned",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "evrptw.experiments.stage052_native_architectures.subprocess.run",
        lambda command, **_kwargs: Receipt(
            revision + "\n" if command[1] == "rev-parse" else status
        ),
    )

    with pytest.raises(RuntimeError, match=message):
        _require_campaign_identity(
            tmp_path,
            continuity_lease_token="lease-token",
            expected_revision="a" * 40,
        )


def _plan(scope: str, tmp_path: Path):  # type: ignore[no-untyped-def]
    from evrptw.experiments.stage02_route_reduction import FORMAL_INSTANCES

    instances = PAIRED_INSTANCES if scope == "paired" else tuple(FORMAL_INSTANCES)
    warm_starts = {
        (instance_name, seed): (
            (("C1",),),
            {"source_solution_sha256": "d" * 64},
        )
        for instance_name in instances
        for seed in SEEDS
    }
    return build_axis_plan(
        scope,
        attempt=1,
        benchmark_dir=tmp_path / "benchmarks",
        output_root=tmp_path / "results",
        scheduler_socket_path=str(tmp_path / "scheduler.sock"),
        wheel_sha256="a" * 64,
        native_sha256="b" * 64,
        scheduler_sha256="f" * 64,
        revision="c" * 40,
        warm_starts=warm_starts,
    )


def _c5_warm_start(root: Path) -> tuple[tuple[str, ...], ...]:
    instance = parse_schneider(root / "data/schneider/c101C5.txt")
    return tuple((customer.name,) for customer in instance.customers)


def _c5_source_provenance(root: Path, destination: Path) -> dict[str, object]:
    instance = parse_schneider(root / "data/schneider/c101C5.txt")
    sequences = _c5_warm_start(root)
    routes = [list(solve_exact_charging(instance, sequence).route) for sequence in sequences]
    objective_key = list(
        SolutionObjective.from_report(instance, validate_routes(instance, routes)).key
    )
    payload = {"axes": {"wall_clock": {"routes": routes, "objective_key": objective_key}}}
    destination.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return {
        "source_solution_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "source_solution_path": str(destination),
        "source_axis": "wall_clock",
        "source_customer_sequences_sha256": canonical_customer_sequences_sha256(
            sequences
        ),
        "source_objective_key": objective_key,
    }


def test_warm_start_bundle_binds_hash_identity_and_customer_coverage(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    routes = _c5_warm_start(root)
    source_path = tmp_path / "source-solution.json"
    source_provenance = _c5_source_provenance(root, source_path)
    bundle = tmp_path / "warm-starts.json"
    payload = {
        "schema_version": WARM_START_SCHEMA_VERSION,
        "records": [
            {
                "instance": "c101C5",
                "seed": 2014,
                "customer_sequences": [list(route) for route in routes],
                **source_provenance,
            }
        ],
    }
    data = json.dumps(payload, sort_keys=True).encode("utf-8")
    bundle.write_bytes(data)
    bundle.with_suffix(".json.sha256").write_text(
        hashlib.sha256(data).hexdigest(),
        encoding="ascii",
    )

    loaded = load_warm_start_bundle(
        bundle,
        benchmark_dir=root / "data/schneider",
    )

    assert loaded[("c101C5", 2014)][0] == routes
    assert loaded[("c101C5", 2014)][1]["warm_start_bundle_sha256"] == (
        hashlib.sha256(data).hexdigest()
    )

    bundle.with_suffix(".json.sha256").write_text("0" * 64, encoding="ascii")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        load_warm_start_bundle(bundle, benchmark_dir=root / "data/schneider")

    bundle.with_suffix(".json.sha256").write_text(
        hashlib.sha256(data).hexdigest(),
        encoding="ascii",
    )
    source_path.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="source solution hash mismatch"):
        load_warm_start_bundle(bundle, benchmark_dir=root / "data/schneider")


def test_paired_plan_has_360_axes_and_rotates_all_five_modes(tmp_path: Path) -> None:
    plan = _plan("paired", tmp_path)

    assert len(plan) * len(MODES) == expected_axis_count("paired") == 360
    assert all(set(rotated_modes(task)) == set(MODES) for task in plan)
    first_mode_counts = Counter(rotated_modes(task)[0] for task in plan)
    assert max(first_mode_counts.values()) - min(first_mode_counts.values()) <= 1


def test_pilot_plan_has_180_wall_clock_axes_and_independent_labels(
    tmp_path: Path,
) -> None:
    plan = _plan("pilot", tmp_path)
    labels = run_labels_for_scope("pilot", 1)

    assert len(plan) * len(MODES) == expected_axis_count("pilot") == 180
    assert {task.axis for task in plan} == {"wall_clock_30"}
    assert len(set(labels.values())) == len(MODES)
    assert all("_pilot_attempt01" in label for label in labels.values())


def _review_fixture_records(
    root: Path, evidence_root: Path
) -> tuple[ReviewRecord, ...]:
    instance = parse_schneider(root / "data/schneider/c101C5.txt")
    charging = [solve_exact_charging(instance, (customer.name,)) for customer in instance.customers]
    assert all(result.feasible for result in charging)
    routes = [list(result.route) for result in charging]
    report = validate_routes(instance, routes)
    assert report.feasible
    objective = SolutionObjective.from_report(instance, report)
    records = []
    empty_rows = {
        "count": 0,
        "sha256": hashlib.sha256(b"stage05.2-row-evidence-v1\0").hexdigest(),
    }
    measurement_evidence: dict[str, Any] = {
        "present": True,
        "exact_route_order": empty_rows,
        "cache_lifecycle": empty_rows,
        "deadline_boundaries": empty_rows,
    }
    measurement_evidence["sha256"] = hashlib.sha256(
        json.dumps(
            measurement_evidence,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    for mode in MODES:
        payload: dict[str, Any] = {
            "schema_version": "stage05.2-native-architecture-comparison-v3",
            "status": "completed",
            "mode": mode.value,
            "repeat": 0,
            "axis": "fixed_work",
            "instance": "c101C5",
            "seed": 2014,
            "revision": "c" * 40,
            "wheel_sha256": "d" * 64,
            "native_sha256": "e" * 64,
            "run_label": f"stage05.2_native_architecture_{mode.value}_test_attempt01",
            "routes": routes,
            "objective": list(objective.key),
            "solver_seconds": 1.0,
            "effective_iterations": 50,
            "exact_started_calls": 10,
            "exact_completed_calls": 10,
            "candidate_work_hash": "a" * 64,
            "route_result_hash": "b" * 64,
            "trajectory": empty_rows,
            "operator_statistics": {},
            "stage04_statistics": {},
            "stage04_events": empty_rows,
            "candidate_transaction_events": empty_rows,
            "measurement_evidence": measurement_evidence,
            "semantic_completeness": {
                "candidate_control": True,
                "stage04": True,
                "measurement_trace": True,
            },
            "fallback_count": 0,
            "native_execution_statistics": {"fallback_count": 0},
            "backend_metrics": {"launch_occupancies": [4]},
            "topology": {
                "rss_bytes": 1024,
                "cpu_utilization_percent_of_one_core": 100.0,
            },
            "cache_memory_bytes": 512,
            "artifact_bytes": 2048,
            "persistence_seconds": 0.01,
            "throughput": {
                "effective_iterations_per_second": 50.0,
                "candidate_transactions_per_second": 10.0,
                "screened_routes_per_second": 10.0,
                "exact_started_per_second": 10.0,
            },
        }
        path = evidence_root / f"{mode.value}.json"
        data = json.dumps(payload, sort_keys=True).encode("utf-8")
        path.write_bytes(data)
        path.with_suffix(path.suffix + ".sha256").write_text(
            hashlib.sha256(data).hexdigest(), encoding="ascii"
        )
        records.append(ReviewRecord(path, payload))
    return tuple(records)


def test_independent_review_replays_routes_and_accepts_equal_fixed_work(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    review = review_records(
        _review_fixture_records(root, tmp_path),
        scope="paired",
        benchmark_dir=root / "data/schneider",
    )

    assert review["axis_replay_passed"] is True
    gates = review["differential_gates"]
    assert isinstance(gates, dict)
    assert all(bool(gates[mode.value]["passed"]) for mode in MODES[2:])
    assert "五模式事实表" in render_report(review)


def test_semantic_trajectory_reports_the_real_first_divergence() -> None:
    baseline = [
        {"lane": "legacy", "iteration": 0, "operator": "relocate"},
        {"lane": "constraint", "iteration": 0, "operator": "station_pressure"},
        {"lane": "legacy", "iteration": 1, "operator": "swap"},
    ]
    candidate = [
        baseline[0],
        {"lane": "constraint", "iteration": 0, "operator": "shaw_related"},
        baseline[2],
    ]

    assert _common_prefix(baseline, candidate) == 1


def test_v5_reviewer_rejects_scheduler_overlap_outside_host_wave(
    tmp_path: Path,
) -> None:
    labels = run_labels_for_scope("paired", 91)
    run_dir = tmp_path / labels["current_stage052"]
    _write_signed_json(
        run_dir / "run_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "revision": "a" * 40,
            "wheel_sha256": "b" * 64,
            "native_sha256": "c" * 64,
            "scheduler_sha256": "d" * 64,
            "topology": {
                "scheduler_startup_seconds": 0.1,
                "scheduler_shutdown_seconds": 0.1,
                "scheduler_observed": [],
                "mode_wave_resources": [
                    {
                        "mode": "current_stage052",
                        "elapsed_seconds": 1.0,
                        "process_tree_cpu_seconds": 1.0,
                        "peak_aggregate_rss_bytes": 1,
                        "peak_aggregate_pss_bytes": 1,
                        "scheduler_process_id": 123,
                        "scheduler_startup_seconds": 0.0,
                        "scheduler_shutdown_seconds": 0.0,
                    }
                ],
            },
        },
    )

    with pytest.raises(RuntimeError, match="exclusive mode wave"):
        load_records("paired", attempt=91, results_root=tmp_path)


def test_canonical_candidate_event_has_stable_candidate_identity() -> None:
    event = {
        "event_type": "candidate_state",
        "timestamp_seconds": 1.25,
        "lane": "constraint_lane",
        "iteration": 7,
        "operator": "shaw_related",
        "candidate_route_keys": ["route-b", "route-a"],
        "candidate_full_route_keys": ["full-b", "full-a"],
        "candidate_id": "producer-local-a",
        "accepted": False,
    }

    canonical = _canonical_trace_event(event)
    repeated = _canonical_trace_event(
        {
            **event,
            "timestamp_seconds": 9.0,
            "candidate_id": "producer-local-b",
        }
    )

    assert canonical == repeated
    assert canonical["candidate_id"] == (
        "067f6e7fb31c2e26543409583f6b9877247769891162f00de974d42bd994c154"
    )
    assert "timestamp_seconds" not in canonical

    with pytest.raises(ValueError, match="candidate_id"):
        _semantic_trajectory(
            {"semantic_trajectory": [{**canonical, "candidate_id": "0" * 64}]}
        )
    assert _semantic_trajectory({"semantic_trajectory": [canonical]}) == [
        canonical
    ]
    with pytest.raises(ValueError, match="candidate_route_keys"):
        _canonical_trace_event({"event_type": "candidate_state"})
    with pytest.raises(ValueError, match="candidate_route_keys"):
        _semantic_trajectory(
            {
                "semantic_trajectory": [
                    {
                        "event_type": "candidate_state",
                        "candidate_route_keys": "not-an-array",
                        "candidate_id": "0" * 64,
                    }
                ]
            }
        )
    with pytest.raises(ValueError, match="only candidate_state"):
        _semantic_trajectory(
            {"semantic_trajectory": [{"event_type": "cache_event"}]}
        )


def test_first_divergence_reports_coordinates_and_differing_fields() -> None:
    baseline = [
        {
            "lane": "constraint_lane",
            "iteration": 4,
            "operator": "shaw_related",
            "candidate_id": "candidate-4",
            "accepted": True,
            "status": "accepted",
        }
    ]
    candidate = [
        {
            "lane": "constraint_lane",
            "iteration": 4,
            "operator": "shaw_related",
            "candidate_id": "candidate-4",
            "accepted": False,
            "status": "rejected",
        }
    ]

    divergence = _describe_first_divergence(baseline, candidate)

    assert divergence == {
        "index": 0,
        "lane": "constraint_lane",
        "iteration": 4,
        "operator": "shaw_related",
        "candidate_id": "candidate-4",
        "differing_fields": {
            "accepted": {"baseline": True, "candidate": False},
            "status": {"baseline": "accepted", "candidate": "rejected"},
        },
        "baseline": baseline[0],
        "candidate": candidate[0],
    }


def test_first_divergence_distinguishes_missing_field_from_null() -> None:
    baseline = [
        {
            "lane": "legacy",
            "iteration": 2,
            "operator": "route_merge",
            "candidate_id": "candidate-2",
            "reason": None,
        }
    ]
    candidate = [{"lane": "legacy"}]

    divergence = _describe_first_divergence(baseline, candidate)

    assert divergence is not None
    assert divergence["lane"] == "legacy"
    assert divergence["iteration"] == 2
    assert divergence["operator"] == "route_merge"
    assert divergence["candidate_id"] == "candidate-2"
    assert divergence["differing_fields"] == {
        "candidate_id": {
            "baseline": "candidate-2",
            "candidate": {"field_missing": True},
        },
        "iteration": {
            "baseline": 2,
            "candidate": {"field_missing": True},
        },
        "operator": {
            "baseline": "route_merge",
            "candidate": {"field_missing": True},
        },
        "reason": {
            "baseline": None,
            "candidate": {"field_missing": True},
        }
    }


def test_canonical_semantic_streams_find_non_candidate_first_divergence() -> None:
    empty_streams = {
        name: []
        for name in (
            "candidate_state",
            "operator",
            "stage04",
            "candidate_transaction",
            "exact_work",
            "exact_result",
            "cache",
            "deadline",
        )
    }
    baseline_streams = {name: list(rows) for name, rows in empty_streams.items()}
    candidate_streams = {name: list(rows) for name, rows in empty_streams.items()}
    baseline_streams["stage04"] = [
        {
            "lane": "legacy",
            "iteration": 7,
            "operator": "route_merge",
            "candidate_id": "candidate-7",
            "weight": 2.0,
            "stream_ordinal": 0,
        }
    ]
    candidate_streams["stage04"] = [
        {
            **baseline_streams["stage04"][0],
            "weight": 3.0,
        }
    ]

    baseline = _canonical_semantic_events(
        {"canonical_semantic_streams": baseline_streams}
    )
    candidate = _canonical_semantic_events(
        {"canonical_semantic_streams": candidate_streams}
    )
    divergence = _describe_first_divergence(baseline, candidate)

    assert _common_prefix(baseline, candidate) == 0
    assert divergence is not None
    assert divergence["lane"] == "legacy"
    assert divergence["iteration"] == 7
    assert divergence["operator"] == "route_merge"
    assert divergence["candidate_id"] == "candidate-7"
    assert divergence["differing_fields"] == {
        "weight": {"baseline": 2.0, "candidate": 3.0}
    }


def test_v6_axis_replay_rejects_bad_candidate_id_on_wall_clock_axis(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    source = _review_fixture_records(root, tmp_path)[0]
    payload = dict(source.payload)
    payload["schema_version"] = (
        "stage05.2-native-architecture-comparison-v6"
    )
    payload["axis"] = "wall_clock_30"
    payload["semantic_trajectory"] = [
        {
            "event_type": "candidate_state",
            "lane": "legacy",
            "iteration": 0,
            "operator": "route_elimination",
            "candidate_route_keys": [],
            "candidate_id": "0" * 64,
        }
    ]

    replay = _replay_record(
        ReviewRecord(source.path, payload),
        root / "data" / "schneider",
    )

    assert replay == {
        "valid": False,
        "reason": (
            "semantic trajectory replay failed: semantic candidate_id does not "
            "match its canonical route projection"
        ),
    }

    for missing_value in ("absent", None):
        missing_payload = dict(payload)
        if missing_value == "absent":
            del missing_payload["semantic_trajectory"]
        else:
            missing_payload["semantic_trajectory"] = None
        assert _replay_record(
            ReviewRecord(source.path, missing_payload),
            root / "data" / "schneider",
        ) == {
            "valid": False,
            "reason": (
                "semantic trajectory replay failed: v4 evidence is missing"
            ),
        }


def test_cuda_condition_does_not_reuse_exact_backend_occupancy(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    records = list(_review_fixture_records(root, tmp_path))
    host = next(record for record in records if record.mode.value == "host_scheduler")
    assert isinstance(host.payload, dict)
    host.payload["backend_metrics"] = {"launch_occupancies": [64]}

    condition = _scheduler_screening_occupancy((host,))

    assert condition == {
        "available": False,
        "condition_met": False,
        "maximum": None,
        "reason": "native candidate-screening occupancy is not recorded",
    }


def test_raw_axis_inventory_binds_json_and_sidecar_bytes(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    records = _review_fixture_records(root, tmp_path)
    before = _raw_axis_inventory(records)
    sidecar = records[0].path.with_suffix(records[0].path.suffix + ".sha256")
    sidecar.write_text(sidecar.read_text(encoding="ascii") + "\n", encoding="ascii")

    after = _raw_axis_inventory(records)

    assert before["axis_count"] == len(MODES)
    assert before["tree_sha256"] != after["tree_sha256"]


def test_review_writer_emits_hash_bound_review_manifest(tmp_path: Path) -> None:
    output_json = tmp_path / "paired_review.json"
    output_markdown = tmp_path / "paired_report.md"
    review = {
        "schema_version": "test-review-v1",
        "scope": "paired",
        "review_status": "NOT_READY",
        "reviewer_provenance": {
            "repository_revision": "a" * 40,
            "source_sha256": "b" * 64,
        },
        "producer_identity": {
            "repository_revisions": ["c" * 40],
            "wheel_sha256": ["d" * 64],
            "native_sha256": ["e" * 64],
            "scheduler_sha256": ["f" * 64],
            "run_labels": ["stage05.2_native_architecture_test_attempt01"],
        },
        "mode_metrics": {mode.value: {} for mode in MODES},
        "performance": {},
        "differential_gates": {},
        "historical_attempt72": {
            "available": False,
            "identity_verified": False,
            "comparison_count": 0,
        },
        "cuda_evaluation_condition": {
            "available": False,
            "condition_met": False,
            "maximum": None,
            "reason": "native candidate-screening occupancy is not recorded",
        },
        "axis_replay_passed": False,
        "axis_count": 0,
    }

    write_review(review, output_json=output_json, output_markdown=output_markdown)

    manifest = json.loads(
        (tmp_path / "paired_review_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["reviewer_provenance"] == review["reviewer_provenance"]
    assert manifest["producer_identity"] == review["producer_identity"]
    assert manifest["files"][output_json.name] == hashlib.sha256(
        output_json.read_bytes()
    ).hexdigest()
    assert manifest["files"][output_markdown.name] == hashlib.sha256(
        output_markdown.read_bytes()
    ).hexdigest()


def test_one_wall_clock_group_runs_all_five_modes_with_one_scheduler(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    endpoint = tmp_path / "scheduler.sock"
    task = ArchitectureAxisTask(
        scope="paired",
        repeat=0,
        axis="wall_clock_30",
        instance_name="c101C5",
        seed=2014,
        benchmark_dir=root / "data/schneider",
        output_root=tmp_path,
        run_labels=run_labels_for_scope("paired", 99),
        scheduler_socket_path=str(endpoint),
        wheel_sha256="a" * 64,
        native_sha256="b" * 64,
        scheduler_sha256="f" * 64,
        revision="c" * 40,
        initial_customer_sequences=_c5_warm_start(root),
        initial_solution_provenance=_c5_source_provenance(
            root, tmp_path / "wall-clock-source.json"
        ),
    )

    with NativeHostScheduler(endpoint):
        written = _run_group(task)

    assert len(written) == len(MODES)
    payloads = [json.loads(Path(path).read_bytes()) for path in written]
    assert {payload["mode"] for payload in payloads} == {mode.value for mode in MODES}
    assert {payload["scheduler_sha256"] for payload in payloads} == {"f" * 64}
    status_by_mode = {payload["mode"]: payload["status"] for payload in payloads}
    assert status_by_mode == {
        "current_stage052": "completed",
        "python_candidate_control": "completed",
        "per_solve_runtime": "completed",
        "full_native_alns": "failed",
        "host_scheduler": "failed",
    }
    assert all(payload["persistence_seconds"] > 0.0 for payload in payloads)
    assert not tuple(tmp_path.rglob("*.persistence-probe-*"))


def test_one_fixed_work_group_retains_every_mode_axis(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    endpoint = tmp_path / "scheduler.sock"
    task = ArchitectureAxisTask(
        scope="paired",
        repeat=0,
        axis="fixed_work",
        instance_name="c101C5",
        seed=2014,
        benchmark_dir=root / "data/schneider",
        output_root=tmp_path,
        run_labels=run_labels_for_scope("paired", 98),
        scheduler_socket_path=str(endpoint),
        wheel_sha256="a" * 64,
        native_sha256="b" * 64,
        scheduler_sha256="f" * 64,
        revision="c" * 40,
        initial_customer_sequences=_c5_warm_start(root),
        initial_solution_provenance=_c5_source_provenance(
            root, tmp_path / "fixed-work-source.json"
        ),
    )

    with NativeHostScheduler(endpoint):
        written = _run_group(task)

    payloads = [json.loads(Path(path).read_bytes()) for path in written]
    assert len(payloads) == len(MODES)
    assert {payload["mode"] for payload in payloads} == {mode.value for mode in MODES}
    status_by_mode = {payload["mode"]: payload["status"] for payload in payloads}
    assert status_by_mode == {
        "current_stage052": "completed",
        "python_candidate_control": "completed",
        "per_solve_runtime": "completed",
        "full_native_alns": "failed",
        "host_scheduler": "failed",
    }
