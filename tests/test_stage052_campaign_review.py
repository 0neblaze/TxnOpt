from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import orjson
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import evrptw.experiments.stage052_campaign_review as campaign_review_module
import tools.publish_stage052_artifacts as publisher_module
from evrptw.artifacts import (
    ANYTIME_CHECKPOINT_SCHEMA,
    ArtifactBundleWriter,
    ArtifactIntegrityError,
    ArtifactReader,
    ArtifactRunContext,
    ArtifactStorageConfig,
    _schema_fingerprint,
)
from evrptw.best_known import BEST_KNOWN_VALUES
from evrptw.experiments.stage052_campaign_review import (
    COMPACT_TRACE_SCHEMA_FINGERPRINTS,
    CampaignGeometryRecord,
    _publish_review,
    _read_bounded_compact_trace,
    _verify_review_storage_migration,
    audit_campaign_planning,
    audit_streamed_events,
    replay_streamed_shard_events,
    review_stage052_campaign,
    review_status_for_scope,
    summarize_streamed_global_bests,
    validate_axis_event_reconciliation,
    validate_bks_reference,
    validate_campaign_selection_lock,
    validate_compact_trace_index,
    validate_formal_geometry,
)
from evrptw.objective import SolutionObjective
from evrptw.parser import parse_schneider
from evrptw.stage052_campaign import (
    AnytimeCheckpoint,
    BatchManifest,
    BenchmarkCampaignConfig,
    CampaignManifest,
    StorageRoot,
    StorageRootLocator,
    VolumeIdentity,
    directory_byte_count,
    directory_checksum,
)
from evrptw.stage052_evidence import (
    CAMPAIGN_FORMAL_GATES,
    CAMPAIGN_PILOT_GATES,
    STAGE052_RESOURCE_MEASUREMENT_SCOPE,
    STAGE052_RESOURCE_SCHEMA_VERSION,
    BatchPersistenceEnvelope,
    PersistenceInterval,
    Stage052PersistenceAttribution,
    Stage052PrerequisiteIdentity,
    verify_stage052_review_files,
)
from evrptw.validation import validate_routes
from tools.publish_stage052_artifacts import (
    _verify_live_formal_chain,
    dry_run_stage052_publication,
    publish_stage052_artifacts,
    verify_canonical_registry_against_trusted,
    verify_published_stage052_review,
)

_SOURCE_SNAPSHOT = {
    "repository_revision": "a" * 40,
    "mount": {
        "source": "/dev/test",
        "filesystem": "ext4",
        "uuid": "test-uuid",
        "target": "/test-source",
    },
    "tracked_file_count": 100,
    "read_only": True,
}


def _power_load_payload(maximum_runtime_load1: float) -> dict[str, object]:
    power_observation = {
        "ac_online": True,
        "battery_saver": False,
        "battery_life_percent": 75,
        "battery_flag": 8,
        "active_power_scheme": "balanced-guid",
    }
    windows = [
        {
            "started_at_seconds": started,
            "duration_seconds": 30.0,
            "maximum_load1": 1.0,
            "maximum_unrelated_process_average_cores": 0.0,
            "logical_cpu_count": 24,
            "process_cpu_samples": [
                {"sampled_at_seconds": started, "cpu_seconds_by_pid": {}},
                {"sampled_at_seconds": started + 30.0, "cpu_seconds_by_pid": {}},
            ],
        }
        for started in (0.0, 30.0)
    ]
    return {
        "schema_version": "stage05.2-batch-power-load-v1",
        "run_label": "stage05.2_benchmark_attempt49",
        "batch_id": "batch0001",
        "status": "complete",
        "native_power_boundary": {
            "schema_version": "stage05.2-native-power-boundary-v1",
            "before": power_observation,
            "after": power_observation,
            "stable_invariants": [
                "ac_online",
                "battery_saver",
                "active_power_scheme",
            ],
            "invariants_unchanged": True,
        },
        "preflight": {
            "power_source": "AC Power",
            "low_power_mode_enabled": False,
            "windows": windows,
        },
        "runtime": {
            "sample_count": 2,
            "power_source_violations": 0,
            "low_power_mode_violations": 0,
            "maximum_load1": maximum_runtime_load1,
            "maximum_permitted_load1": 20.0,
            "maximum_unrelated_process_average_cores": 0.0,
            "logical_cpu_count": 24,
            "process_cpu_samples": [
                {"sampled_at_seconds": 0.0, "cpu_seconds_by_pid": {}},
                {"sampled_at_seconds": 30.0, "cpu_seconds_by_pid": {}},
            ],
        },
    }


def test_reviewer_accepts_runtime_load_within_audited_machine_headroom() -> None:
    passed, detail = campaign_review_module._validate_power_load(
        _power_load_payload(19.9),
        run_label="stage05.2_benchmark_attempt49",
        batch_id="batch0001",
        selected_workers=4,
    )

    assert passed
    assert detail == "continuous batch power/load sampling passed"


def test_reviewer_rejects_runtime_load_beyond_audited_machine_headroom() -> None:
    passed, detail = campaign_review_module._validate_power_load(
        _power_load_payload(20.1),
        run_label="stage05.2_benchmark_attempt49",
        batch_id="batch0001",
        selected_workers=4,
    )

    assert passed is False
    assert detail == "continuous batch power/load sampling violated a threshold"


def test_reviewer_rejects_missing_producer_runtime_load_ceiling() -> None:
    payload = _power_load_payload(19.9)
    runtime = payload["runtime"]
    assert isinstance(runtime, dict)
    del runtime["maximum_permitted_load1"]

    passed, _ = campaign_review_module._validate_power_load(
        payload,
        run_label="stage05.2_benchmark_attempt49",
        batch_id="batch0001",
        selected_workers=4,
    )

    assert passed is False


def test_reviewer_rejects_non_frozen_logical_cpu_count() -> None:
    payload = _power_load_payload(19.9)
    runtime = payload["runtime"]
    assert isinstance(runtime, dict)
    runtime["logical_cpu_count"] = 16

    passed, _ = campaign_review_module._validate_power_load(
        payload,
        run_label="stage05.2_benchmark_attempt49",
        batch_id="batch0001",
        selected_workers=4,
    )

    assert passed is False


def test_campaign_gate_contract_requires_source_snapshot() -> None:
    assert "source_snapshot" in CAMPAIGN_PILOT_GATES
    assert "source_snapshot" in CAMPAIGN_FORMAL_GATES


def test_campaign_volume_probe_uses_shared_cross_platform_implementation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = VolumeIdentity(device_uuid="test-ext4", filesystem="ext4")
    observed_paths: list[Path] = []

    def fake_probe(path: Path) -> VolumeIdentity:
        observed_paths.append(path)
        return expected

    monkeypatch.setattr(campaign_review_module, "probe_volume_identity", fake_probe)

    path = Path("/home/test/stage052-active")
    assert campaign_review_module._default_volume_probe(path) == expected
    assert observed_paths == [path]


def test_successor_review_verifies_the_migration_predecessor_campaign(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    migration = tmp_path / "migration.json"
    current = tmp_path / "stage05.2_benchmark_attempt36"
    predecessor = tmp_path / "stage05.2_benchmark_attempt26"
    expected = {"run_label": predecessor.name}
    calls: list[Path] = []

    def verify_successor(
        path: Path,
        *,
        evidence_dir: Path,
        locator: object,
        volume_probe: object,
    ) -> dict[str, object]:
        del locator, volume_probe
        assert path == migration
        calls.append(evidence_dir)
        return expected

    monkeypatch.setattr(
        campaign_review_module,
        "verify_successor_storage_migration_evidence",
        verify_successor,
    )
    monkeypatch.setattr(
        campaign_review_module,
        "verify_campaign_storage_migration",
        lambda *_args, **_kwargs: pytest.fail(
            "successor review must not bind the old attestation to the current campaign"
        ),
    )

    assert (
        _verify_review_storage_migration(
            migration,
            campaign=SimpleNamespace(),  # type: ignore[arg-type]
            campaign_dir=current,
            locator=SimpleNamespace(),  # type: ignore[arg-type]
            volume_probe=lambda _: VolumeIdentity("unused", "unused"),
            evidence_dir=predecessor,
        )
        == expected
    )
    assert calls == [predecessor]


def test_rolling_capacity_replay_uses_canonical_campaign_reserves() -> None:
    gib = 1024**3
    config = BenchmarkCampaignConfig.pilot(
        run_label="stage05.2_benchmark_attempt16",
        staging_root_alias="staging",
        archive_root_aliases=("archive",),
        selected_backend="native_cpu",
        selected_exact_backend="cpu_batch",
        selected_workers=4,
        native_profile="stage05.2-native-kernels-v1",
    )
    staging = VolumeIdentity(device_uuid="staging-ext4", filesystem="ext4")
    archive = VolumeIdentity(device_uuid="archive-9p", filesystem="9p")
    current = SimpleNamespace(
        archive_root_alias="archive",
        estimated_bytes=2 * gib,
        actual_bytes=3 * gib,
    )
    campaign = SimpleNamespace(
        run_label=config.run_label,
        scope=config.scope,
        storage_roots={"staging": staging, "archive": archive},
        batches=(current,),
    )

    assert campaign_review_module._expected_rolling_capacity_required(
        campaign=campaign,
        config=config,
        batch_index=0,
        phase="pre_dispatch",
        staging_root_alias="staging",
        archive_root_aliases=("archive",),
    ) == {
        "archive-9p": 52 * gib,
        "staging-ext4": 82 * gib,
    }
    assert campaign_review_module._expected_rolling_capacity_required(
        campaign=campaign,
        config=config,
        batch_index=0,
        phase="pre_archive",
        staging_root_alias="staging",
        archive_root_aliases=("archive",),
    ) == {
        "archive-9p": 53 * gib,
        "staging-ext4": 50 * gib,
    }
    assert campaign_review_module._expected_rolling_capacity_required(
        campaign=campaign,
        config=config,
        batch_index=0,
        phase="post_archive",
        staging_root_alias="staging",
        archive_root_aliases=("archive",),
    ) == {
        "archive-9p": 50 * gib,
        "staging-ext4": 50 * gib,
    }


@pytest.fixture(autouse=True)
def _verified_source_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        campaign_review_module,
        "verify_stage052_source_snapshot",
        lambda _root: dict(_SOURCE_SNAPSHOT),
    )


def _formal_geometry() -> tuple[CampaignGeometryRecord, ...]:
    records: list[CampaignGeometryRecord] = []
    ordinal = 0
    for record in sorted(
        BEST_KNOWN_VALUES,
        key=lambda item: (item.customer_count, item.instance),
    ):
        budgets = (30,) if record.customer_count < 100 else (30, 60, 300)
        checkpoint_count = 4 if record.customer_count < 100 else 16
        for seed in range(2014, 2024):
            ordinal += 1
            records.append(
                CampaignGeometryRecord(
                    shard_id=f"shard{ordinal:04d}",
                    batch_id="batch0001",
                    instance=record.instance,
                    seed=seed,
                    customer_count=record.customer_count,
                    budgets_seconds=budgets,
                    checkpoint_count=checkpoint_count,
                )
            )
    return tuple(records)


def test_formal_campaign_geometry_is_independently_exact() -> None:
    audit = validate_formal_geometry(_formal_geometry())

    assert audit.passed is True
    assert audit.shard_count == 920
    assert audit.axis_count == 2_040
    assert audit.declared_solver_seconds == 229_200
    assert audit.checkpoint_count == 10_400


def test_formal_campaign_geometry_rejects_a_missing_shard() -> None:
    audit = validate_formal_geometry(_formal_geometry()[:-1])

    assert audit.passed is False
    assert "920" in audit.detail


def test_campaign_planning_replay_requires_canonical_pilot_next_fit(
    tmp_path: Path,
) -> None:
    volume = VolumeIdentity(device_uuid="planning-volume", filesystem="apfs")
    staging = tmp_path / "staging"
    archive = tmp_path / "archive"
    staging.mkdir()
    archive.mkdir()
    locator = StorageRootLocator(
        {
            "staging": StorageRoot("staging", staging, volume),
            "archive": StorageRoot("archive", archive, volume),
        }
    )
    config = BenchmarkCampaignConfig.pilot(
        run_label="stage05.2_benchmark_attempt01",
        staging_root_alias="staging",
        archive_root_aliases=("archive",),
        selected_backend="native_cpu",
        selected_exact_backend="cpu_batch",
        selected_workers=2,
        native_profile="stage05.2-native-kernels-v1",
    )
    plan = config.build_plan()
    free_by_alias = {"staging": 200 * 1024**3, "archive": 200 * 1024**3}
    capacity = config.plan_archive_roots(
        plan,
        locator,
        free_bytes_by_alias=free_by_alias,
    )
    config_sha = hashlib.sha256(
        json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    campaign = CampaignManifest.planned(
        config=config,
        plan=plan,
        capacity=capacity,
        locator=locator,
        configuration_sha256=config_sha,
        prerequisite_review_sha256="a" * 64,
    )
    windows = [
        {
            "started_at_seconds": float(start),
            "duration_seconds": 30.0,
            "maximum_load1": 1.0,
            "maximum_unrelated_process_average_cores": 0.0,
            "logical_cpu_count": 12,
            "process_cpu_samples": [
                {"sampled_at_seconds": float(start), "cpu_seconds_by_pid": {}},
                {"sampled_at_seconds": float(start + 30), "cpu_seconds_by_pid": {}},
            ],
        }
        for start in (0, 30)
    ]
    preflight = {
        "schema_version": "stage05.2-campaign-preflight-v1",
        "run_label": config.run_label,
        "scope": "pilot",
        "power_source": "AC Power",
        "low_power_mode_enabled": False,
        "windows": windows,
        "volume_identities": locator.tracked_payload(("staging", "archive")),
        "free_bytes_by_alias": free_by_alias,
    }

    passed, detail = audit_campaign_planning(
        campaign=campaign,
        config=config,
        expected_plan=plan,
        plan_payload=plan.to_dict(),
        preflight_payload=preflight,
        capacity_payload=capacity.to_dict(),
        locator=locator,
    )

    assert passed, detail
    assert len(plan.batches) == 3
    counterfeit = plan.to_dict()
    counterfeit["batches"] = [
        {
            "batch_id": "batch0001",
            "shard_ids": [f"shard{index:04d}" for index in range(1, 37)],
            "estimated_bytes": 72 * 1024**3,
        }
    ]
    passed, detail = audit_campaign_planning(
        campaign=campaign,
        config=config,
        expected_plan=plan,
        plan_payload=counterfeit,
        preflight_payload=preflight,
        capacity_payload=capacity.to_dict(),
        locator=locator,
    )
    assert passed is False
    assert "signed plan" in detail


def test_streaming_event_audit_rejects_acceptance_after_deadline() -> None:
    events = (
        {
            "event_id": 1,
            "benchmark_axis": "wall_clock_30",
            "event_type": "route_evaluation",
            "kind": "exact_call",
            "evaluation_id": 1,
            "exact_started": True,
            "exact_completed": True,
            "started_at": 0.1,
            "completed_at": 0.2,
            "status": "completed_feasible",
        },
        {
            "event_id": 2,
            "benchmark_axis": "wall_clock_30",
            "event_type": "deadline_boundary",
            "timestamp_seconds": 30.0,
        },
        {
            "event_id": 3,
            "benchmark_axis": "wall_clock_30",
            "event_type": "candidate_state",
            "status": "accepted",
            "accepted": True,
            "global_best": False,
            "candidate_vehicle_delta": 0,
        },
    )

    audit = audit_streamed_events(events, {"wall_clock_30": 30})

    assert audit.passed is False
    assert "deadline" in audit.detail


def test_streaming_event_audit_uses_lane_boundary_for_pause_aware_budget() -> None:
    events = (
        {
            "event_id": 1,
            "benchmark_axis": "wall_clock_30",
            "lane": "wall_clock_30:legacy",
            "event_type": "candidate_state",
            "status": "accepted",
            "accepted": True,
            "global_best": False,
            "candidate_feasible": True,
            "candidate_vehicle_delta": 0,
            "timestamp_seconds": 30.35,
        },
        {
            "event_id": 2,
            "benchmark_axis": "wall_clock_30",
            "lane": "wall_clock_30:legacy",
            "event_type": "deadline_boundary",
            "timestamp_seconds": 30.0,
        },
    )

    audit = audit_streamed_events(events, {"wall_clock_30": 30})

    assert audit.passed


def test_native_axis_allows_deadline_interrupt_before_native_invocation() -> None:
    raw_axis = {
        "backend": "cpu_batch",
        "started_calls": 2,
        "completed_calls": 1,
        "backend_metrics": {
            "exact_calls": 2,
            "batch_launches": 2,
            "work_batches": 2,
            "native_invocations": 1,
            "native_fallbacks": 0,
            "launch_occupancies": [1, 1],
        },
    }
    trace_axis = {
        "result_summary": {
            "exact_started_calls": 2,
            "exact_completed_calls": 1,
            "exact_interrupted_calls": 1,
            "screening_statistics": {"native_protocol_fallbacks": 0},
        },
        "persistence_pipeline": {
            "mode": "bounded_async_thread",
            "queue_max_batches": 1,
            "writer_thread_switch_interval_seconds": 0.5,
            "submitted_batches": 1,
            "completed_batches": 1,
            "writer_active_nanoseconds": 10,
            "writer_cpu_nanoseconds": 5,
            "producer_active_nanoseconds": 10,
            "persistence_union_nanoseconds": 15,
            "solver_persistence_union_nanoseconds": 10,
            "solver_persistence_critical_path_nanoseconds": 8,
            "solver_producer_active_nanoseconds": 8,
            "solver_writer_cpu_nanoseconds": 5,
            "producer_wait_nanoseconds": 0,
            "peak_queued_batches": 1,
            "batch_ledger": [
                {
                    "ordinal": 0,
                    "row_count": 1,
                    "event_token_sha256": "a" * 64,
                }
            ],
        },
    }

    passed, detail = campaign_review_module._native_axis_valid(raw_axis, trace_axis)

    assert passed, detail


def test_streaming_event_audit_treats_interrupted_exact_as_deadline_boundary() -> None:
    events = (
        {
            "event_id": 1,
            "benchmark_axis": "wall_clock_30",
            "lane": "wall_clock_30:legacy",
            "event_type": "route_evaluation",
            "evaluation_id": 1,
            "status": "interrupted_deadline",
            "exact_started": True,
            "exact_completed": False,
            "started_at": 29.8,
            "completed_at": 30.2,
        },
        {
            "event_id": 2,
            "benchmark_axis": "wall_clock_30",
            "lane": "wall_clock_30:legacy",
            "event_type": "candidate_state",
            "status": "accepted",
            "accepted": True,
            "global_best": False,
            "candidate_feasible": True,
            "candidate_vehicle_delta": 0,
        },
    )

    audit = audit_streamed_events(events, {"wall_clock_30": 30})

    assert audit.deadline_axes == ("wall_clock_30",)
    assert audit.passed is False
    assert "accepted after deadline" in audit.detail


def test_streaming_event_audit_keeps_deadline_boundaries_lane_local() -> None:
    events = (
        {
            "event_id": 1,
            "benchmark_axis": "wall_clock_30",
            "lane": "wall_clock_30:legacy",
            "event_type": "deadline_boundary",
            "timestamp_seconds": 29.9,
        },
        {
            "event_id": 2,
            "benchmark_axis": "wall_clock_30",
            "lane": "wall_clock_30:constraint_lane",
            "event_type": "cache_event",
            "operation": "lookup_result",
            "lookup_result": "miss",
            "cache_key_digest": "constraint-route",
            "timestamp_seconds": 29.91,
        },
        {
            "event_id": 3,
            "benchmark_axis": "wall_clock_30",
            "lane": "wall_clock_30:constraint_lane",
            "event_type": "route_evaluation",
            "kind": "exact_call",
            "evaluation_id": 1,
            "exact_started": True,
            "exact_completed": True,
            "started_at": 29.92,
            "completed_at": 29.93,
            "cache_key_digest": "constraint-route",
            "status": "completed_feasible",
        },
        {
            "event_id": 4,
            "benchmark_axis": "wall_clock_30",
            "lane": "wall_clock_30:constraint_lane",
            "event_type": "cache_event",
            "operation": "store",
            "cache_key_digest": "constraint-route",
            "timestamp_seconds": 29.94,
        },
    )

    audit = audit_streamed_events(events, {"wall_clock_30": 30})

    assert audit.passed is True, audit.detail
    assert audit.deadline_axes == ("wall_clock_30",)


def test_streaming_event_audit_rejects_native_fallback() -> None:
    audit = audit_streamed_events(
        (
            {
                "event_id": 1,
                "benchmark_axis": "wall_clock_30",
                "event_type": "execution_error",
                "failure_reason": "native fallback to Python",
            },
        ),
        {"wall_clock_30": 30},
    )

    assert audit.passed is False
    assert "fallback" in audit.detail


def test_streaming_event_audit_accepts_ordered_cache_exact_candidate_flow() -> None:
    events = (
        {
            "event_id": 1,
            "benchmark_axis": "wall_clock_30",
            "event_type": "cache_event",
            "operation": "lookup_result",
            "lookup_result": "miss",
            "cache_key_digest": "route-a",
        },
        {
            "event_id": 2,
            "benchmark_axis": "wall_clock_30",
            "event_type": "route_evaluation",
            "kind": "exact_call",
            "evaluation_id": 1,
            "exact_started": True,
            "exact_completed": True,
            "started_at": 0.1,
            "completed_at": 0.2,
            "cache_key_digest": "route-a",
            "status": "completed_feasible",
        },
        {
            "event_id": 3,
            "benchmark_axis": "wall_clock_30",
            "event_type": "cache_event",
            "operation": "store",
            "cache_key_digest": "route-a",
        },
        {
            "event_id": 4,
            "benchmark_axis": "wall_clock_30",
            "event_type": "candidate_state",
            "timestamp_seconds": 0.3,
            "status": "accepted",
            "candidate_feasible": True,
            "accepted": True,
            "global_best": True,
            "candidate_vehicle_delta": -1,
            "candidate_route_keys": ["route-a"],
        },
    )

    audit = audit_streamed_events(events, {"wall_clock_30": 30})

    assert audit.passed is True
    assert audit.exact_started == audit.exact_completed == 1
    assert audit.accepted_candidates == audit.global_bests == 1


def test_streaming_event_audit_tracks_all_route_evaluation_ids_with_cache_hits() -> None:
    events = (
        {
            "event_id": 1,
            "benchmark_axis": "wall_clock_30",
            "event_type": "route_evaluation",
            "kind": "exact_call",
            "evaluation_id": 1,
            "exact_started": True,
            "exact_completed": False,
            "started_at": 0.1,
            "completed_at": None,
        },
        {
            "event_id": 2,
            "benchmark_axis": "wall_clock_30",
            "event_type": "route_evaluation",
            "kind": "cache_hit",
            "evaluation_id": 2,
            "exact_started": False,
            "exact_completed": False,
            "started_at": 0.2,
            "completed_at": 0.2,
        },
        {
            "event_id": 3,
            "benchmark_axis": "wall_clock_30",
            "event_type": "route_evaluation",
            "kind": "exact_call",
            "evaluation_id": 3,
            "exact_started": True,
            "exact_completed": False,
            "started_at": 0.3,
            "completed_at": None,
        },
    )

    audit = audit_streamed_events(events, {"wall_clock_30": 30})

    assert audit.passed is True
    assert audit.exact_started == 2


def test_campaign_reviewer_recomputes_async_batch_ledger() -> None:
    events = [
        {
            "event_type": "operator_call",
            "benchmark_axis": "wall_clock_30",
            "lane": "wall_clock_30:legacy",
            "iteration": iteration,
            "operator": "repair",
        }
        for iteration in range(3)
    ]
    digest = hashlib.sha256()
    for event in events:
        digest.update(
            orjson.dumps(
                campaign_review_module._pipeline_event_token_from_logical_row(  # noqa: SLF001
                    event
                )
            )
            + b"\n"
        )
    pipeline = {
        "batch_ledger": [
            {
                "ordinal": 0,
                "row_count": len(events),
                "event_token_sha256": digest.hexdigest(),
            }
        ]
    }
    trace_axes = {"wall_clock_30": {"persistence_pipeline": pipeline}}

    campaign_review_module._audit_async_persistence_ledgers(  # noqa: SLF001
        events, trace_axes
    )
    pipeline["batch_ledger"][0]["event_token_sha256"] = "f" * 64
    with pytest.raises(ArtifactIntegrityError, match="digest does not replay"):
        campaign_review_module._audit_async_persistence_ledgers(  # noqa: SLF001
            events, trace_axes
        )


def test_campaign_reviewer_replays_each_logical_event_stream_once() -> None:
    instance = parse_schneider(Path("data/schneider/c101C5.txt"))
    baseline = json.loads(
        Path(
            "experiments/baselines/stage00/solutions/"
            "c101C5-alns_exact_charging-2014.json"
        ).read_text(encoding="utf-8")
    )
    routes = baseline["routes"]
    report = validate_routes(instance, routes)
    assert report.feasible
    objective = SolutionObjective.from_report(instance, report).key
    customer_names = {customer.name for customer in instance.customers}
    full_route_keys = [
        "route:" + "|".join(f"{len(str(node))}:{node}" for node in route)
        for route in routes
    ]
    route_keys = [
        "route:"
        + "|".join(
            f"{len(str(node))}:{node}"
            for node in route
            if str(node) in customer_names
        )
        for route in routes
    ]
    events = [
        {
            "event_id": 1,
            "event_type": "candidate_state",
            "benchmark_axis": "wall_clock_30",
            "lane": "wall_clock_30:legacy",
            "iteration": 1,
            "status": "accepted",
            "candidate_feasible": True,
            "accepted": True,
            "global_best": True,
            "candidate_vehicle_delta": -1,
            "candidate_route_keys": route_keys,
            "candidate_full_route_keys": full_route_keys,
            "candidate_objective_key": list(objective),
            "timestamp_seconds": 0.25,
        }
    ]
    digest = hashlib.sha256()
    digest.update(
        orjson.dumps(
            campaign_review_module._pipeline_event_token_from_logical_row(  # noqa: SLF001
                events[0]
            )
        )
        + b"\n"
    )
    trace_axes = {
        "wall_clock_30": {
            "persistence_pipeline": {
                "batch_ledger": [
                    {
                        "ordinal": 0,
                        "row_count": 1,
                        "event_token_sha256": digest.hexdigest(),
                    }
                ]
            }
        }
    }
    legacy_audit = audit_streamed_events(events, {"wall_clock_30": 30})
    campaign_review_module._audit_async_persistence_ledgers(  # noqa: SLF001
        events, trace_axes
    )
    legacy_histories = summarize_streamed_global_bests(
        events,
        {"wall_clock_30": 30},
        instance=instance,
    )

    class SingleUseEvents:
        def __init__(self) -> None:
            self.iterations = 0

        def __iter__(self) -> Iterator[dict[str, object]]:
            self.iterations += 1
            if self.iterations != 1:
                raise AssertionError("logical event stream was replayed more than once")
            yield from events

    source = SingleUseEvents()
    replay = replay_streamed_shard_events(
        source,
        trace_axes,
        {"wall_clock_30": 30},
        instance=instance,
    )

    assert source.iterations == 1
    assert replay.audit.passed is True
    assert replay.audit.event_count == 1
    assert replay.audit == legacy_audit
    assert replay.global_best_histories == legacy_histories
    assert replay.global_best_histories["wall_clock_30"][0].objective_key == objective
    assert replay.logical_pass_count == 1


def test_spawned_shard_summary_rejects_raw_payload_fields() -> None:
    summary = campaign_review_module._ShardReplayProcessResult(  # noqa: SLF001
        run_label="stage05.2_benchmark_attempt99",
        batch_id="batch0001",
        shard_id="shard0001",
        instance="c101C5",
        seed=2014,
        replay_rows=[],
        checkpoints=[],
        logical_events=1,
        child_pid=123,
        child_peak_rss_bytes=456,
        pyarrow_allocated_bytes_after=0,
        scratch_cleaned=True,
        elapsed_seconds=0.5,
    )
    payload = json.loads(summary.to_json())
    payload["raw_events"] = [{"event_id": 1}]

    with pytest.raises(ArtifactIntegrityError, match="summary fields"):
        campaign_review_module._ShardReplayProcessResult.from_json(  # noqa: SLF001
            json.dumps(payload)
        )


def test_failed_spawned_shard_is_not_retried_and_cleans_scratch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scratch_root = tmp_path / "scratch"
    scratch_root.mkdir()
    batch_dir = tmp_path / "batch0001"
    batch_dir.mkdir()
    progress_path = tmp_path / "review-progress.jsonl"
    monkeypatch.setenv("STAGE052_REVIEW_TMPDIR", str(scratch_root))
    monkeypatch.setenv("STAGE052_REVIEW_PROGRESS_LOG", str(progress_path))

    with pytest.raises(ArtifactIntegrityError, match="sidecar"):
        campaign_review_module._replay_shard_in_fresh_process(  # noqa: SLF001
            batch_dir=batch_dir,
            shard_relative_path="c101C5/2014/missing_shard_manifest.json",
            benchmark_dir=Path("data/schneider"),
            scope="pilot",
            run_label="stage05.2_benchmark_attempt99",
            batch_id="batch0001",
            shard_id="shard0001",
            expected_instance="c101C5",
            expected_seed=2014,
        )

    progress = [
        json.loads(line)
        for line in progress_path.read_text(encoding="utf-8").splitlines()
    ]
    assert sum(row["event"] == "campaign_shard_replay_start" for row in progress) == 1
    child_failure = next(
        row for row in progress if row["event"] == "campaign_shard_replay_child_failed"
    )
    parent_failure = next(
        row for row in progress if row["event"] == "campaign_shard_replay_failed"
    )
    assert child_failure["child_pid"] != parent_failure["parent_pid"]
    assert child_failure["scratch_cleaned"] is True
    assert parent_failure["scratch_cleaned"] is True
    assert parent_failure["scratch_cleanup_recovered"] is True
    assert list(scratch_root.iterdir()) == []


def test_campaign_reviewer_restores_prepared_screening_empty_reason_token() -> None:
    token = campaign_review_module._pipeline_event_token_from_logical_row(  # noqa: SLF001
        {
            "event_type": "screening_decision",
            "benchmark_axis": "wall_clock_30",
            "lane": "wall_clock_30:constraint",
            "iteration": 1,
            "operator": "repair",
            "route_key": "route:2:C1",
            "decision_id": 2,
            "status": "pass",
            "reason": None,
        }
    )

    assert token[10] == ""


def test_axis_reconciliation_rejects_missing_deadline_and_exact_events() -> None:
    event_audit = audit_streamed_events(
        (
            {
                "event_id": 1,
                "benchmark_axis": "wall_clock_30",
                "event_type": "candidate_state",
                "accepted": False,
                "global_best": False,
            },
        ),
        {"wall_clock_30": 30},
    )
    raw_axes = {
        "wall_clock_30": {
            "runtime_seconds": 30.0,
            "termination_reason": "wall_clock_deadline",
            "started_calls": 1,
            "completed_calls": 1,
        }
    }

    audit = validate_axis_event_reconciliation(
        raw_axes,
        event_audit,
        customer_count=100,
    )

    assert audit.passed is False
    assert "exact-call" in audit.detail
    assert "deadline" in audit.detail


def test_axis_reconciliation_rejects_cross_axis_counter_cancellation() -> None:
    events = (
        {
            "event_id": 1,
            "benchmark_axis": "wall_clock_30",
            "event_type": "cache_event",
            "operation": "lookup_result",
            "lookup_result": "miss",
            "cache_key_digest": "route-a",
        },
        {
            "event_id": 2,
            "benchmark_axis": "wall_clock_30",
            "event_type": "route_evaluation",
            "evaluation_id": 1,
            "exact_started": True,
            "exact_completed": True,
            "started_at": 0.1,
            "completed_at": 0.2,
            "cache_key_digest": "route-a",
        },
        {
            "event_id": 3,
            "benchmark_axis": "wall_clock_60",
            "event_type": "candidate_state",
            "accepted": False,
            "global_best": False,
        },
    )
    event_audit = audit_streamed_events(
        events,
        {"wall_clock_30": 30, "wall_clock_60": 60},
    )
    raw_axes = {
        "wall_clock_30": {
            "runtime_seconds": 0.5,
            "termination_reason": "iteration_limit",
            "started_calls": 0,
            "completed_calls": 0,
        },
        "wall_clock_60": {
            "runtime_seconds": 0.5,
            "termination_reason": "iteration_limit",
            "started_calls": 1,
            "completed_calls": 1,
        },
    }

    audit = validate_axis_event_reconciliation(
        raw_axes,
        event_audit,
        customer_count=5,
    )

    assert event_audit.passed is True
    assert audit.passed is False
    assert "per-axis" in audit.detail


def test_streaming_audit_has_a_hard_pending_key_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "evrptw.experiments.stage052_campaign_review.MAX_STREAM_AUDIT_KEYS",
        2,
    )
    events = tuple(
        {
            "event_id": event_id,
            "benchmark_axis": "wall_clock_30",
            "event_type": "cache_event",
            "operation": "lookup_result",
            "lookup_result": "miss",
            "cache_key_digest": f"route-{event_id}",
        }
        for event_id in range(1, 4)
    )

    audit = audit_streamed_events(events, {"wall_clock_30": 30})

    assert audit.passed is False
    assert "bounded" in audit.detail


def test_global_best_stream_summary_keeps_only_checkpoint_visible_events() -> None:
    instance = parse_schneider(Path("data/schneider/c101C5.txt"))
    baseline = json.loads(
        Path(
            "experiments/baselines/stage00/solutions/c101C5-alns_exact_charging-2014.json"
        ).read_text(encoding="utf-8")
    )
    routes = baseline["routes"]
    report = validate_routes(instance, routes)
    assert report.feasible
    objective = SolutionObjective.from_report(instance, report).key
    full_route_keys = [
        "route:" + "|".join(f"{len(str(node))}:{node}" for node in route) for route in routes
    ]
    customer_names = {customer.name for customer in instance.customers}
    customer_sequences = [
        [node for node in route if node in customer_names] for route in routes
    ]
    route_keys = [
        "route:" + "|".join(f"{len(str(node))}:{node}" for node in route)
        for route in customer_sequences
    ]
    events = (
        {
            "event_type": "candidate_state",
            "benchmark_axis": "wall_clock_30",
            "accepted": True,
            "global_best": True,
            "timestamp_seconds": 0.25,
            "iteration": 1,
            "candidate_route_keys": route_keys,
            "candidate_full_route_keys": full_route_keys,
            "candidate_objective_key": list(objective),
        },
    )

    summaries = summarize_streamed_global_bests(
        events,
        {"wall_clock_30": 30},
        instance=instance,
    )

    assert len(summaries["wall_clock_30"]) == 1

    missing_full_routes = (
        {
            key: value
            for key, value in events[0].items()
            if key != "candidate_full_route_keys"
        },
    )
    with pytest.raises(ValueError, match="lacks complete candidate route identity"):
        summarize_streamed_global_bests(
            missing_full_routes,
            {"wall_clock_30": 30},
            instance=instance,
        )

    tampered = ({**events[0], "candidate_objective_key": [objective[0], objective[1] + 1, 0, 0]},)
    with pytest.raises(ValueError, match="routes/objective mismatch"):
        summarize_streamed_global_bests(
            tampered,
            {"wall_clock_30": 30},
            instance=instance,
        )

    wrong_projection = (
        {
            **events[0],
            "candidate_route_keys": [
                "route:" + "|".join(f"{len(node)}:{node}" for node in customer_sequences[0][1:]),
                *route_keys[1:],
            ],
        },
    )
    with pytest.raises(ValueError, match="complete routes/customer sequences mismatch"):
        summarize_streamed_global_bests(
            wrong_projection,
            {"wall_clock_30": 30},
            instance=instance,
        )


def test_streaming_audit_rejects_global_best_without_candidate_routes() -> None:
    audit = audit_streamed_events(
        (
            {
                "event_id": 1,
                "event_type": "candidate_state",
                "benchmark_axis": "wall_clock_30",
                "accepted": True,
                "global_best": True,
                "status": "accepted",
                "candidate_feasible": True,
                "candidate_vehicle_delta": 0,
                "timestamp_seconds": 0.25,
                "iteration": 1,
            },
        ),
        {"wall_clock_30": 30},
    )

    assert audit.passed is False
    assert "route identity" in audit.detail


def test_pilot_can_never_publish_formal_readiness() -> None:
    assert review_status_for_scope("pilot", passed=True) == ("READY_FOR_STAGE052_FORMAL_BENCHMARK")
    assert review_status_for_scope("formal", passed=True) == "READY_FOR_STAGE05_3"
    assert review_status_for_scope("formal", passed=False) == "NOT_READY"


def _selection_inputs() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    native = {
        "enabled": True,
        "exact_charging": True,
        "screening": True,
        "propagation": True,
        "distance_matrix": True,
        "abi_version": "stage05.2-native-kernels-v1",
        "context_policy": "pack_once_per_solve",
        "failure_policy": "fail_fast_no_fallback",
    }
    runtime = {
        "schema_version": "stage05.2-runtime-identity-v2",
        "repository_revision": "a" * 40,
        "repository_dirty": False,
        "wheel_sha256": "1" * 64,
        "python_executable_sha256": "2" * 64,
        "native_extension_sha256": "3" * 64,
        "dependency_manifest_sha256": "4" * 64,
        "installed_distribution_sha256": "5" * 64,
        "installed_editable": False,
    }
    metadata: dict[str, object] = {
        "run_label": "stage05.2_accelerator_pilot_attempt02",
        "component": "accelerator_pilot",
        "scope": "performance",
        "backend": "cpu_batch",
        "execution_backend": "native_cpu",
        "worker_count": 2,
        "optimization_profile": "native",
        "native_kernel_config": native,
        "repository_revision": "a" * 40,
        "repository_dirty": False,
        "configuration_sha256": "6" * 64,
        "runtime_identity": runtime,
        "performance_provenance": {
            "operator_surface": {"stage02_config_sha256": "7" * 64},
            "instance_sha256": {"c101_21": "8" * 64},
        },
        "storage_policy_version": "artifact-storage-v2",
        "screening_schema_version": "screening_decisions_v3",
    }
    review: dict[str, object] = {
        "run_label": metadata["run_label"],
        "component": metadata["component"],
        "scope": metadata["scope"],
        "status": "READY_FOR_STAGE052_BENCHMARK",
        "raw_manifest_sha256": "9" * 64,
        "accelerator_decision": "GPU_NOT_JUSTIFIED",
        "campaign_prerequisite_review_sha256": "e" * 64,
        "selected_backend": "native_cpu",
        "selected_exact_backend": "cpu_batch",
        "selected_workers": 2,
        "selected_optimization_profile": "native",
        "native_configuration": native,
    }
    identity: dict[str, object] = {
        "run_label": metadata["run_label"],
        "component": metadata["component"],
        "scope": metadata["scope"],
        "status": review["status"],
        "repository_revision": metadata["repository_revision"],
        "configuration_sha256": metadata["configuration_sha256"],
        "raw_manifest_sha256": review["raw_manifest_sha256"],
        "review_manifest_sha256": "b" * 64,
    }
    return metadata, review, identity


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_signed_json(path: Path, payload: object) -> None:
    _write_json(path, payload)
    path.with_suffix(".sha256").write_text(
        hashlib.sha256(path.read_bytes()).hexdigest() + "\n",
        encoding="utf-8",
    )


def _build_accepted_f02(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    raw_dir = tmp_path / "stage05.2_accelerator_pilot_attempt02"
    metadata, review, _ = _selection_inputs()
    pilot_instances = (
        "c101C5",
        "r105C5",
        "rc105C5",
        "c104C10",
        "r103C10",
        "rc102C10",
        "c106C15",
        "r105C15",
        "rc103C15",
        "c101_21",
        "r101_21",
        "rc101_21",
    )
    provenance = metadata["performance_provenance"]
    assert isinstance(provenance, dict)
    provenance["instance_sha256"] = {
        instance: hashlib.sha256(
            (Path("data/schneider") / f"{instance}.txt").read_bytes()
        ).hexdigest()
        for instance in pilot_instances
    }
    config = tmp_path / "stage052.toml"
    config.write_text("[stage05_2]\nprofile = 'native'\n", encoding="utf-8")
    config_sha256 = hashlib.sha256(config.read_bytes()).hexdigest()
    metadata["configuration_sha256"] = config_sha256
    writer = ArtifactBundleWriter(
        raw_dir,
        ArtifactRunContext(
            "stage05.2",
            "accelerator_pilot",
            "stage05.2_accelerator_pilot_attempt02",
        ),
        ArtifactStorageConfig(
            storage_policy_version="artifact-storage-v2",
            screening_schema_version="screening_decisions_v3",
        ),
    )
    writer.write_control(metadata=metadata, configuration_path=config)
    bundle = writer.finalize()
    review["raw_manifest_sha256"] = hashlib.sha256(bundle.manifest_path.read_bytes()).hexdigest()
    review.update(
        {
            "schema_version": "stage05.2-review-v1",
            "gates": {"accelerator_decision": {"passed": True, "detail": "passed"}},
        }
    )
    generation_id = hashlib.sha256(b"passed\n\0passed\n").hexdigest()
    generation = raw_dir / "review/generations" / generation_id
    generation.mkdir(parents=True)
    files: dict[str, str] = {}
    for name in ("review_findings.csv", "review_report.md"):
        path = generation / name
        path.write_text("passed\n", encoding="utf-8")
        relative = path.relative_to(raw_dir / "review").as_posix()
        files[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    review["files"] = files
    _write_json(raw_dir / "review/review_manifest.json", review)
    return raw_dir, metadata


def _build_complete_pilot_campaign(
    tmp_path: Path,
    *,
    batch_transfer_mode: str = "same_volume_atomic_rename",
) -> tuple[Path, StorageRootLocator, VolumeIdentity]:
    prerequisite_dir, predecessor_metadata = _build_accepted_f02(tmp_path)
    prerequisite_review = prerequisite_dir / "review/review_manifest.json"
    prerequisite_hash = hashlib.sha256(prerequisite_review.read_bytes()).hexdigest()
    run_label = "stage05.2_benchmark_attempt01"
    campaign_dir = tmp_path / run_label
    staging_root = tmp_path / "staging"
    archive_root = tmp_path / "archive"
    staging_root.mkdir()
    archive_root.mkdir()
    volume = VolumeIdentity(device_uuid="synthetic-apfs-volume", filesystem="apfs")
    locator = StorageRootLocator(
        {
            "staging": StorageRoot("staging", staging_root, volume),
            "archive": StorageRoot("archive", archive_root, volume),
        }
    )
    batch_dir = archive_root / run_label / "batch0001"
    writer = ArtifactBundleWriter(
        batch_dir,
        ArtifactRunContext("stage05.2", "benchmark", run_label),
        ArtifactStorageConfig(
            storage_policy_version="artifact-storage-v2",
            screening_schema_version="screening_decisions_v3",
        ),
    )
    campaign_configuration_sha256 = "c" * 64
    metadata = {
        "run_label": run_label,
        "component": "benchmark",
        "scope": "pilot",
        "backend": "cpu_batch",
        "execution_backend": "native_cpu",
        "worker_count": 2,
        "worker_process_lifecycle": "one_shard_per_spawned_process",
        "worker_runtime_warmup": "in_memory_arrow_zstd1",
        "native_profile": "stage05.2-native-kernels-v1",
        "native_kernel_config": predecessor_metadata["native_kernel_config"],
        "repository_revision": predecessor_metadata["repository_revision"],
        "repository_dirty": False,
        "configuration_sha256": predecessor_metadata["configuration_sha256"],
        "campaign_configuration_sha256": campaign_configuration_sha256,
        "campaign_prerequisite_review_sha256": prerequisite_hash,
        "runtime_identity": predecessor_metadata["runtime_identity"],
        "source_snapshot": dict(_SOURCE_SNAPSHOT),
        "performance_provenance": predecessor_metadata["performance_provenance"],
        "storage_policy_version": "artifact-storage-v2",
        "screening_schema_version": "screening_decisions_v3",
        "staging_root_alias": "staging",
        "archive_root_aliases_exercised": ["archive"],
        "failure_handling_drill_passed": True,
        "raw_replay_drill_passed": True,
    }
    writer.write_control(metadata=metadata)
    bks_counts = {item.instance: item.customer_count for item in BEST_KNOWN_VALUES}
    instances = (
        "c101C5",
        "r105C5",
        "rc105C5",
        "c104C10",
        "r103C10",
        "rc102C10",
        "c106C15",
        "r105C15",
        "rc103C15",
        "c101_21",
        "r101_21",
        "rc101_21",
    )
    identities = sorted(
        (bks_counts[instance], instance, seed)
        for instance in instances
        for seed in (2014, 2015, 2016)
    )
    per_run: list[dict[str, object]] = []
    event_row_count = 0
    for ordinal, (_, instance_name, seed) in enumerate(identities):
        baseline = json.loads(
            (
                Path("experiments/baselines/stage00/solutions")
                / f"{instance_name}-alns_exact_charging-{seed}.json"
            ).read_text(encoding="utf-8")
        )
        routes = baseline["routes"]
        instance = parse_schneider(Path("data/schneider") / f"{instance_name}.txt")
        report = validate_routes(instance, routes)
        assert report.feasible
        objective = SolutionObjective.from_report(instance, report).key
        shard = writer.open_v2_shard(
            instance=instance_name,
            seed=seed,
            shard_ordinal=ordinal,
            worker_identity=f"pid-{101 + ordinal}",
        )
        critical_events: list[dict[str, object]] = [
            {
                "event_type": "candidate_state",
                "benchmark_axis": "wall_clock_30",
                "lane": "wall_clock_30:legacy",
                "operator": "synthetic_rejected",
                "iteration": 1,
                "status": "rejected",
                "candidate_feasible": True,
                "accepted": False,
                "global_best": False,
                "candidate_vehicle_delta": 0,
                "timestamp_seconds": 0.25,
            }
        ]
        if bks_counts[instance_name] == 100:
            critical_events.append(
                {
                    "event_type": "deadline_boundary",
                    "benchmark_axis": "wall_clock_30",
                    "lane": "wall_clock_30:legacy",
                    "operator": "deadline",
                    "iteration": 1000,
                    "timestamp_seconds": 30.0,
                }
            )
        event_row_count += len(critical_events)
        shard.append(
            route_dictionary={},
            critical_events=critical_events,
        )
        checkpoints = AnytimeCheckpoint.for_axis(
            instance=instance_name,
            seed=seed,
            axis_budget_seconds=30,
            initial_objective_key=objective,
            accepted_global_bests=(),
            max_iterations_completed_at_seconds=(0.5 if bks_counts[instance_name] < 100 else None),
            final_objective_key=(objective if bks_counts[instance_name] < 100 else None),
        )
        checkpoint_path = (
            batch_dir
            / instance_name
            / str(seed)
            / f"{run_label}_anytime_checkpoints_{instance_name}_{seed}.parquet"
        )
        pq.write_table(
            pa.Table.from_pylist(
                [
                    {
                        **item.to_dict(),
                        "objective_key": {
                            "vehicle_count": item.objective_key[0],
                            "total_distance": item.objective_key[1],
                            "total_charging_time": item.objective_key[2],
                            "charging_count": item.objective_key[3],
                        },
                    }
                    for item in checkpoints
                ],
                schema=ANYTIME_CHECKPOINT_SCHEMA,
            ),
            checkpoint_path,
        )
        writer.record_existing_file(
            checkpoint_path,
            artifact_type="anytime_checkpoints",
            artifact_subtype="checkpoint_v1",
            storage_format="parquet",
            row_count=len(checkpoints),
            schema_fingerprint=_schema_fingerprint(ANYTIME_CHECKPOINT_SCHEMA),
        )
        raw_axis = {
            "backend": "cpu_batch",
            "backend_metrics": {
                "exact_calls": 0,
                "batch_launches": 0,
                "work_batches": 0,
                "native_invocations": 0,
                "native_fallbacks": 0,
                "launch_occupancies": [],
            },
            "started_calls": 0,
            "completed_calls": 0,
            "objective_key": list(objective),
            "initial_objective_key": list(objective),
            "valid": True,
            "validator_passed": True,
            "runtime_seconds": (0.5 if bks_counts[instance_name] < 100 else 30.0),
            "termination_reason": (
                "iteration_limit" if bks_counts[instance_name] < 100 else "wall_clock_deadline"
            ),
            "iteration_limit_completed_at_seconds": (
                0.5 if bks_counts[instance_name] < 100 else None
            ),
            "effective_iterations": 1000,
        }
        solution_axis = {
            "routes": routes,
            "objective_key": list(objective),
            "initial_routes": routes,
            "initial_objective_key": list(objective),
            "feasible": True,
        }
        trace_axis = {
            "result_summary": {
                "exact_started_calls": 0,
                "exact_completed_calls": 0,
                "exact_interrupted_calls": 0,
                "screening_statistics": {"native_protocol_fallbacks": 0},
            },
            "persistence_pipeline": {
                "mode": "bounded_async_thread",
                "queue_max_batches": 1,
                "writer_thread_switch_interval_seconds": (
                    campaign_review_module
                    .STAGE052_WRITER_THREAD_SWITCH_INTERVAL_SECONDS
                ),
                "submitted_batches": 1,
                "completed_batches": 1,
                "writer_active_nanoseconds": 1,
                "writer_cpu_nanoseconds": 1,
                "producer_active_nanoseconds": 1,
                "persistence_union_nanoseconds": 1,
                "solver_persistence_union_nanoseconds": 1,
                "solver_persistence_critical_path_nanoseconds": 1,
                "solver_producer_active_nanoseconds": 1,
                "solver_writer_cpu_nanoseconds": 1,
                "producer_wait_nanoseconds": 0,
                "peak_queued_batches": 1,
                "batch_ledger": [
                    {
                        "ordinal": 0,
                        "row_count": len(critical_events),
                        "event_token_sha256": hashlib.sha256(
                            b"".join(
                                orjson.dumps(
                                    campaign_review_module
                                    ._pipeline_event_token_from_logical_row(event)  # noqa: SLF001
                                )
                                + b"\n"
                                for event in critical_events
                            )
                        ).hexdigest(),
                    }
                ],
            },
        }
        shard.finalize(
            raw_payload={"axes": {"wall_clock_30": raw_axis}},
            solution_payload={"axes": {"wall_clock_30": solution_axis}},
            trace_payload={
                "campaign_trace_schema_version": "stage05.2-campaign-trace-v1",
                "run_label": run_label,
                "instance": instance_name,
                "seed": seed,
                "axes": {"wall_clock_30": trace_axis},
            },
            environment_payload={},
        )
        per_run.append(
            {
                "instance": instance_name,
                "seed": seed,
                "axis": "wall_clock_30",
                "vehicle_count": objective[0],
                "total_distance": objective[1],
                "total_charging_time": objective[2],
                "charging_count": objective[3],
                "native_fallbacks": 0,
                "native_protocol_fallbacks": 0,
                "solver_seconds": 30.0,
                "artifact_persistence_seconds": 1.0,
            }
        )
    control = batch_dir / "control"
    per_run_path = control / "per_run_results.csv"
    fields = list(per_run[0])
    import csv

    with per_run_path.open("w", newline="", encoding="utf-8") as handle:
        csv_writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        csv_writer.writeheader()
        csv_writer.writerows(per_run)
    writer.record_existing_file(
        per_run_path,
        artifact_type="per_run_results",
        row_count=len(per_run),
    )
    timing_path = control / "batch_timing_evidence.json"
    _write_json(
        timing_path,
        {
            "schema_version": "stage05.2-timing-evidence-v1",
            "run_label": run_label,
            "component": "benchmark",
            "rows": [
                {
                    "instance": row["instance"],
                    "seed": row["seed"],
                    "axis": row["axis"],
                    "solver_started_ns": 0,
                    "solver_completed_ns": 30_000_000_000,
                    "finalize_started_ns": 0,
                    "finalize_completed_ns": 36_000_000_000,
                    "live_stream_persistence_ns": 0,
                    "axis_event_count": 1,
                    "total_event_count": 36,
                    "axis_count": 36,
                }
                for row in per_run
            ],
        },
    )
    writer.record_existing_file(timing_path, artifact_type="timing_evidence")
    resource = {
        "schema_version": STAGE052_RESOURCE_SCHEMA_VERSION,
        "run_label": run_label,
        "component": "benchmark",
        "configured_worker_count": 2,
        "measurement_scope": STAGE052_RESOURCE_MEASUREMENT_SCOPE,
        "run_wall_seconds": 1080.0,
        "sample_interval_seconds": 0.05,
        "parent_pid": 100,
        "descendant_pids": list(range(101, 137)),
        "aggregate_peak_rss_bytes": 3_000_000,
        "process_peak_rss_bytes": {
            "100": 1_000_000,
            **{str(pid): 1_000_000 for pid in range(101, 137)},
        },
        "mean_active_cores": 1.0,
        "peak_active_cores": 2.0,
        "load1_min": 0.5,
        "load1_mean": 1.0,
        "load1_max": 1.5,
        "load1_sample_count": 2,
        "sample_count": 2,
        "status": "complete",
    }
    resource_path = control / "batch_resource_summary.json"
    _write_json(resource_path, resource)
    writer.record_existing_file(resource_path, artifact_type="resource_summary")
    power_path = control / "batch_power_load.json"
    _write_json(
        power_path,
        {
            "schema_version": "stage05.2-batch-power-load-v1",
            "run_label": run_label,
            "batch_id": "batch0001",
            "status": "complete",
            "native_power_boundary": {
                "schema_version": "stage05.2-native-power-boundary-v1",
                "before": {
                    "ac_online": True,
                    "battery_saver": False,
                    "battery_life_percent": 50,
                    "battery_flag": 8,
                    "active_power_scheme": "balanced-guid",
                },
                "after": {
                    "ac_online": True,
                    "battery_saver": False,
                    "battery_life_percent": 75,
                    "battery_flag": 1,
                    "active_power_scheme": "balanced-guid",
                },
                "stable_invariants": [
                    "ac_online",
                    "battery_saver",
                    "active_power_scheme",
                ],
                "invariants_unchanged": True,
            },
            "preflight": {
                "power_source": "AC Power",
                "low_power_mode_enabled": False,
                "windows": [
                    {
                        "started_at_seconds": 0.0,
                        "duration_seconds": 30.0,
                        "maximum_load1": 1.0,
                        "maximum_unrelated_process_average_cores": 0.0,
                        "logical_cpu_count": 24,
                        "process_cpu_samples": [
                            {"sampled_at_seconds": 0.0, "cpu_seconds_by_pid": {}},
                            {"sampled_at_seconds": 30.0, "cpu_seconds_by_pid": {}},
                        ],
                    },
                    {
                        "started_at_seconds": 30.0,
                        "duration_seconds": 30.0,
                        "maximum_load1": 1.0,
                        "maximum_unrelated_process_average_cores": 0.0,
                        "logical_cpu_count": 24,
                        "process_cpu_samples": [
                            {"sampled_at_seconds": 30.0, "cpu_seconds_by_pid": {}},
                            {"sampled_at_seconds": 60.0, "cpu_seconds_by_pid": {}},
                        ],
                    },
                ],
            },
            "runtime": {
                "sample_count": 2,
                "power_source_violations": 0,
                "low_power_mode_violations": 0,
                "maximum_load1": 5.9,
                "maximum_permitted_load1": 20.0,
                "maximum_unrelated_process_average_cores": 0.0,
                "logical_cpu_count": 24,
                "process_cpu_samples": [
                    {"sampled_at_seconds": 0.0, "cpu_seconds_by_pid": {}},
                    {"sampled_at_seconds": 30.0, "cpu_seconds_by_pid": {}},
                ],
            },
        },
    )
    writer.record_existing_file(power_path, artifact_type="power_load")
    bundle = writer.finalize()
    attribution = Stage052PersistenceAttribution(
        run_label=run_label,
        component="benchmark",
        scope="pilot",
        subject_id="batch0001",
        primary_manifest_relative_path=bundle.manifest_path.relative_to(batch_dir).as_posix(),
        primary_manifest_sha256=hashlib.sha256(bundle.manifest_path.read_bytes()).hexdigest(),
        solver_seconds=1080.0,
        shard_persistence_seconds=36.0,
        control_intervals=tuple(
            PersistenceInterval(label, index, index)
            for index, label in enumerate(
                (
                    "batch_write_control",
                    "batch_preflight_control",
                    "batch_timing_and_per_run_control",
                    "batch_resource_control",
                    "batch_runtime_control",
                    "batch_power_load_control",
                    "batch_primary_manifest_finalize",
                ),
                start=1,
            )
        ),
    )
    attribution_path = control / f"{run_label}_batch0001_persistence_attribution.json"
    _write_signed_json(attribution_path, attribution.to_dict())
    shard_manifests = sorted(batch_dir.glob("*/*/*_shard_manifest_*.json"))
    shard_hashes: dict[str, str] = {}
    shard_bytes: dict[str, int] = {}
    for path in shard_manifests:
        payload = json.loads(path.read_text(encoding="utf-8"))
        shard_id = f"shard{int(payload['shard_ordinal']) + 1:04d}"
        shard_hashes[shard_id] = hashlib.sha256(path.read_bytes()).hexdigest()
        shard_bytes[shard_id] = sum(
            item.stat().st_size for item in path.parent.iterdir() if item.is_file()
        )
    verified_batch = BatchManifest(
        run_label=run_label,
        batch_id="batch0001",
        status="verified",
        root_alias="staging",
        archive_root_alias="archive",
        logical_path=f"{run_label}/batch0001",
        volume=volume,
        shard_ids=tuple(f"shard{index:04d}" for index in range(1, 37)),
        estimated_bytes=directory_byte_count(batch_dir),
        checksum_sha256=directory_checksum(batch_dir),
        actual_bytes=directory_byte_count(batch_dir),
        row_count=event_row_count,
        physical_schema="screening_decisions_v3",
        resource_summary_sha256=hashlib.sha256(resource_path.read_bytes()).hexdigest(),
        persistence_attribution_sha256=hashlib.sha256(attribution_path.read_bytes()).hexdigest(),
        control_persistence_seconds=attribution.control_persistence_seconds,
        persistence_ratio=attribution.persistence_ratio,
        shard_manifest_sha256_by_id=shard_hashes,
        shard_actual_bytes_by_id=shard_bytes,
    )
    verified_manifest_path = batch_dir / "batch_manifest.json"
    _write_signed_json(verified_manifest_path, verified_batch.to_dict())
    verified_manifest_sha = hashlib.sha256(verified_manifest_path.read_bytes()).hexdigest()
    batch = verified_batch.mark_archived(
        root_alias="archive",
        volume=volume,
        transfer_mode=batch_transfer_mode,
        archive_transfer_seconds=1.0,
    )
    _write_signed_json(batch_dir / "batch_manifest.json", batch.to_dict())
    archived_manifest_sha = hashlib.sha256(
        (batch_dir / "batch_manifest.json").read_bytes()
    ).hexdigest()
    envelope = BatchPersistenceEnvelope(
        run_label=run_label,
        batch_id="batch0001",
        base_attribution_sha256=hashlib.sha256(attribution_path.read_bytes()).hexdigest(),
        verified_manifest_sha256=verified_manifest_sha,
        archived_manifest_sha256=archived_manifest_sha,
        solver_seconds=attribution.solver_seconds,
        base_persistence_seconds=attribution.total_persistence_seconds,
        state_intervals=(
            PersistenceInterval("verified_batch_manifest_write", 100, 100),
            PersistenceInterval("archived_batch_manifest_write", 101, 101),
        ),
    )
    envelope_path = batch_dir / "batch_persistence_envelope.json"
    _write_signed_json(envelope_path, envelope.to_dict())
    campaign = CampaignManifest(
        run_label=run_label,
        status="complete",
        scope="pilot",
        configuration_sha256=campaign_configuration_sha256,
        prerequisite_review_sha256=prerequisite_hash,
        selected_backend="native_cpu",
        selected_exact_backend="cpu_batch",
        selected_workers=2,
        native_profile="stage05.2-native-kernels-v1",
        storage_policy_version="artifact-storage-v2",
        screening_schema_version="screening_decisions_v3",
        storage_roots={"staging": volume, "archive": volume},
        shard_count=36,
        axis_count=36,
        declared_solver_seconds=1080,
        checkpoint_count=144,
        batches=(batch,),
        batch_persistence_envelope_sha256_by_id={
            "batch0001": hashlib.sha256(envelope_path.read_bytes()).hexdigest()
        },
    )
    campaign_dir.mkdir()
    campaign_path = campaign_dir / "campaign_manifest.json"
    _write_signed_json(campaign_path, campaign.to_dict())
    top_writer = ArtifactBundleWriter(
        campaign_dir,
        ArtifactRunContext("stage05.2", "benchmark", run_label),
        ArtifactStorageConfig(
            storage_policy_version="artifact-storage-v2",
            screening_schema_version="screening_decisions_v3",
        ),
    )
    top_writer.write_control(
        metadata={
            "run_label": run_label,
            "component": "benchmark",
            "scope": "pilot",
            "staging_root_alias": "staging",
            "planned_archive_root_aliases": ["archive"],
            "source_snapshot": dict(_SOURCE_SNAPSHOT),
            "persistence_attribution": "primary_active_writes_v1",
        },
        configuration_path=Path("configs/stage052_performance.toml"),
    )
    top_writer.record_existing_file(campaign_path, artifact_type="campaign_manifest")
    top_writer.record_existing_file(
        campaign_path.with_suffix(".sha256"),
        artifact_type="campaign_manifest_sidecar",
        storage_format="sha256_control",
    )
    top_control = campaign_dir / "control"
    rolling_path = top_control / f"{run_label}_rolling_capacity.json"
    rolling_observations = [
        {
            "schema_version": "stage05.2-rolling-capacity-v1",
            "run_label": run_label,
            "batch_id": "batch0001",
            "phase": phase,
            "free_bytes_by_device": {volume.device_uuid: 100 * 1024**3},
            "required_bytes_by_device": {
                volume.device_uuid: (
                    82 * 1024**3 + batch.estimated_bytes
                    if phase == "pre_dispatch"
                    else 82 * 1024**3
                )
            },
            "passed": True,
        }
        for phase in ("pre_dispatch", "pre_archive", "post_archive")
    ]
    _write_signed_json(
        rolling_path,
        {
            "schema_version": "stage05.2-rolling-capacity-summary-v1",
            "run_label": run_label,
            "scope": "pilot",
            "status": "complete",
            "passed": True,
            "observations": rolling_observations,
        },
    )
    rolling_journals: list[Path] = []
    for ordinal, observation in enumerate(rolling_observations, start=1):
        journal = (
            top_control
            / "rolling_capacity_observations"
            / f"{ordinal:04d}_batch0001_{observation['phase']}.json"
        )
        _write_signed_json(journal, observation)
        rolling_journals.append(journal)
    planned_batch = replace(
        batch,
        status="planned",
        root_alias="staging",
        checksum_sha256=None,
        actual_bytes=None,
        row_count=None,
        physical_schema=None,
        resource_summary_sha256=None,
        persistence_attribution_sha256=None,
        control_persistence_seconds=None,
        persistence_ratio=None,
        shard_manifest_sha256_by_id=None,
        shard_actual_bytes_by_id=None,
        transfer_mode=None,
        archive_transfer_seconds=None,
    )
    planned_campaign = replace(
        campaign,
        status="planned",
        batches=(planned_batch,),
        batch_persistence_envelope_sha256_by_id={},
    )
    failed_batch = planned_batch.mark_failed("injected campaign failure-handling drill")
    failed_campaign = planned_campaign.with_batch(failed_batch).mark_failed(
        "injected campaign failure-handling drill"
    )
    partial_path = top_control / "failure_state_drill" / "injected_partial_payload.bin"
    partial_path.parent.mkdir(parents=True)
    deterministic_payload = hashlib.sha256(run_label.encode("utf-8")).digest() * 256
    partial_path.write_bytes(deterministic_payload[: len(deterministic_payload) // 2])
    failure_path = top_control / f"{run_label}_failure_state_drill.json"
    _write_signed_json(
        failure_path,
        {
            "schema_version": "stage05.2-failure-state-drill-v1",
            "run_label": run_label,
            "status": "passed",
            "failure_type": "injected_partial_write",
            "expected_total_bytes": len(deterministic_payload),
            "injected_after_bytes": len(deterministic_payload) // 2,
            "partial_file_relative_path": partial_path.relative_to(campaign_dir).as_posix(),
            "partial_file_sha256": hashlib.sha256(partial_path.read_bytes()).hexdigest(),
            "failed_batch": failed_batch.to_dict(),
            "failed_campaign": failed_campaign.to_dict(),
        },
    )
    raw_replay_path = top_control / f"{run_label}_raw_replay_drill.json"
    _write_signed_json(
        raw_replay_path,
        {
            "schema_version": "stage05.2-raw-replay-drill-v1",
            "run_label": run_label,
            "batch_count": 1,
            "passed": True,
            "results": [
                {
                    "batch_id": "batch0001",
                    "raw_manifest_sha256": hashlib.sha256(
                        bundle.manifest_path.read_bytes()
                    ).hexdigest(),
                    "directory_checksum_sha256": batch.checksum_sha256,
                    "actual_bytes": batch.actual_bytes,
                    "artifact_count": len(ArtifactReader(batch_dir).manifest["artifacts"]),
                }
            ],
        },
    )
    dry_logical_path = f"{run_label}/archive_dry_run/archive"
    dry_dir = archive_root.joinpath(*Path(dry_logical_path).parts)
    dry_dir.mkdir(parents=True)
    dry_payload = dry_dir / "archive_probe.bin"
    dry_payload.write_bytes(f"{run_label}:archive\n".encode())
    dry_payload_sha = hashlib.sha256(dry_payload.read_bytes()).hexdigest()
    dry_batch = BatchManifest(
        run_label=run_label,
        batch_id="batch9001",
        status="archived",
        root_alias="archive",
        archive_root_alias="archive",
        logical_path=dry_logical_path,
        volume=volume,
        shard_ids=("shard9001",),
        estimated_bytes=directory_byte_count(dry_dir),
        checksum_sha256=directory_checksum(dry_dir),
        actual_bytes=directory_byte_count(dry_dir),
        row_count=0,
        physical_schema="archive_dry_run_v1",
        resource_summary_sha256=dry_payload_sha,
        persistence_attribution_sha256=dry_payload_sha,
        control_persistence_seconds=0.0,
        persistence_ratio=0.0,
        shard_manifest_sha256_by_id={"shard9001": dry_payload_sha},
        shard_actual_bytes_by_id={"shard9001": directory_byte_count(dry_dir)},
        transfer_mode="same_volume_atomic_rename",
        archive_transfer_seconds=0.0,
    )
    _write_signed_json(dry_dir / "batch_manifest.json", dry_batch.to_dict())
    archive_dry_run_path = top_control / f"{run_label}_archive_dry_run.json"
    _write_signed_json(
        archive_dry_run_path,
        {
            "schema_version": "stage05.2-archive-dry-run-v1",
            "run_label": run_label,
            "staging_root_alias": "staging",
            "archive_root_aliases_exercised": ["archive"],
            "passed": True,
            "results": [
                {
                    "archive_root_alias": "archive",
                    "logical_path": dry_logical_path,
                    "volume_identity": volume.to_dict(),
                    "transfer_mode": "same_volume_atomic_rename",
                    "archive_transfer_seconds": 0.0,
                    "checksum_sha256": dry_batch.checksum_sha256,
                    "actual_bytes": dry_batch.actual_bytes,
                }
            ],
        },
    )
    publication_path = top_control / f"{run_label}_publication_dry_run.json"
    dry_payload_bytes = b'{"status":"dry_run"}\n'
    dry_payload_digest = hashlib.sha256(dry_payload_bytes).hexdigest()
    dry_trusted = {
        "schema_version": "stage05.2-publication-dry-run-manifest-v1",
        "run_label": run_label,
        "payload_relative_path": "generation/payload.json",
        "payload_sha256": dry_payload_digest,
        "status": "READY_DRY_RUN",
    }
    _write_signed_json(
        publication_path,
        {
            "schema_version": "stage05.2-publication-dry-run-v1",
            "run_label": run_label,
            "payload_sha256": dry_payload_digest,
            "trusted_manifest_sha256": hashlib.sha256(
                (json.dumps(dry_trusted, indent=2, sort_keys=True) + "\n").encode()
            ).hexdigest(),
            "trusted_manifest_replaced_last": True,
            "passed": True,
        },
    )
    for signed_path, artifact_type in (
        (rolling_path, "rolling_capacity"),
        (failure_path, "failure_state_drill"),
        (raw_replay_path, "raw_replay_drill"),
        (archive_dry_run_path, "archive_dry_run"),
        (publication_path, "publication_dry_run"),
    ):
        top_writer.record_existing_file(signed_path, artifact_type=artifact_type)
        top_writer.record_existing_file(
            signed_path.with_suffix(".sha256"),
            artifact_type=f"{artifact_type}_sidecar",
            storage_format="sha256_control",
        )
    for journal in rolling_journals:
        top_writer.record_existing_file(
            journal,
            artifact_type="rolling_capacity_observation",
        )
        top_writer.record_existing_file(
            journal.with_suffix(".sha256"),
            artifact_type="rolling_capacity_observation_sidecar",
            storage_format="sha256_control",
        )
    top_writer.record_existing_file(
        partial_path,
        artifact_type="failure_state_drill_partial",
        storage_format="binary",
    )
    top_bundle = top_writer.finalize()
    campaign_interval_labels = (
        "campaign_parent_write_control",
        "campaign_plan_write",
        "campaign_preflight_write",
        "campaign_manifest_initial_write",
        "failure_state_drill_partial_write",
        "failure_state_drill_summary_write",
        "batch0001_pre_dispatch_rolling_capacity_journal_write",
        "batch0001_pre_dispatch_rolling_capacity_write",
        "batch0001_campaign_verified_state_write",
        "batch0001_pre_archive_rolling_capacity_journal_write",
        "batch0001_pre_archive_rolling_capacity_write",
        "batch0001_campaign_archived_state_write",
        "batch0001_persistence_envelope_write",
        "batch0001_campaign_envelope_state_write",
        "batch0001_post_archive_rolling_capacity_journal_write",
        "batch0001_post_archive_rolling_capacity_write",
        "campaign_rolling_capacity_complete_write",
        "raw_replay_drill_summary_write",
        "archive_dry_run_archive_payload_write",
        "archive_dry_run_archive_verified_manifest_write",
        "archive_dry_run_archive_archived_manifest_write",
        "archive_dry_run_summary_write",
        "publication_dry_run_payload_write",
        "publication_dry_run_trusted_manifest_write",
        "publication_dry_run_summary_write",
        "campaign_aggregate_per_run_write",
        "campaign_anytime_checkpoints_write",
        "campaign_primary_manifest_finalize",
    )
    campaign_attribution = Stage052PersistenceAttribution(
        run_label=run_label,
        component="benchmark",
        scope="pilot",
        subject_id="campaign",
        primary_manifest_relative_path=top_bundle.manifest_path.relative_to(
            campaign_dir
        ).as_posix(),
        primary_manifest_sha256=hashlib.sha256(top_bundle.manifest_path.read_bytes()).hexdigest(),
        solver_seconds=envelope.solver_seconds,
        shard_persistence_seconds=envelope.total_persistence_seconds,
        control_intervals=tuple(
            PersistenceInterval(label, 1000 + index, 1000 + index)
            for index, label in enumerate(campaign_interval_labels)
        ),
    )
    _write_signed_json(
        campaign_dir / "control" / f"{run_label}_persistence_attribution.json",
        campaign_attribution.to_dict(),
    )
    return campaign_dir, locator, volume


def test_campaign_selection_lock_binds_f02_backend_worker_native_and_provenance() -> None:
    metadata, review, identity = _selection_inputs()

    audit = validate_campaign_selection_lock(
        campaign_backend="native_cpu",
        campaign_exact_backend="cpu_batch",
        campaign_workers=2,
        campaign_native_profile="stage05.2-native-kernels-v1",
        prerequisite_metadata=metadata,
        prerequisite_review=review,
        prerequisite_identity=identity,
    )

    assert audit.passed is True
    assert audit.selection_lock["selected_workers"] == 2
    assert audit.selection_lock["native_kernel_config"] == metadata["native_kernel_config"]
    assert audit.selection_lock["accelerator_review_manifest_sha256"] == "b" * 64


def test_campaign_selection_lock_binds_promoted_cuda_backend() -> None:
    metadata, review, identity = _selection_inputs()
    metadata["execution_backend"] = "cuda"
    metadata["optimization_profile"] = "cuda"
    review["selected_backend"] = "cuda"
    review["selected_optimization_profile"] = "cuda"
    review["accelerator_decision"] = "ACCELERATOR_PROMOTED"

    audit = validate_campaign_selection_lock(
        campaign_backend="cuda",
        campaign_exact_backend="cpu_batch",
        campaign_workers=2,
        campaign_native_profile="stage05.2-native-kernels-v1",
        prerequisite_metadata=metadata,
        prerequisite_review=review,
        prerequisite_identity=identity,
    )

    assert audit.passed
    assert audit.selection_lock["selected_backend"] == "cuda"
    assert audit.selection_lock["accelerator_decision"] == "ACCELERATOR_PROMOTED"


@pytest.mark.parametrize("field", ("selected_workers", "native_configuration"))
def test_campaign_selection_lock_rejects_f02_review_drift(field: str) -> None:
    metadata, review, identity = _selection_inputs()
    review[field] = 4 if field == "selected_workers" else {"enabled": False}

    audit = validate_campaign_selection_lock(
        campaign_backend="native_cpu",
        campaign_exact_backend="cpu_batch",
        campaign_workers=2,
        campaign_native_profile="stage05.2-native-kernels-v1",
        prerequisite_metadata=metadata,
        prerequisite_review=review,
        prerequisite_identity=identity,
    )

    assert audit.passed is False


def test_campaign_selection_lock_rejects_incomplete_native_profile() -> None:
    metadata, review, identity = _selection_inputs()
    native_value = metadata["native_kernel_config"]
    assert isinstance(native_value, dict)
    native = dict(native_value)
    native["screening"] = False
    metadata["native_kernel_config"] = native
    review["native_configuration"] = native

    audit = validate_campaign_selection_lock(
        campaign_backend="native_cpu",
        campaign_exact_backend="cpu_batch",
        campaign_workers=2,
        campaign_native_profile="stage05.2-native-kernels-v1",
        prerequisite_metadata=metadata,
        prerequisite_review=review,
        prerequisite_identity=identity,
    )

    assert audit.passed is False
    assert "native configuration" in audit.detail


def test_compact_trace_index_has_a_hard_schema_and_size_boundary() -> None:
    valid = {
        "campaign_trace_schema_version": "stage05.2-campaign-trace-v1",
        "trace_storage_version": "stage03-trace-index-v2",
        "run_label": "stage05.2_benchmark_attempt01",
        "instance": "c101C5",
        "seed": 2014,
        "axes": {"wall_clock_30": {"result_summary": {}}},
        "route_dictionary_ref": "c101C5/2014/routes.parquet",
        "events_ref": "c101C5/2014/events.parquet",
        "screening_checks_ref": "c101C5/2014/checks.parquet",
        "screening_definitions_ref": "c101C5/2014/definitions.parquet",
        "screening_occurrences_ref": "c101C5/2014/occurrences.parquet",
        "diagnostic_ref": "c101C5/2014/diagnostic.parquet",
        "screening_schema_version": "screening_decisions_v3",
        "lane_dictionary": {},
        "operator_dictionary": {},
        "schema_fingerprints": dict(COMPACT_TRACE_SCHEMA_FINGERPRINTS),
        "event_identity": {"shard_ordinal": 0, "local_field": "event_id"},
    }
    validate_compact_trace_index(valid, byte_size=1024)

    with pytest.raises(ValueError, match="compact trace"):
        validate_compact_trace_index(valid, byte_size=16 * 1024 * 1024 + 1)
    with pytest.raises(ValueError, match="compact trace"):
        validate_compact_trace_index({"axes": {}}, byte_size=10)
    missing = dict(valid)
    missing["schema_fingerprints"] = {
        key: value
        for key, value in COMPACT_TRACE_SCHEMA_FINGERPRINTS.items()
        if key != "diagnostic"
    }
    with pytest.raises(ValueError, match="compact trace"):
        validate_compact_trace_index(missing, byte_size=1024)
    extra = dict(valid)
    extra["schema_fingerprints"] = {
        **COMPACT_TRACE_SCHEMA_FINGERPRINTS,
        "unregistered": "0" * 64,
    }
    with pytest.raises(ValueError, match="compact trace"):
        validate_compact_trace_index(extra, byte_size=1024)
    wrong = dict(valid)
    wrong["schema_fingerprints"] = {
        **COMPACT_TRACE_SCHEMA_FINGERPRINTS,
        "diagnostic": "0" * 64,
    }
    with pytest.raises(ValueError, match="compact trace"):
        validate_compact_trace_index(wrong, byte_size=1024)


def test_compact_trace_physical_size_is_checked_before_json_materialization(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "oversized_trace.json"
    trace_path.write_bytes(b"{" + b" " * (16 * 1024 * 1024) + b"}")
    descriptor = {
        "relative_path": trace_path.name,
        "artifact_type": "trace",
        "artifact_subtype": "compact_index_v1",
        "retention_class": "critical",
        "storage_format": "json_control",
        "compression": "none",
        "evidence_completeness": "complete",
        "checksum": "a" * 64,
        "byte_size": trace_path.stat().st_size,
        "row_count": None,
        "schema_fingerprint": "",
        "storage_policy_version": "artifact-storage-v2",
    }

    class FakeReader:
        manifest = {"artifacts": [descriptor]}
        result = SimpleNamespace(run_dir=tmp_path)

        def read_json(self, _relative_path: str) -> dict[str, object]:
            raise AssertionError("oversized trace must be rejected before read_json")

    with pytest.raises(ArtifactIntegrityError, match="16 MiB"):
        _read_bounded_compact_trace(FakeReader(), descriptor)  # type: ignore[arg-type]

    forged = dict(descriptor)
    forged["byte_size"] = 1
    with pytest.raises(ArtifactIntegrityError, match="descriptor"):
        _read_bounded_compact_trace(FakeReader(), forged)  # type: ignore[arg-type]


def test_bks_reference_rejects_gap_columns(tmp_path: Path) -> None:
    source = Path("experiments/baselines/schneider_best_known.csv")
    clean = validate_bks_reference(source)
    assert clean.passed is True

    tampered = tmp_path / "bks.csv"
    lines = source.read_text(encoding="utf-8").splitlines()
    lines[0] += ",distance_gap"
    lines[1:] = [f"{line},0" for line in lines[1:]]
    tampered.write_text("\n".join(lines) + "\n", encoding="utf-8")

    audit = validate_bks_reference(tampered)
    assert audit.passed is False
    assert "gap" in audit.detail


def _accepted_review(
    tmp_path: Path,
    *,
    scope: str = "formal",
    status: str = "READY_FOR_STAGE05_3",
    run_label: str = "stage05.2_benchmark_attempt02",
) -> Path:
    review_dir = tmp_path / "raw" / "review"
    generation = review_dir / "generations" / "pending"
    generation.mkdir(parents=True)
    targets = {
        "per_run_results": "per_run_results.csv",
        "family_summary": "family_summary.csv",
        "budget_summary": "budget_summary.csv",
        "anytime_summary": "anytime_summary.csv",
        "resource_summary": "resource_summary.csv",
        "persistence_summary": "persistence_summary.csv",
        "performance_gates": "performance_gates.csv",
        "gpu_decision": "gpu_decision.json",
        "failure_analysis": "failure_analysis.csv",
        "review_findings": "review_findings.csv",
        "review_report": "review_report.md",
    }
    publication_files: dict[str, dict[str, str]] = {}
    selection_metadata, _, _ = _selection_inputs()
    native = selection_metadata["native_kernel_config"]
    native_sha256 = hashlib.sha256(
        json.dumps(native, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    for key, name in targets.items():
        path = generation / name
        if key == "gpu_decision":
            path.write_text(
                json.dumps(
                    {
                        "schema_version": "stage05.2-gpu-decision-publication-v1",
                        "run_label": run_label,
                        "decision": "GPU_NOT_JUSTIFIED",
                        "selected_backend": "native_cpu",
                        "selected_exact_backend": "cpu_batch",
                        "selected_workers": 2,
                        "native_profile": "stage05.2-native-kernels-v1",
                        "native_config_sha256": native_sha256,
                        "accelerator_review_manifest_sha256": "d" * 64,
                        "campaign_prerequisite_review_sha256": "e" * 64,
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        else:
            path.write_text(f"{key}\n", encoding="utf-8")
        publication_files[key] = {
            "relative_path": path.relative_to(review_dir).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    generation_digest = hashlib.sha256()
    for key in sorted(publication_files):
        generation_digest.update(key.encode("utf-8") + b"\0")
        generation_digest.update((generation / targets[key]).read_bytes())
    final_generation = generation.with_name(generation_digest.hexdigest())
    generation.rename(final_generation)
    for key, item in publication_files.items():
        item["relative_path"] = (final_generation / targets[key]).relative_to(review_dir).as_posix()
    manifest = {
        "schema_version": "stage05.2-campaign-review-v1",
        "run_label": run_label,
        "component": "benchmark",
        "scope": scope,
        "status": status,
        "raw_manifest_sha256": "a" * 64,
        "raw_campaign_manifest_sha256": "b" * 64,
        "selected_backend": "native_cpu",
        "selected_exact_backend": "cpu_batch",
        "selected_workers": 2,
        "native_profile": "stage05.2-native-kernels-v1",
        "accelerator_decision": "GPU_NOT_JUSTIFIED",
        "campaign_prerequisite_review_sha256": "e" * 64,
        "native_kernel_config": native,
        "native_configuration": native,
        "selection_lock": {
            "selected_backend": "native_cpu",
            "selected_exact_backend": "cpu_batch",
            "selected_workers": 2,
            "native_profile": "stage05.2-native-kernels-v1",
            "native_config_sha256": native_sha256,
            "native_kernel_config": native,
            "accelerator_decision": "GPU_NOT_JUSTIFIED",
            "accelerator_review_manifest_sha256": "d" * 64,
        },
        "publication_files": publication_files,
        "files": {item["relative_path"]: item["sha256"] for item in publication_files.values()},
        "gates": {
            gate: {"passed": True, "detail": "passed"}
            for gate in (CAMPAIGN_PILOT_GATES if scope == "pilot" else CAMPAIGN_FORMAL_GATES)
        },
    }
    manifest_path = review_dir / "review_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def test_atomic_publisher_places_trusted_manifest_last(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "tools.publish_stage052_artifacts._verify_live_formal_chain",
        lambda **_kwargs: None,
    )
    review_manifest = _accepted_review(tmp_path)
    repository = tmp_path / "repository"

    def interrupt(stage: str) -> None:
        if stage == "before_trusted_manifest":
            raise RuntimeError("injected publication interruption")

    with pytest.raises(RuntimeError, match="interruption"):
        publish_stage052_artifacts(
            review_manifest=review_manifest,
            repository_root=repository,
            prerequisite_dir=tmp_path / "g01",
            checkpoint=interrupt,
        )

    trusted = (
        repository / "experiments/manifests/stage05.2_performance_benchmark_artifact_manifest.json"
    )
    assert not trusted.exists()
    assert list((repository / "experiments/summaries").glob("*.csv"))


def test_campaign_review_files_reject_swapped_publication_key_mapping(
    tmp_path: Path,
) -> None:
    review_path = _accepted_review(tmp_path)
    review = json.loads(review_path.read_text(encoding="utf-8"))
    publication = review["publication_files"]
    publication["per_run_results"], publication["review_report"] = (
        publication["review_report"],
        publication["per_run_results"],
    )

    with pytest.raises(ValueError, match="key/filename"):
        verify_stage052_review_files(review_path.parent.parent, review)


def test_publisher_rejects_ready_review_with_missing_mandatory_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    review_path = _accepted_review(tmp_path)
    review = json.loads(review_path.read_text(encoding="utf-8"))
    del review["gates"]["rolling_capacity_replay"]
    _write_json(review_path, review)
    monkeypatch.setattr(
        "tools.publish_stage052_artifacts._verify_live_formal_chain",
        lambda **_kwargs: None,
    )
    repository = tmp_path / "repository"

    with pytest.raises(ValueError, match="mandatory gate"):
        publish_stage052_artifacts(
            review_manifest=review_path,
            repository_root=repository,
            prerequisite_dir=tmp_path / "g01",
        )

    assert not repository.exists()


def test_atomic_publisher_interruption_after_sidecar_cannot_leave_ready_pointer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "tools.publish_stage052_artifacts._verify_live_formal_chain",
        lambda **_kwargs: None,
    )
    review_manifest = _accepted_review(tmp_path)
    repository = tmp_path / "repository"

    def interrupt(stage: str) -> None:
        if stage == "after_sidecar_before_trusted_manifest":
            raise RuntimeError("injected sidecar interruption")

    with pytest.raises(RuntimeError, match="sidecar interruption"):
        publish_stage052_artifacts(
            review_manifest=review_manifest,
            repository_root=repository,
            prerequisite_dir=tmp_path / "g01",
            checkpoint=interrupt,
        )

    trusted = (
        repository / "experiments/manifests/stage05.2_performance_benchmark_artifact_manifest.json"
    )
    assert not trusted.exists()


def test_publisher_rechecks_source_review_after_final_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "tools.publish_stage052_artifacts._verify_live_formal_chain",
        lambda **_kwargs: None,
    )
    review_manifest = _accepted_review(tmp_path)
    repository = tmp_path / "repository"

    def withdraw_review(stage: str) -> None:
        if stage == "after_sidecar_before_trusted_manifest":
            review = json.loads(review_manifest.read_text(encoding="utf-8"))
            review["status"] = "NOT_READY"
            _write_json(review_manifest, review)

    with pytest.raises(ValueError, match="status|pointer changed"):
        publish_stage052_artifacts(
            review_manifest=review_manifest,
            repository_root=repository,
            prerequisite_dir=tmp_path / "g01",
            checkpoint=withdraw_review,
        )

    trusted = (
        repository / "experiments/manifests/stage05.2_performance_benchmark_artifact_manifest.json"
    )
    assert not trusted.exists()


def test_publisher_rechecks_live_g01_chain_immediately_before_trusted_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def verify_live(**_kwargs: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("injected current G01 review withdrawal")

    monkeypatch.setattr(
        "tools.publish_stage052_artifacts._verify_live_formal_chain",
        verify_live,
    )
    review_manifest = _accepted_review(tmp_path)
    repository = tmp_path / "repository"

    with pytest.raises(ValueError, match="G01 review withdrawal"):
        publish_stage052_artifacts(
            review_manifest=review_manifest,
            repository_root=repository,
            prerequisite_dir=tmp_path / "g01",
        )

    assert calls == 2
    assert not (
        repository / "experiments/manifests/stage05.2_performance_benchmark_artifact_manifest.json"
    ).exists()


def test_atomic_publisher_commits_self_verifying_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "tools.publish_stage052_artifacts._verify_live_formal_chain",
        lambda **_kwargs: None,
    )
    review_manifest = _accepted_review(tmp_path)
    repository = tmp_path / "repository"

    outputs = publish_stage052_artifacts(
        review_manifest=review_manifest,
        repository_root=repository,
        prerequisite_dir=tmp_path / "g01",
    )

    trusted = json.loads(outputs["artifact_manifest"].read_text(encoding="utf-8"))
    assert trusted["status"] == "READY_FOR_STAGE05_3"
    registry = outputs["artifact_registry"]
    assert registry.is_file()
    assert trusted["artifact_registry"] == registry.relative_to(repository).as_posix()
    verify_canonical_registry_against_trusted(
        registry_path=outputs["artifact_registry"],
        trusted_manifest_path=outputs["artifact_manifest"],
    )
    verify_published_stage052_review(outputs["published_review_manifest"])
    for relative, expected in trusted["files"].items():
        observed = hashlib.sha256((repository / relative).read_bytes()).hexdigest()
        assert observed == expected


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("selected_backend", "metal"),
        ("campaign_prerequisite_review_sha256", "0" * 64),
    ),
)
def test_registry_verifier_rejects_trusted_provenance_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
) -> None:
    monkeypatch.setattr(
        "tools.publish_stage052_artifacts._verify_live_formal_chain",
        lambda **_kwargs: None,
    )
    review_manifest = _accepted_review(tmp_path)
    repository = tmp_path / "repository"
    outputs = publish_stage052_artifacts(
        review_manifest=review_manifest,
        repository_root=repository,
        prerequisite_dir=tmp_path / "g01",
    )
    trusted_path = outputs["artifact_manifest"]
    trusted = json.loads(trusted_path.read_text(encoding="utf-8"))
    trusted[field] = value
    _write_json(trusted_path, trusted)

    with pytest.raises(ValueError, match="provenance"):
        verify_canonical_registry_against_trusted(
            registry_path=outputs["artifact_registry"],
            trusted_manifest_path=trusted_path,
        )


@pytest.mark.parametrize("field", ("status", "relative_path", "sha256"))
def test_registry_verifier_rejects_semantic_row_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    monkeypatch.setattr(
        "tools.publish_stage052_artifacts._verify_live_formal_chain",
        lambda **_kwargs: None,
    )
    review_manifest = _accepted_review(tmp_path)
    repository = tmp_path / "repository"
    outputs = publish_stage052_artifacts(
        review_manifest=review_manifest,
        repository_root=repository,
        prerequisite_dir=tmp_path / "g01",
    )
    registry = outputs["artifact_registry"]
    with registry.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = handle.seek(0) or next(csv.reader(handle))
    assert rows
    rows[0][field] = {
        "status": "NOT_READY",
        "relative_path": "experiments/summaries/does-not-exist.csv",
        "sha256": "0" * 64,
    }[field]
    with registry.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    trusted_path = outputs["artifact_manifest"]
    trusted = json.loads(trusted_path.read_text(encoding="utf-8"))
    trusted["files"][trusted["artifact_registry"]] = hashlib.sha256(
        registry.read_bytes()
    ).hexdigest()
    _write_json(trusted_path, trusted)

    with pytest.raises(ValueError, match="registry"):
        verify_canonical_registry_against_trusted(
            registry_path=registry,
            trusted_manifest_path=trusted_path,
        )


def test_same_run_new_review_generation_preserves_old_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "tools.publish_stage052_artifacts._verify_live_formal_chain",
        lambda **_kwargs: None,
    )
    repository = tmp_path / "repository"
    first_review = _accepted_review(tmp_path / "first")
    first_outputs = publish_stage052_artifacts(
        review_manifest=first_review,
        repository_root=repository,
        prerequisite_dir=tmp_path / "g01",
    )
    first_trusted = json.loads(first_outputs["artifact_manifest"].read_text(encoding="utf-8"))
    first_generation_files = {
        relative: (repository / relative).read_bytes() for relative in first_trusted["files"]
    }
    second_review = _accepted_review(tmp_path / "second")
    second_payload = json.loads(second_review.read_text(encoding="utf-8"))
    second_payload["gates"]["campaign_identity"]["detail"] = "reviewer-only correction"
    _write_json(second_review, second_payload)

    second_outputs = publish_stage052_artifacts(
        review_manifest=second_review,
        repository_root=repository,
        prerequisite_dir=tmp_path / "g01",
    )
    second_trusted = json.loads(second_outputs["artifact_manifest"].read_text(encoding="utf-8"))

    assert second_trusted["generation_id"] != first_trusted["generation_id"]
    assert second_outputs["artifact_registry"] != first_outputs["artifact_registry"]
    assert all(
        (repository / relative).read_bytes() == payload
        for relative, payload in first_generation_files.items()
    )
    verify_canonical_registry_against_trusted(
        registry_path=second_outputs["artifact_registry"],
        trusted_manifest_path=second_outputs["artifact_manifest"],
    )


@pytest.mark.parametrize(
    "checkpoint_name",
    ("before_trusted_manifest", "after_sidecar_before_trusted_manifest"),
)
def test_interrupted_new_publication_preserves_prior_ready_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    checkpoint_name: str,
) -> None:
    monkeypatch.setattr(
        "tools.publish_stage052_artifacts._verify_live_formal_chain",
        lambda **_kwargs: None,
    )
    repository = tmp_path / "repository"
    first = _accepted_review(
        tmp_path / "first",
        run_label="stage05.2_benchmark_attempt02",
    )
    first_outputs = publish_stage052_artifacts(
        review_manifest=first,
        repository_root=repository,
        prerequisite_dir=tmp_path / "g01",
    )
    trusted_path = first_outputs["artifact_manifest"]
    trusted_bytes = trusted_path.read_bytes()
    trusted = json.loads(trusted_bytes)
    second = _accepted_review(
        tmp_path / "second",
        run_label="stage05.2_benchmark_attempt03",
    )

    def interrupt(stage: str) -> None:
        if stage == checkpoint_name:
            raise RuntimeError("injected second-generation interruption")

    with pytest.raises(RuntimeError, match="second-generation"):
        publish_stage052_artifacts(
            review_manifest=second,
            repository_root=repository,
            prerequisite_dir=tmp_path / "g01",
            checkpoint=interrupt,
        )

    assert trusted_path.read_bytes() == trusted_bytes
    assert all(
        hashlib.sha256((repository / relative).read_bytes()).hexdigest() == checksum
        for relative, checksum in trusted["files"].items()
    )
    with pytest.raises(ValueError, match="differs|does not match"):
        verify_canonical_registry_against_trusted(
            registry_path=(repository / "experiments/registries/stage05.2_artifact_registry.csv"),
            trusted_manifest_path=trusted_path,
        )


def test_publisher_rejects_source_toctou_before_any_tracked_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "tools.publish_stage052_artifacts._verify_live_formal_chain",
        lambda **_kwargs: None,
    )
    review = _accepted_review(tmp_path)
    repository = tmp_path / "repository"
    real_copy = publisher_module._copy_fsync
    injected = False

    def replace_before_copy(source: Path, destination: Path) -> None:
        nonlocal injected
        if not injected:
            injected = True
            source.write_bytes(b"UNREVIEWED\n")
        real_copy(source, destination)

    monkeypatch.setattr(publisher_module, "_copy_fsync", replace_before_copy)

    with pytest.raises(ValueError, match="changed during copy"):
        publish_stage052_artifacts(
            review_manifest=review,
            repository_root=repository,
            prerequisite_dir=tmp_path / "g01",
        )

    assert not list((repository / "experiments").rglob("*.csv"))
    assert not list((repository / "experiments").rglob("*.json"))


def test_publisher_live_chain_rejects_raw_tamper_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_dir = tmp_path / "results/stage05.2_benchmark_attempt02"
    review_pointer = raw_dir / "review" / "review_manifest.json"
    review_pointer.parent.mkdir(parents=True)
    raw_manifest = raw_dir / "manifest_stage05.2_benchmark_attempt02.json"
    raw_manifest.write_text("raw\n", encoding="utf-8")
    campaign_path = raw_dir / "campaign_manifest.json"
    campaign_path.write_text("campaign\n", encoding="utf-8")
    attribution = raw_dir / "control/stage05.2_benchmark_attempt02_persistence_attribution.json"
    _write_signed_json(attribution, {"status": "complete"})
    prerequisite = tmp_path / "results/stage05.2_benchmark_attempt01"
    g01_review = prerequisite / "review/review_manifest.json"
    _write_json(g01_review, {"status": "READY_FOR_STAGE052_FORMAL_BENCHMARK"})
    review = {
        "run_label": raw_dir.name,
        "raw_manifest_sha256": hashlib.sha256(raw_manifest.read_bytes()).hexdigest(),
        "raw_campaign_manifest_sha256": hashlib.sha256(campaign_path.read_bytes()).hexdigest(),
        "persistence_attribution_sha256": hashlib.sha256(attribution.read_bytes()).hexdigest(),
        "persistence_attribution_sidecar_sha256": hashlib.sha256(
            attribution.with_suffix(".sha256").read_bytes()
        ).hexdigest(),
        "selected_backend": "native_cpu",
        "selected_exact_backend": "cpu_batch",
        "selected_workers": 2,
        "campaign_prerequisite_review_sha256": hashlib.sha256(g01_review.read_bytes()).hexdigest(),
    }
    _write_json(review_pointer, review)
    campaign = SimpleNamespace(
        run_label=raw_dir.name,
        scope="formal",
        status="complete",
        prerequisite_review_sha256=hashlib.sha256(g01_review.read_bytes()).hexdigest(),
        selected_backend="native_cpu",
        selected_exact_backend="cpu_batch",
        selected_workers=2,
    )
    monkeypatch.setattr(
        "evrptw.artifacts.ArtifactReader",
        lambda _path: SimpleNamespace(result=SimpleNamespace(manifest_path=raw_manifest)),
    )
    monkeypatch.setattr(
        "evrptw.stage052_campaign.load_campaign_manifest",
        lambda _path: campaign,
    )
    monkeypatch.setattr(
        "evrptw.stage052_campaign_runner.load_benchmark_execution_lock",
        lambda *_args, **_kwargs: SimpleNamespace(
            selected_backend="native_cpu",
            selected_exact_backend="cpu_batch",
            selected_workers=2,
        ),
    )

    _verify_live_formal_chain(
        review_manifest=review_pointer,
        review=review,
        prerequisite_dir=prerequisite,
        repository_root=tmp_path,
    )
    raw_manifest.write_text("tampered\n", encoding="utf-8")

    with pytest.raises(ValueError, match="no longer binds"):
        _verify_live_formal_chain(
            review_manifest=review_pointer,
            review=review,
            prerequisite_dir=prerequisite,
            repository_root=tmp_path,
        )
    raw_manifest.write_text("raw\n", encoding="utf-8")
    _write_json(g01_review, {"status": "NOT_READY"})
    with pytest.raises(ValueError, match="current G01 review"):
        _verify_live_formal_chain(
            review_manifest=review_pointer,
            review=review,
            prerequisite_dir=prerequisite,
            repository_root=tmp_path,
        )


def test_pilot_publication_dry_run_leaves_no_tracked_ready_manifest(
    tmp_path: Path,
) -> None:
    review_manifest = _accepted_review(
        tmp_path,
        scope="pilot",
        status="READY_FOR_STAGE052_FORMAL_BENCHMARK",
    )
    workspace = tmp_path / "dry-run-workspace"

    report = dry_run_stage052_publication(
        review_manifest=review_manifest,
        workspace_root=workspace,
    )

    assert report["status"] == "passed"
    assert not list(workspace.glob("stage05.2-publication-dry-run-*"))
    assert not (workspace / "experiments").exists()


def test_campaign_reviewer_publishes_not_ready_for_missing_raw_campaign(
    tmp_path: Path,
) -> None:
    campaign = tmp_path / "stage05.2_benchmark_attempt02"
    campaign.mkdir()
    root = tmp_path / "archive"
    root.mkdir()
    volume = VolumeIdentity(device_uuid="test-volume", filesystem="apfs")
    locator = StorageRootLocator({"archive": StorageRoot("archive", root, volume)})

    outputs = review_stage052_campaign(
        campaign_dir=campaign,
        benchmark_dir=tmp_path / "benchmarks",
        bks_path=Path("experiments/baselines/schneider_best_known.csv"),
        scope="formal",
        prerequisite_dir=tmp_path / "pilot",
        locator=locator,
        volume_probe=lambda _: volume,
    )

    review = json.loads(outputs["review_manifest"].read_text(encoding="utf-8"))
    assert review["status"] == "NOT_READY"
    assert review["gates"]["campaign_replay"]["passed"] is False


def test_campaign_review_pointer_archives_prior_manifest_with_lineage(
    tmp_path: Path,
) -> None:
    campaign_dir = tmp_path / "stage05.2_benchmark_attempt01"
    raw_bundle = ArtifactBundleWriter(
        campaign_dir,
        ArtifactRunContext(
            "stage05.2",
            "benchmark",
            "stage05.2_benchmark_attempt01",
        ),
        ArtifactStorageConfig(
            storage_policy_version="artifact-storage-v2",
            screening_schema_version="screening_decisions_v3",
        ),
    ).finalize()
    raw_sha256 = hashlib.sha256(raw_bundle.manifest_path.read_bytes()).hexdigest()
    keys = (
        "per_run_results",
        "family_summary",
        "budget_summary",
        "anytime_summary",
        "resource_summary",
        "persistence_summary",
        "performance_gates",
        "gpu_decision",
        "failure_analysis",
        "review_findings",
        "review_report",
    )
    payloads = {key: f"{key}\n".encode() for key in keys}
    first_outputs = _publish_review(
        campaign_dir=campaign_dir,
        manifest={
            "schema_version": "stage05.2-campaign-review-v1",
            "run_label": campaign_dir.name,
            "scope": "pilot",
            "status": "READY_FOR_STAGE052_FORMAL_BENCHMARK",
            "raw_manifest_sha256": raw_sha256,
        },
        payloads=payloads,
    )
    first_bytes = first_outputs["review_manifest"].read_bytes()
    first_sha = hashlib.sha256(first_bytes).hexdigest()

    second_outputs = _publish_review(
        campaign_dir=campaign_dir,
        manifest={
            "schema_version": "stage05.2-campaign-review-v1",
            "run_label": campaign_dir.name,
            "scope": "pilot",
            "status": "NOT_READY",
            "raw_manifest_sha256": raw_sha256,
        },
        payloads=payloads,
    )

    archived = campaign_dir / "review/history" / first_sha / "review_manifest.json"
    current = json.loads(second_outputs["review_manifest"].read_text(encoding="utf-8"))
    assert archived.read_bytes() == first_bytes
    assert current["previous_review_manifest_sha256"] == first_sha
    assert current["review_history"] == [first_sha]
    verify_stage052_review_files(
        campaign_dir,
        json.loads(archived.read_text(encoding="utf-8")),
    )


def test_campaign_review_rejects_corrupt_prior_files_and_history(
    tmp_path: Path,
) -> None:
    campaign_dir = tmp_path / "stage05.2_benchmark_attempt01"
    raw_bundle = ArtifactBundleWriter(
        campaign_dir,
        ArtifactRunContext("stage05.2", "benchmark", campaign_dir.name),
        ArtifactStorageConfig(
            storage_policy_version="artifact-storage-v2",
            screening_schema_version="screening_decisions_v3",
        ),
    ).finalize()
    raw_sha256 = hashlib.sha256(raw_bundle.manifest_path.read_bytes()).hexdigest()
    keys = (
        "per_run_results",
        "family_summary",
        "budget_summary",
        "anytime_summary",
        "resource_summary",
        "persistence_summary",
        "performance_gates",
        "gpu_decision",
        "failure_analysis",
        "review_findings",
        "review_report",
    )
    payloads = {key: f"{key}\n".encode() for key in keys}
    manifest = {
        "schema_version": "stage05.2-campaign-review-v1",
        "run_label": campaign_dir.name,
        "scope": "pilot",
        "status": "NOT_READY",
        "raw_manifest_sha256": raw_sha256,
    }
    outputs = _publish_review(
        campaign_dir=campaign_dir,
        manifest=manifest,
        payloads=payloads,
    )
    original_manifest = outputs["review_manifest"].read_bytes()
    current = json.loads(outputs["review_manifest"].read_text(encoding="utf-8"))
    next_payloads = {**payloads, "review_report": b"second review\n"}
    first_file = campaign_dir / "review" / next(iter(current["files"]))
    original = first_file.read_bytes()
    first_file.write_text("tampered\n", encoding="utf-8")

    with pytest.raises(ArtifactIntegrityError, match="checksum"):
        _publish_review(campaign_dir=campaign_dir, manifest=manifest, payloads=next_payloads)

    first_file.write_bytes(original)
    current["review_history"] = "invalid"
    outputs["review_manifest"].write_text(
        json.dumps(current, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ArtifactIntegrityError, match="history"):
        _publish_review(campaign_dir=campaign_dir, manifest=manifest, payloads=next_payloads)

    outputs["review_manifest"].write_bytes(original_manifest)
    second = _publish_review(
        campaign_dir=campaign_dir,
        manifest=manifest,
        payloads=next_payloads,
    )
    second_payload = json.loads(second["review_manifest"].read_text(encoding="utf-8"))
    prior_sha256 = second_payload["review_history"][0]
    (campaign_dir / "review" / "history" / prior_sha256 / "review_manifest.json").unlink()
    with pytest.raises(ArtifactIntegrityError, match="history manifest is missing"):
        _publish_review(campaign_dir=campaign_dir, manifest=manifest, payloads=payloads)


def test_noncanonical_single_batch_pilot_cannot_receive_ready_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign, locator, volume = _build_complete_pilot_campaign(tmp_path)
    prerequisite = tmp_path / "stage05.2_accelerator_pilot_attempt02"
    prerequisite_reader = ArtifactReader(prerequisite)
    prerequisite_review = prerequisite / "review/review_manifest.json"
    prerequisite_metadata = prerequisite_reader.read_json(
        next(
            str(item["relative_path"])
            for item in prerequisite_reader.manifest["artifacts"]
            if item["artifact_type"] == "manifest_metadata"
        )
    )
    synthetic_identity = Stage052PrerequisiteIdentity(
        run_label=prerequisite.name,
        component="accelerator_pilot",
        status="READY_FOR_STAGE052_BENCHMARK",
        repository_revision=str(prerequisite_metadata["repository_revision"]),
        configuration_sha256=str(prerequisite_metadata["configuration_sha256"]),
        raw_manifest_sha256=hashlib.sha256(
            prerequisite_reader.result.manifest_path.read_bytes()
        ).hexdigest(),
        review_manifest_sha256=hashlib.sha256(prerequisite_review.read_bytes()).hexdigest(),
        scope="performance",
    )
    monkeypatch.setattr(
        "evrptw.experiments.stage052_campaign_review.verify_stage052_evidence_input",
        lambda _raw_dir, _requirement: synthetic_identity,
    )
    progress_path = tmp_path / "review-progress.jsonl"
    monkeypatch.setenv("STAGE052_REVIEW_PROGRESS_LOG", str(progress_path))

    outputs = review_stage052_campaign(
        campaign_dir=campaign,
        benchmark_dir=Path("data/schneider"),
        bks_path=Path("experiments/baselines/schneider_best_known.csv"),
        scope="pilot",
        prerequisite_dir=prerequisite,
        locator=locator,
        volume_probe=lambda _: volume,
    )

    review = json.loads(outputs["review_manifest"].read_text(encoding="utf-8"))
    assert review["status"] == "NOT_READY"
    assert review["selected_optimization_profile"] == "native"
    assert review["gates"]["campaign_planning_replay"]["passed"] is False
    progress = [
        json.loads(line)
        for line in progress_path.read_text(encoding="utf-8").splitlines()
    ]
    completed = [
        row for row in progress if row["event"] == "campaign_shard_replay_complete"
    ]
    assert len(completed) == 36
    assert len({row["child_pid"] for row in completed}) == 36
    assert all(row["logical_pass_count"] == 1 for row in completed)
    assert all(row["scratch_cleaned"] is True for row in completed)
    assert all(row["child_peak_rss_bytes"] > 0 for row in completed)
    parent_rss = [int(row["parent_rss_bytes"]) for row in completed]
    assert max(parent_rss) - min(parent_rss) < 256 * 1024 * 1024
