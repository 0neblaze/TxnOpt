from __future__ import annotations

from dataclasses import replace

import pyarrow as pa
import pytest

from evrptw.stage052_replay import (
    ReplayIntegrityError,
    VerifiedShardBundle,
    replay_verified_shard,
)

_SCHEMA = pa.schema(
    [
        pa.field("event_id", pa.int64(), nullable=False),
        pa.field("benchmark_axis", pa.string(), nullable=False),
        pa.field("lane", pa.string(), nullable=False),
        pa.field("event_type", pa.string(), nullable=False),
        pa.field("timestamp_seconds", pa.float64()),
        pa.field("started_at", pa.float64()),
        pa.field("completed_at", pa.float64()),
        pa.field("iteration", pa.int64()),
        pa.field("evaluation_id", pa.int64()),
        pa.field("cache_key_digest", pa.string()),
        pa.field("operation", pa.string()),
        pa.field("lookup_result", pa.string()),
        pa.field("status", pa.string()),
        pa.field("reason", pa.string()),
        pa.field("route_key", pa.string()),
        pa.field("decision_id", pa.int64()),
        pa.field("kind", pa.string()),
        pa.field("operator", pa.string()),
        pa.field("exact_started", pa.bool_()),
        pa.field("exact_completed", pa.bool_()),
        pa.field("feasible", pa.bool_()),
        pa.field("accepted", pa.bool_()),
        pa.field("global_best", pa.bool_()),
        pa.field("candidate_vehicle_delta", pa.int64()),
        pa.field("native_fallback", pa.bool_()),
        pa.field("failure_reason", pa.string()),
    ]
)


def _batch(rows: list[dict[str, object]]) -> pa.RecordBatch:
    return pa.RecordBatch.from_pylist(rows, schema=_SCHEMA)


def _valid_rows() -> list[dict[str, object]]:
    axis = "wall_clock_30"
    lane = f"{axis}:legacy"
    return [
        {
            "event_id": 1,
            "benchmark_axis": axis,
            "lane": lane,
            "event_type": "cache_event",
            "iteration": 1,
            "cache_key_digest": "route-a",
            "operation": "lookup_result",
            "lookup_result": "miss",
        },
        {
            "event_id": 2,
            "benchmark_axis": axis,
            "lane": lane,
            "event_type": "route_evaluation",
            "iteration": 1,
            "evaluation_id": 1,
            "cache_key_digest": "route-a",
            "started_at": 1.0,
            "completed_at": 2.0,
            "exact_started": True,
            "exact_completed": True,
            "feasible": True,
        },
        {
            "event_id": 3,
            "benchmark_axis": axis,
            "lane": lane,
            "event_type": "cache_event",
            "iteration": 1,
            "cache_key_digest": "route-a",
            "operation": "store",
        },
        {
            "event_id": 4,
            "benchmark_axis": axis,
            "lane": lane,
            "event_type": "candidate_state",
            "iteration": 1,
            "accepted": True,
            "global_best": True,
            "candidate_vehicle_delta": 0,
        },
        {
            "event_id": 5,
            "benchmark_axis": axis,
            "lane": lane,
            "event_type": "deadline_boundary",
            "timestamp_seconds": 30.0,
        },
    ]


def _bundle(rows: list[dict[str, object]]) -> VerifiedShardBundle:
    return VerifiedShardBundle(
        shard_id="shard0001",
        axis_budgets={"wall_clock_30": 30},
        event_batches=(_batch(rows[:2]), _batch(rows[2:])),
        source_schema_version="screening_decisions_v3",
    )


def test_python_reference_and_native_arrow_replay_are_field_identical() -> None:
    bundle = _bundle(_valid_rows())

    reference = replay_verified_shard(bundle, backend="python_reference")
    native = replay_verified_shard(bundle, backend="native_arrow")

    assert native == replace(reference, backend="native_arrow")
    assert native.event_count == 5
    assert native.exact_started == 1
    assert native.exact_completed == 1
    assert native.accepted_candidates == 1
    assert native.global_bests == 1
    assert native.native_fallback_count == 0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda rows: [rows[0], {**rows[1], "event_id": 1}, *rows[2:]],
            "strictly increasing",
        ),
        (
            lambda rows: [
                rows[0],
                {**rows[2], "event_id": 2},
                {**rows[1], "event_id": 3},
                *rows[3:],
            ],
            "cache store precedes exact completion",
        ),
        (
            lambda rows: [*rows, {**rows[3], "event_id": 6}],
            "candidate accepted after deadline",
        ),
    ],
)
def test_replay_adapters_fail_at_the_same_corruption(
    mutation: object,
    message: str,
) -> None:
    mutate = mutation
    assert callable(mutate)
    bundle = _bundle(mutate(_valid_rows()))  # type: ignore[operator]

    failures: list[str] = []
    for backend in ("python_reference", "native_arrow"):
        with pytest.raises(ReplayIntegrityError, match=message) as error:
            replay_verified_shard(bundle, backend=backend)
        failures.append(str(error.value))

    assert failures[0] == failures[1]


def test_native_arrow_backend_never_falls_back_to_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle(_valid_rows())
    monkeypatch.setattr(
        "evrptw.stage052_replay._native_replay",
        lambda _bundle: (_ for _ in ()).throw(RuntimeError("native crash")),
    )

    with pytest.raises(RuntimeError, match="native crash"):
        replay_verified_shard(bundle, backend="native_arrow")
