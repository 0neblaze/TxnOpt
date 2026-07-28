"""Differential and performance gates for Stage 5.2 native Arrow replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from evrptw.artifacts import ArtifactReader, atomic_write_signed_json
from evrptw.stage052_replay import (
    ReplaySummary,
    replay_verified_shard,
    verified_artifact_shard_bundle,
)

_OLD_REFERENCE_EVENTS_PER_SECOND = 41_966.0
_NATIVE_SINGLE_CHILD_MULTIPLIER = 4.0


@dataclass(frozen=True, slots=True)
class ReplayShardSpec:
    batch_dir: Path
    shard_id: str
    instance: str
    seed: int
    event_relative_path: str
    axis_budgets: Mapping[str, int]


def _axis_budget(axis: str) -> int:
    prefix, separator, raw_budget = axis.rpartition("_")
    if not prefix or separator != "_" or not raw_budget.isdigit():
        raise RuntimeError(f"replay benchmark axis has no wall-clock budget: {axis}")
    budget = int(raw_budget)
    if budget <= 0:
        raise RuntimeError(f"replay benchmark axis budget is invalid: {axis}")
    return budget


def discover_replay_shards(campaign_dir: Path) -> tuple[ReplayShardSpec, ...]:
    """Discover only complete, manifest-authenticated event shards."""

    specs: list[ReplayShardSpec] = []
    for batch_dir in sorted(campaign_dir.glob("batch[0-9][0-9][0-9][0-9]")):
        if not batch_dir.is_dir():
            continue
        reader = ArtifactReader(batch_dir)
        artifacts = reader.manifest.get("artifacts")
        if not isinstance(artifacts, list):
            raise RuntimeError(f"batch manifest lacks artifacts: {batch_dir}")
        for reference in artifacts:
            if (
                not isinstance(reference, Mapping)
                or reference.get("artifact_type") != "events"
                or reference.get("artifact_subtype") != "critical"
            ):
                continue
            relative = reference.get("relative_path")
            if not isinstance(relative, str):
                raise RuntimeError("event artifact relative path is invalid")
            parts = Path(relative).parts
            if len(parts) < 3:
                raise RuntimeError(f"event artifact path is non-canonical: {relative}")
            instance = parts[0]
            try:
                seed = int(parts[1])
            except ValueError as error:
                raise RuntimeError(f"event artifact seed is invalid: {relative}") from error
            raw_relative = str(
                Path(relative)
                .with_name(Path(relative).name.replace("_events_", "_raw_", 1))
                .with_suffix(".json")
            )
            raw = reader.read_json(raw_relative)
            axes = raw.get("axes")
            if not isinstance(axes, Mapping) or not axes:
                raise RuntimeError(f"raw shard axes are missing: {relative}")
            axis_budgets = {str(axis): _axis_budget(str(axis)) for axis in axes}
            specs.append(
                ReplayShardSpec(
                    batch_dir=batch_dir,
                    shard_id=f"{instance}/{seed}",
                    instance=instance,
                    seed=seed,
                    event_relative_path=relative,
                    axis_budgets=axis_budgets,
                )
            )
    return tuple(
        sorted(specs, key=lambda item: (item.instance.casefold(), item.seed))
    )


def _summary_payload(summary: ReplaySummary) -> dict[str, object]:
    payload = asdict(summary)
    payload.pop("backend")
    return payload


def benchmark_replay_specs(
    specs: Sequence[ReplayShardSpec],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Run both adapters and require exact per-field equality."""

    if not specs:
        raise ValueError("replay differential corpus must not be empty")
    rows: list[dict[str, object]] = []
    total_events = 0
    reference_seconds = 0.0
    native_seconds = 0.0
    aggregate_digest = hashlib.sha256()
    for spec in specs:
        reader = ArtifactReader(spec.batch_dir)
        bundle = verified_artifact_shard_bundle(
            reader=reader,
            shard_id=spec.shard_id,
            event_relative_path=spec.event_relative_path,
            axis_budgets=spec.axis_budgets,
        )
        started = time.perf_counter()
        reference = replay_verified_shard(bundle, backend="python_reference")
        reference_elapsed = time.perf_counter() - started
        started = time.perf_counter()
        native = replay_verified_shard(bundle, backend="native_arrow")
        native_elapsed = time.perf_counter() - started
        if native != replace(reference, backend="native_arrow"):
            raise RuntimeError(f"replay adapter mismatch: {spec.shard_id}")
        summary_payload = _summary_payload(reference)
        aggregate_digest.update(
            json.dumps(
                summary_payload,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        )
        total_events += reference.event_count
        reference_seconds += reference_elapsed
        native_seconds += native_elapsed
        rows.append(
            {
                "shard_id": spec.shard_id,
                "batch_dir": str(spec.batch_dir),
                "axes": dict(spec.axis_budgets),
                "event_count": reference.event_count,
                "reference_seconds": reference_elapsed,
                "native_seconds": native_elapsed,
                "reference_events_per_second": (
                    reference.event_count / reference_elapsed
                ),
                "native_events_per_second": reference.event_count / native_elapsed,
                "summary_sha256": hashlib.sha256(
                    json.dumps(
                        summary_payload,
                        allow_nan=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    ).encode("utf-8")
                ).hexdigest(),
                "field_identical": True,
            }
        )
    native_events_per_second = total_events / native_seconds
    reference_events_per_second = total_events / reference_seconds
    gate_target = (
        _OLD_REFERENCE_EVENTS_PER_SECOND * _NATIVE_SINGLE_CHILD_MULTIPLIER
    )
    aggregate = {
        "shard_count": len(specs),
        "event_count": total_events,
        "reference_seconds": reference_seconds,
        "native_seconds": native_seconds,
        "reference_events_per_second": reference_events_per_second,
        "native_events_per_second": native_events_per_second,
        "native_speedup_over_observed_reference": (
            native_events_per_second / reference_events_per_second
        ),
        "old_reference_events_per_second": _OLD_REFERENCE_EVENTS_PER_SECOND,
        "native_single_child_target_events_per_second": gate_target,
        "native_single_child_gate_passed": native_events_per_second >= gate_target,
        "field_identity_gate_passed": True,
        "aggregate_summary_sha256": aggregate_digest.hexdigest(),
    }
    return rows, aggregate


def representative_attempt73_specs(
    attempt73_dir: Path,
    *,
    attempt72_dir: Path,
) -> tuple[ReplayShardSpec, ...]:
    """Select the available Formal C corpus plus accepted Pilot R/RC controls."""

    formal = discover_replay_shards(attempt73_dir)
    pilot = discover_replay_shards(attempt72_dir)

    def one(
        corpus: Iterable[ReplayShardSpec],
        instance: str,
        seed: int,
    ) -> ReplayShardSpec:
        matches = [
            item
            for item in corpus
            if item.instance == instance and item.seed == seed
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"representative replay corpus lacks {instance}/{seed}"
            )
        return matches[0]

    # Attempt73 was interrupted before the R and RC families.  Its sealed C
    # Formal shard covers 30/60/300-second axes; accepted Attempt72 supplies the
    # R/RC 30-second family controls without importing either corpus into a new
    # campaign.
    return (
        one(formal, "c101C5", 2014),
        one(formal, "c101_21", 2014),
        one(pilot, "r101_21", 2014),
        one(pilot, "rc101_21", 2014),
    )


def run_replay_benchmark(
    *,
    attempt72_dir: Path,
    attempt73_dir: Path,
    output_path: Path,
) -> dict[str, object]:
    """Run the complete Attempt72 differential plus the mixed read-only corpus."""

    attempt72_specs = discover_replay_shards(attempt72_dir)
    if len(attempt72_specs) != 36:
        raise RuntimeError(
            f"Attempt72 differential corpus must contain 36 shards, got {len(attempt72_specs)}"
        )
    attempt72_rows, attempt72_aggregate = benchmark_replay_specs(attempt72_specs)
    atomic_write_signed_json(
        output_path,
        {
            "schema_version": "stage05.2-replay-differential-v1",
            "status": "attempt72_complete_representative_pending",
            "corpus_policy": {
                "attempt72": "accepted_pilot_full_36_shards",
                "attempt73": "immutable_interrupted_read_only",
                "campaign_geometry_contribution": 0,
                "readiness_count_contribution": 0,
            },
            "attempt72": {
                "rows": attempt72_rows,
                "aggregate": attempt72_aggregate,
            },
            "passed": False,
        },
    )
    representative = representative_attempt73_specs(
        attempt73_dir,
        attempt72_dir=attempt72_dir,
    )
    representative_rows, representative_aggregate = benchmark_replay_specs(
        representative
    )
    payload = {
        "schema_version": "stage05.2-replay-differential-v1",
        "status": "complete",
        "corpus_policy": {
            "attempt72": "accepted_pilot_full_36_shards",
            "attempt73": "immutable_interrupted_read_only",
            "campaign_geometry_contribution": 0,
            "readiness_count_contribution": 0,
        },
        "attempt72": {
            "rows": attempt72_rows,
            "aggregate": attempt72_aggregate,
        },
        "representative": {
            "rows": representative_rows,
            "aggregate": representative_aggregate,
            "coverage_note": (
                "Attempt73 contains sealed C-family shards only; accepted Attempt72 "
                "provides the R/RC 30-second controls."
            ),
        },
        "passed": bool(
            attempt72_aggregate["field_identity_gate_passed"]
            and attempt72_aggregate["native_single_child_gate_passed"]
            and representative_aggregate["field_identity_gate_passed"]
        ),
    }
    atomic_write_signed_json(output_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark Stage 5.2 Python/native Arrow replay"
    )
    parser.add_argument("--attempt72-dir", type=Path, required=True)
    parser.add_argument("--attempt73-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    payload = run_replay_benchmark(
        attempt72_dir=arguments.attempt72_dir.resolve(),
        attempt73_dir=arguments.attempt73_dir.resolve(),
        output_path=arguments.output.resolve(),
    )
    attempt72 = payload.get("attempt72")
    if not isinstance(attempt72, Mapping):
        raise RuntimeError("replay benchmark report lacks Attempt72 results")
    print(json.dumps(attempt72.get("aggregate"), indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
