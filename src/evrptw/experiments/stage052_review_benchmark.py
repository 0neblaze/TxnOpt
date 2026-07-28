"""Benchmark the batch-scoped native reviewer scheduler on sealed Pilot shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import psutil  # type: ignore[import-untyped]

from evrptw.artifacts import ArtifactReader, atomic_write_signed_json
from evrptw.experiments.stage052_campaign_review import (
    _replay_batch_in_fresh_processes,
    _ShardReplayProcessResult,
    _ShardReplayTask,
)
from evrptw.stage052_campaign import load_batch_manifest
from evrptw.stage052_resources import (
    ReviewMemoryContract,
    derive_review_memory_contract,
)

_REVIEW_WORKERS = (1, 2, 4)
_OLD_REFERENCE_EVENTS_PER_SECOND = 41_966.0
_FINAL_REVIEW_MULTIPLIER = 6.0


@dataclass(frozen=True, slots=True)
class ReviewCandidateResult:
    workers: int
    elapsed_seconds: float
    event_count: int
    events_per_second: float
    parent_baseline_rss_bytes: int
    per_child_p99_rss_bytes: int
    semantic_digest: str
    memory_contract: ReviewMemoryContract

    def to_dict(self) -> dict[str, object]:
        return {
            "workers": self.workers,
            "elapsed_seconds": self.elapsed_seconds,
            "event_count": self.event_count,
            "events_per_second": self.events_per_second,
            "parent_baseline_rss_bytes": self.parent_baseline_rss_bytes,
            "per_child_p99_rss_bytes": self.per_child_p99_rss_bytes,
            "semantic_digest": self.semantic_digest,
            "memory_contract": self.memory_contract.to_dict(),
        }


def _strict_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"review benchmark {field_name} is not an integer")
    return value


def _batch_tasks(
    *,
    batch_dir: Path,
    benchmark_dir: Path,
) -> tuple[_ShardReplayTask, ...]:
    reader = ArtifactReader(batch_dir)
    batch = load_batch_manifest(batch_dir / "batch_manifest.json")
    artifacts = reader.manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise RuntimeError("review benchmark batch lacks artifact references")
    tasks: list[_ShardReplayTask] = []
    for reference in artifacts:
        if (
            not isinstance(reference, Mapping)
            or reference.get("artifact_type") != "shard_manifest"
        ):
            continue
        relative = reference.get("relative_path")
        if not isinstance(relative, str):
            raise RuntimeError("review benchmark shard manifest path is invalid")
        shard = reader.read_json(relative)
        checksum = reference.get("checksum")
        shard_ids = [
            shard_id
            for shard_id, digest in (batch.shard_manifest_sha256_by_id or {}).items()
            if digest == checksum
        ]
        shard_id = shard_ids[0] if len(shard_ids) == 1 else None
        instance = shard.get("instance")
        if (
            not isinstance(shard_id, str)
            or not isinstance(instance, str)
            or shard_id not in batch.shard_ids
        ):
            raise RuntimeError("review benchmark shard identity is invalid")
        tasks.append(
            _ShardReplayTask(
                batch_dir=batch_dir,
                shard_relative_path=relative,
                benchmark_dir=benchmark_dir,
                scope="pilot",
                run_label=batch.run_label,
                batch_id=batch.batch_id,
                shard_id=shard_id,
                expected_instance=instance,
                expected_seed=_strict_int(shard.get("seed"), "seed"),
            )
        )
    ordered = tuple(sorted(tasks, key=lambda item: item.ordinal))
    if tuple(task.shard_id for task in ordered) != batch.shard_ids:
        raise RuntimeError("review benchmark tasks do not cover the exact batch shards")
    return ordered


def _p99(values: Sequence[int]) -> int:
    if not values:
        raise ValueError("review benchmark RSS sample set is empty")
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * 0.99) - 1)
    return ordered[index]


def _semantic_digest(results: Sequence[_ShardReplayProcessResult]) -> str:
    payload = [
        {
            "shard_id": result.shard_id,
            "instance": result.instance,
            "seed": result.seed,
            "replay_rows": list(result.replay_rows),
            "checkpoints": list(result.checkpoints),
            "logical_events": result.logical_events,
            "screening_definition_rows": result.screening_definition_rows,
        }
        for result in results
    ]
    return hashlib.sha256(
        json.dumps(
            payload,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def benchmark_review_candidate(
    *,
    campaign_dir: Path,
    benchmark_dir: Path,
    workers: int,
    available_memory_bytes: int,
) -> ReviewCandidateResult:
    """Replay all 36 shards using one frozen scheduler candidate."""

    if workers not in _REVIEW_WORKERS:
        raise ValueError("review workers must be 1, 2, or 4")
    parent = psutil.Process()
    parent_baseline = int(parent.memory_info().rss)
    results: list[_ShardReplayProcessResult] = []
    started = time.perf_counter()
    for batch_dir in sorted(campaign_dir.glob("batch[0-9][0-9][0-9][0-9]")):
        if not batch_dir.is_dir():
            continue
        results.extend(
            _replay_batch_in_fresh_processes(
                _batch_tasks(batch_dir=batch_dir, benchmark_dir=benchmark_dir),
                review_workers=workers,
            )
        )
    elapsed = time.perf_counter() - started
    if len(results) != 36:
        raise RuntimeError(f"review benchmark expected 36 shards, got {len(results)}")
    event_count = sum(result.logical_events for result in results)
    p99 = _p99([result.child_peak_rss_bytes for result in results])
    contract = derive_review_memory_contract(
        parent_baseline_rss_bytes=parent_baseline,
        per_child_p99_rss_bytes=p99,
        review_workers=workers,
        available_memory_bytes=available_memory_bytes,
    )
    return ReviewCandidateResult(
        workers=workers,
        elapsed_seconds=elapsed,
        event_count=event_count,
        events_per_second=event_count / elapsed,
        parent_baseline_rss_bytes=parent_baseline,
        per_child_p99_rss_bytes=p99,
        semantic_digest=_semantic_digest(results),
        memory_contract=contract,
    )


def select_review_candidate(
    candidates: Sequence[ReviewCandidateResult],
) -> ReviewCandidateResult:
    """Choose the fastest candidate, preferring fewer workers within five percent."""

    by_workers = {candidate.workers: candidate for candidate in candidates}
    if set(by_workers) != set(_REVIEW_WORKERS) or len(candidates) != len(
        _REVIEW_WORKERS
    ):
        raise ValueError("review calibration requires exactly one 1/2/4 result")
    digests = {candidate.semantic_digest for candidate in candidates}
    counts = {candidate.event_count for candidate in candidates}
    if len(digests) != 1 or len(counts) != 1:
        raise RuntimeError("review scheduler candidates are not semantically identical")
    fastest = max(candidate.events_per_second for candidate in candidates)
    tied = [
        candidate
        for candidate in candidates
        if candidate.events_per_second >= fastest * 0.95
    ]
    return min(tied, key=lambda candidate: candidate.workers)


def run_review_benchmark(
    *,
    campaign_dir: Path,
    benchmark_dir: Path,
    report_path: Path,
    contract_path: Path,
) -> dict[str, object]:
    """Run 1/2/4 candidates and publish the selected memory contract."""

    available_memory = int(psutil.virtual_memory().available)
    candidates = tuple(
        benchmark_review_candidate(
            campaign_dir=campaign_dir,
            benchmark_dir=benchmark_dir,
            workers=workers,
            available_memory_bytes=available_memory,
        )
        for workers in _REVIEW_WORKERS
    )
    selected = select_review_candidate(candidates)
    final_target = _OLD_REFERENCE_EVENTS_PER_SECOND * _FINAL_REVIEW_MULTIPLIER
    payload = {
        "schema_version": "stage05.2-review-calibration-v1",
        "campaign_role": "accepted_pilot_read_only_calibration",
        "campaign_geometry_contribution": 0,
        "available_memory_bytes": available_memory,
        "candidates": [candidate.to_dict() for candidate in candidates],
        "selected_review_workers": selected.workers,
        "selected_memory_contract": selected.memory_contract.to_dict(),
        "old_serial_reference_events_per_second": (
            _OLD_REFERENCE_EVENTS_PER_SECOND
        ),
        "final_review_target_events_per_second": final_target,
        "final_review_gate_passed": selected.events_per_second >= final_target,
        "semantic_identity_gate_passed": True,
    }
    atomic_write_signed_json(report_path, payload)
    atomic_write_signed_json(contract_path, selected.memory_contract.to_dict())
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark Stage 5.2 batch-scoped native reviewer"
    )
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    arguments = parser.parse_args()
    payload = run_review_benchmark(
        campaign_dir=arguments.campaign_dir.resolve(),
        benchmark_dir=arguments.benchmark_dir.resolve(),
        report_path=arguments.report.resolve(),
        contract_path=arguments.contract.resolve(),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["final_review_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
