from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

from evrptw.artifacts import atomic_write_signed_json
from evrptw.experiments import stage052_calibration
from evrptw.experiments.stage052_calibration import (
    MeasuredProducerCandidate,
    ProducerMemoryFloor,
    load_attempt73_memory_floor,
    run_stage052_resource_calibration,
)
from evrptw.stage052_resources import (
    ParquetBenchmark,
    ProducerBenchmark,
    load_producer_resource_contract,
)


def test_calibration_runs_every_candidate_and_seals_selected_contract(
    tmp_path: Path,
    monkeypatch,
) -> None:
    observed_workers: list[int] = []
    observed_parquet: list[tuple[int, int]] = []

    def producer_runner(**kwargs) -> MeasuredProducerCandidate:
        workers = int(kwargs["workers"])
        observed_workers.append(workers)
        throughput = {4: 100.0, 5: 118.0, 6: 122.0}[workers]
        aggregate_gib = {4: 8, 5: 9, 6: 10}[workers]
        return MeasuredProducerCandidate(
            benchmark=ProducerBenchmark(
                workers=workers,
                throughput=throughput,
                aggregate_peak_rss_bytes=aggregate_gib * 1024**3,
                semantic_digest="a" * 64,
                swap_peak_bytes=0,
                fallback_count=0,
                resource_limit_exceeded=False,
            ),
            per_worker_peak_rss_bytes=3 * 1024**3,
        )

    def parquet_runner(**kwargs) -> ParquetBenchmark:
        row_group_size = int(kwargs["row_group_size"])
        queue_depth = int(kwargs["queue_depth"])
        observed_parquet.append((row_group_size, queue_depth))
        elapsed = {
            (65_536, 1): 100.0,
            (65_536, 2): 95.0,
            (262_144, 1): 92.0,
            (262_144, 2): 91.0,
        }[(row_group_size, queue_depth)]
        return ParquetBenchmark(
            row_group_size=row_group_size,
            queue_depth=queue_depth,
            persistence_seconds=elapsed,
            aggregate_peak_rss_bytes=4 * 1024**3,
            semantic_digest="b" * 64,
        )

    monkeypatch.setattr(
        stage052_calibration.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(available=16 * 1024**3),
    )
    contract_path = tmp_path / "stage052_resource_calibration.local.json"
    contract = run_stage052_resource_calibration(
        corpus_dir=tmp_path / "corpus",
        output_root=tmp_path / "calibration",
        contract_path=contract_path,
        root=tmp_path,
        config_path=tmp_path / "config.toml",
        producer_runner=producer_runner,
        parquet_runner=parquet_runner,
        memory_floor=ProducerMemoryFloor(
            four_worker_aggregate_peak_rss_bytes=1024,
            per_worker_peak_rss_bytes=1024,
            source_sha256_by_batch={
                "batch0001": "c" * 64,
                "batch0002": "d" * 64,
            },
        ),
    )

    assert observed_workers == [4, 5, 6]
    assert observed_parquet == [
        (65_536, 1),
        (65_536, 2),
        (262_144, 1),
        (262_144, 2),
    ]
    assert contract.selected_workers == 5
    assert contract.row_group_size == 65_536
    assert contract.queue_depth == 1
    assert load_producer_resource_contract(contract_path) == contract
    assert (tmp_path / "calibration" / "calibration_report.sha256").is_file()


def test_attempt73_memory_floor_uses_sealed_long_shard_resource_peaks(
    tmp_path: Path,
) -> None:
    for index, batch_id in enumerate(("batch0001", "batch0002"), start=1):
        batch_dir = tmp_path / batch_id
        resource_path = (
            batch_dir
            / "control"
            / "stage05.2_benchmark_attempt73_resource_summary.json"
        )
        resource_path.parent.mkdir(parents=True)
        atomic_write_signed_json(
            resource_path,
            {
                "status": "complete",
                "configured_worker_count": 4,
                "aggregate_peak_rss_bytes": index * 1000,
                "process_peak_rss_bytes": {"1": index * 100},
            },
        )
        resource_sha256 = hashlib.sha256(resource_path.read_bytes()).hexdigest()
        atomic_write_signed_json(
            batch_dir / "batch_manifest.json",
            {
                "run_label": "stage05.2_benchmark_attempt73",
                "batch_id": batch_id,
                "status": "archived",
                "resource_summary_sha256": resource_sha256,
            },
        )

    floor = load_attempt73_memory_floor(tmp_path)

    assert floor.four_worker_aggregate_peak_rss_bytes == 2000
    assert floor.per_worker_peak_rss_bytes == 200
    assert floor.projected_aggregate_peak_rss_bytes(6) == 3000
