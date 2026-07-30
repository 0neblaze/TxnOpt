from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from evrptw.artifacts import atomic_write_signed_json, signed_sidecar_matches
from evrptw.experiments import stage052_calibration
from evrptw.experiments.stage052_calibration import (
    FormalCampaignMemoryFloor,
    MeasuredProducerCandidate,
    ProducerMemoryFloor,
    load_attempt73_memory_floor,
    load_failed_formal_memory_floor,
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
            aggregate_memory_source="cgroup_v2",
            cgroup_path="/stage052-calibration.service",
            cgroup_swap_peak_bytes=0,
        )

    def parquet_runner(**kwargs) -> ParquetBenchmark:
        row_group_size = int(kwargs["row_group_size"])
        queue_depth = int(kwargs["queue_depth"])
        observed_parquet.append((row_group_size, queue_depth))
        elapsed = {
            (16_384, 1): 101.0,
            (16_384, 2): 99.0,
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

    def formal_memory_runner(**kwargs) -> MeasuredProducerCandidate:
        workers = int(kwargs["workers"])
        return MeasuredProducerCandidate(
            benchmark=ProducerBenchmark(
                workers=workers,
                throughput=1.0,
                aggregate_peak_rss_bytes=9 * 1024**3,
                semantic_digest="c" * 64,
                # Host-wide swap is telemetry only. The isolated cgroup swap
                # counter below is the hard Formal resource gate.
                swap_peak_bytes=12_345,
                fallback_count=0,
                resource_limit_exceeded=False,
            ),
            per_worker_peak_rss_bytes=3 * 1024**3,
            aggregate_memory_source="cgroup_v2",
            cgroup_path="/stage052-calibration.service",
            cgroup_swap_peak_bytes=0,
        )

    monkeypatch.setattr(
        stage052_calibration.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(
            available=14 * 1024**3,
            total=16 * 1024**3,
        ),
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
        formal_memory_runner=formal_memory_runner,
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
        (16_384, 1),
        (16_384, 2),
        (65_536, 1),
        (65_536, 2),
        (262_144, 1),
        (262_144, 2),
    ]
    assert contract.selected_workers == 6
    assert contract.selected_aggregate_peak_rss_bytes == 9 * 1024**3
    assert contract.row_group_size == 65_536
    assert contract.queue_depth == 1
    assert load_producer_resource_contract(contract_path) == contract
    assert (tmp_path / "calibration" / "calibration_report.sha256").is_file()
    report = json.loads(
        (tmp_path / "calibration" / "calibration_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["selection"]["policy"] == "user_locked"
    assert report["selection"]["locked_workers"] == 6
    assert (
        report["formal_memory_measurement"]["benchmark"]["swap_peak_bytes"]
        == 12_345
    )


def test_calibration_contract_includes_selected_formal_memory_measurement(
    tmp_path: Path,
    monkeypatch,
) -> None:
    formal_calls: list[tuple[int, int, int]] = []

    def producer_runner(**kwargs) -> MeasuredProducerCandidate:
        workers = int(kwargs["workers"])
        return MeasuredProducerCandidate(
            benchmark=ProducerBenchmark(
                workers=workers,
                throughput={4: 100.0, 5: 116.0, 6: 140.0}[workers],
                aggregate_peak_rss_bytes={4: 4, 5: 5, 6: 6}[workers] * 1024**3,
                semantic_digest="a" * 64,
                swap_peak_bytes=0,
                fallback_count=0,
                resource_limit_exceeded=False,
            ),
            per_worker_peak_rss_bytes=1024**3,
        )

    def parquet_runner(**kwargs) -> ParquetBenchmark:
        row_group_size = int(kwargs["row_group_size"])
        queue_depth = int(kwargs["queue_depth"])
        elapsed = {
            (16_384, 1): 110.0,
            (16_384, 2): 105.0,
            (65_536, 1): 100.0,
            (65_536, 2): 80.0,
            (262_144, 1): 90.0,
            (262_144, 2): 70.0,
        }[(row_group_size, queue_depth)]
        return ParquetBenchmark(
            row_group_size=row_group_size,
            queue_depth=queue_depth,
            persistence_seconds=elapsed,
            aggregate_peak_rss_bytes=1024**3,
            semantic_digest="b" * 64,
        )

    def formal_memory_runner(**kwargs) -> MeasuredProducerCandidate:
        workers = int(kwargs["workers"])
        row_group_size = int(kwargs["row_group_size"])
        queue_depth = int(kwargs["queue_depth"])
        formal_calls.append((workers, row_group_size, queue_depth))
        return MeasuredProducerCandidate(
            benchmark=ProducerBenchmark(
                workers=workers,
                throughput=1.0,
                aggregate_peak_rss_bytes=14 * 1024**3,
                semantic_digest="c" * 64,
                swap_peak_bytes=0,
                fallback_count=0,
                resource_limit_exceeded=False,
            ),
            per_worker_peak_rss_bytes=3 * 1024**3,
            aggregate_memory_source="cgroup_v2",
            cgroup_path="/stage052-calibration.service",
            cgroup_swap_peak_bytes=0,
        )

    monkeypatch.setattr(
        stage052_calibration.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(
            available=23 * 1024**3,
            total=24 * 1024**3,
        ),
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
        formal_memory_runner=formal_memory_runner,
        formal_campaign_memory_floor=FormalCampaignMemoryFloor(
            workers=6,
            aggregate_peak_rss_bytes=15 * 1024**3,
            per_worker_peak_rss_bytes=3_300_000_000,
            row_group_size=65_536,
            queue_depth=2,
            run_label="stage05.2_benchmark_attempt90",
            batch_id="batch0003",
            resource_summary_sha256="e" * 64,
        ),
        memory_floor=ProducerMemoryFloor(
            four_worker_aggregate_peak_rss_bytes=1024,
            per_worker_peak_rss_bytes=1024,
            source_sha256_by_batch={
                "batch0001": "c" * 64,
                "batch0002": "d" * 64,
            },
        ),
    )

    assert formal_calls == [(6, 262_144, 2)]
    assert contract.selected_workers == 6
    assert contract.selected_aggregate_peak_rss_bytes == 14 * 1024**3
    assert contract.selected_per_worker_peak_rss_bytes == 3_300_000_000
    report = json.loads(
        (tmp_path / "calibration" / "calibration_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["formal_memory_measurement"]["benchmark"]["semantic_digest"] == (
        "c" * 64
    )
    assert report["formal_campaign_memory_floor"]["run_label"] == (
        "stage05.2_benchmark_attempt90"
    )
    assert report["formal_campaign_memory_floor"]["row_group_size"] == 65_536
    assert report["selection"]["row_group_size"] == 262_144

    with pytest.raises(RuntimeError, match="floor worker count"):
        run_stage052_resource_calibration(
            corpus_dir=tmp_path / "corpus",
            output_root=tmp_path / "worker-mismatch-calibration",
            contract_path=tmp_path / "worker-mismatch-contract.json",
            root=tmp_path,
            config_path=tmp_path / "config.toml",
            producer_runner=producer_runner,
            parquet_runner=parquet_runner,
            formal_memory_runner=formal_memory_runner,
            formal_campaign_memory_floor=FormalCampaignMemoryFloor(
                workers=5,
                aggregate_peak_rss_bytes=15 * 1024**3,
                per_worker_peak_rss_bytes=3_300_000_000,
                row_group_size=65_536,
                queue_depth=2,
                run_label="stage05.2_benchmark_attempt90",
                batch_id="batch0003",
                resource_summary_sha256="e" * 64,
            ),
            memory_floor=ProducerMemoryFloor(
                four_worker_aggregate_peak_rss_bytes=1024,
                per_worker_peak_rss_bytes=1024,
                source_sha256_by_batch={
                    "batch0001": "c" * 64,
                    "batch0002": "d" * 64,
                },
            ),
        )


def test_measured_producer_candidate_rejects_shared_user_manager_cgroup() -> None:
    with pytest.raises(ValueError, match="cgroup v2 aggregate memory evidence"):
        MeasuredProducerCandidate(
            benchmark=ProducerBenchmark(
                workers=6,
                throughput=1.0,
                aggregate_peak_rss_bytes=1024,
                semantic_digest="a" * 64,
                swap_peak_bytes=0,
                fallback_count=0,
                resource_limit_exceeded=False,
            ),
            per_worker_peak_rss_bytes=1024,
            aggregate_memory_source="cgroup_v2",
            cgroup_path="/user.slice/user-1000.slice/user@1000.service",
            cgroup_swap_peak_bytes=0,
        )


def test_formal_memory_probe_seals_non_campaign_measurement(
    tmp_path: Path,
    monkeypatch,
) -> None:
    instance_path = tmp_path / "data" / "schneider" / "r205_21.txt"
    instance_path.parent.mkdir(parents=True)
    instance_path.write_text("formal-memory-probe fixture\n", encoding="utf-8")
    calls: list[tuple[int, int, int, int]] = []

    def runner(**kwargs) -> MeasuredProducerCandidate:
        calls.append(
            (
                int(kwargs["workers"]),
                int(kwargs["row_group_size"]),
                int(kwargs["queue_depth"]),
                int(kwargs["aggregate_memory_limit_bytes"]),
            )
        )
        return MeasuredProducerCandidate(
            benchmark=ProducerBenchmark(
                workers=6,
                throughput=10.0,
                aggregate_peak_rss_bytes=14 * 1024**3,
                semantic_digest="a" * 64,
                swap_peak_bytes=0,
                fallback_count=0,
                resource_limit_exceeded=False,
            ),
            per_worker_peak_rss_bytes=3 * 1024**3,
        )

    monkeypatch.setattr(
        stage052_calibration.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(
            available=23 * 1024**3,
            total=24 * 1024**3,
        ),
    )
    output_root = tmp_path / "formal-memory-probe"
    payload = stage052_calibration.run_stage052_formal_memory_probe(
        workers=6,
        row_group_size=65_536,
        queue_depth=2,
        output_root=output_root,
        root=tmp_path,
        config_path=tmp_path / "config.toml",
        run_label="stage05.2_formal_memory_probe_attempt01",
        runner=runner,
    )

    assert calls == [(6, 65_536, 2, 23 * 1024**3)]
    assert payload["campaign_geometry_contribution"] == 0
    assert payload["formal_memory_scope"] == {
        "instance": "r205_21",
        "seeds": [2014, 2015, 2016, 2017, 2018, 2019],
        "axes": ["wall_clock_30", "wall_clock_60", "wall_clock_300"],
    }
    report_path = output_root / "formal_memory_probe_report.json"
    assert json.loads(report_path.read_text(encoding="utf-8")) == payload
    assert report_path.with_suffix(".sha256").is_file()


def test_formal_memory_probe_seals_failure_before_reraising(
    tmp_path: Path,
    monkeypatch,
) -> None:
    instance_path = tmp_path / "data" / "schneider" / "r205_21.txt"
    instance_path.parent.mkdir(parents=True)
    instance_path.write_text("formal-memory-probe fixture\n", encoding="utf-8")

    def runner(**kwargs) -> MeasuredProducerCandidate:
        del kwargs
        raise RuntimeError("runtime guard aborted Stage 5.2 work: aggregate RSS")

    monkeypatch.setattr(
        stage052_calibration.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(
            available=23 * 1024**3,
            total=24 * 1024**3,
        ),
    )
    output_root = tmp_path / "failed-formal-memory-probe"

    with pytest.raises(RuntimeError, match="aggregate RSS"):
        stage052_calibration.run_stage052_formal_memory_probe(
            workers=8,
            row_group_size=65_536,
            queue_depth=2,
            output_root=output_root,
            root=tmp_path,
            config_path=tmp_path / "config.toml",
            run_label="stage05.2_formal_memory_probe_attempt04",
            runner=runner,
        )

    report_path = output_root / "formal_memory_probe_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["failure_reason"] == (
        "RuntimeError: runtime guard aborted Stage 5.2 work: aggregate RSS"
    )
    assert payload["measurement"] is None
    assert payload["campaign_geometry_contribution"] == 0
    assert signed_sidecar_matches(report_path, report_path.with_suffix(".sha256"))


def test_formal_memory_probe_rejects_missing_instance_before_consuming_label(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner_called = False

    def runner(**kwargs) -> MeasuredProducerCandidate:
        del kwargs
        nonlocal runner_called
        runner_called = True
        raise AssertionError("runner must not start without the exact probe input")

    monkeypatch.setattr(
        stage052_calibration.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(
            available=23 * 1024**3,
            total=24 * 1024**3,
        ),
    )
    output_root = tmp_path / "missing-input-formal-memory-probe"

    with pytest.raises(FileNotFoundError, match="r205_21.txt"):
        stage052_calibration.run_stage052_formal_memory_probe(
            workers=6,
            row_group_size=16_384,
            queue_depth=1,
            output_root=output_root,
            root=tmp_path,
            config_path=tmp_path / "config.toml",
            run_label="stage05.2_formal_memory_probe_attempt12",
            runner=runner,
        )

    assert not runner_called
    assert not output_root.exists()


def test_formal_memory_probe_rejects_external_instance_symlink_before_workers(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repository = tmp_path / "repository"
    (repository / "data").mkdir(parents=True)
    external_data = tmp_path / "external-schneider"
    external_data.mkdir()
    (external_data / "r205_21.txt").write_text(
        "external formal-memory-probe fixture\n",
        encoding="utf-8",
    )
    (repository / "data" / "schneider").symlink_to(
        external_data,
        target_is_directory=True,
    )
    runner_called = False

    def runner(**kwargs) -> MeasuredProducerCandidate:
        del kwargs
        nonlocal runner_called
        runner_called = True
        raise AssertionError("runner must not start with an external input symlink")

    monkeypatch.setattr(
        stage052_calibration.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(
            available=23 * 1024**3,
            total=24 * 1024**3,
        ),
    )
    output_root = tmp_path / "symlinked-input-formal-memory-probe"

    with pytest.raises(RuntimeError, match="snapshot-local ordinary file"):
        stage052_calibration.run_stage052_formal_memory_probe(
            workers=6,
            row_group_size=16_384,
            queue_depth=1,
            output_root=output_root,
            root=repository,
            config_path=repository / "config.toml",
            run_label="stage05.2_formal_memory_probe_attempt13",
            runner=runner,
        )

    assert not runner_called
    assert not output_root.exists()


def test_calibration_cli_runs_formal_memory_probe_without_corpus(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[dict[str, object]] = []

    def run_probe(**kwargs) -> dict[str, object]:
        calls.append(kwargs)
        return {"status": "complete"}

    monkeypatch.setattr(
        stage052_calibration,
        "run_stage052_formal_memory_probe",
        run_probe,
    )
    monkeypatch.setattr(
        stage052_calibration.sys,
        "argv",
        [
            "stage052_calibration",
            "--output-root",
            str(tmp_path / "probe"),
            "--repository-root",
            str(tmp_path / "repository"),
            "--run-label",
            "stage05.2_formal_memory_probe_attempt01",
            "--formal-memory-probe-workers",
            "6",
            "--row-group-size",
            "65536",
            "--queue-depth",
            "2",
            "--config",
            "configs/stage052_performance.toml",
        ],
    )

    assert stage052_calibration.main() == 0
    assert len(calls) == 1
    assert calls[0]["workers"] == 6
    assert calls[0]["row_group_size"] == 65_536
    assert calls[0]["queue_depth"] == 2
    assert calls[0]["output_root"] == (tmp_path / "probe").resolve()
    assert calls[0]["root"] == (tmp_path / "repository").resolve()
    assert calls[0]["config_path"] == (
        tmp_path / "repository" / "configs/stage052_performance.toml"
    ).resolve()


def test_calibration_cli_requires_and_loads_failed_formal_memory_floor(
    tmp_path: Path,
    monkeypatch,
) -> None:
    floor = FormalCampaignMemoryFloor(
        workers=6,
        aggregate_peak_rss_bytes=18_449_874_944,
        per_worker_peak_rss_bytes=3_326_586_880,
        row_group_size=65_536,
        queue_depth=2,
        run_label="stage05.2_benchmark_attempt90",
        batch_id="batch0003",
        resource_summary_sha256="a" * 64,
    )
    load_calls: list[tuple[Path, str]] = []
    calibration_calls: list[dict[str, object]] = []

    def load_floor(path: Path, *, batch_id: str) -> FormalCampaignMemoryFloor:
        load_calls.append((path, batch_id))
        return floor

    def run_calibration(**kwargs) -> SimpleNamespace:
        calibration_calls.append(kwargs)
        return SimpleNamespace(to_dict=lambda: {"selected_workers": 6})

    monkeypatch.setattr(
        stage052_calibration,
        "load_failed_formal_memory_floor",
        load_floor,
    )
    monkeypatch.setattr(
        stage052_calibration,
        "run_stage052_resource_calibration",
        run_calibration,
    )
    monkeypatch.setattr(
        stage052_calibration.sys,
        "argv",
        [
            "stage052_calibration",
            "--corpus-dir",
            str(tmp_path / "attempt73"),
            "--output-root",
            str(tmp_path / "calibration"),
            "--repository-root",
            str(tmp_path / "repository"),
            "--contract-path",
            str(tmp_path / "contract.json"),
            "--formal-memory-floor-dir",
            str(tmp_path / "attempt90"),
            "--formal-memory-floor-batch-id",
            "batch0003",
            "--config",
            str(tmp_path / "config.toml"),
        ],
    )

    assert stage052_calibration.main() == 0
    assert load_calls == [((tmp_path / "attempt90").resolve(), "batch0003")]
    assert calibration_calls[0]["formal_campaign_memory_floor"] == floor
    assert calibration_calls[0]["root"] == (tmp_path / "repository").resolve()


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


def _write_failed_formal_memory_floor(
    campaign_dir: Path,
    *,
    failure_reason: str = (
        "RuntimeError: batch process-tree aggregate RSS exceeds its campaign lock"
    ),
) -> tuple[Path, dict[str, object]]:
    run_label = "stage05.2_benchmark_attempt90"
    batch_id = "batch0003"
    batch_dir = campaign_dir / batch_id
    control_dir = batch_dir / "control"
    control_dir.mkdir(parents=True)
    resource_path = control_dir / f"{run_label}_resource_summary.json"
    resource = {
        "schema_version": "stage05.2-run-resource-v3",
        "run_label": run_label,
        "component": "benchmark",
        "configured_worker_count": 6,
        "measurement_scope": "task_scheduling_through_parent_control_preparation",
        "status": "complete",
        "parent_pid": 100,
        "descendant_pids": [101, 102, 103, 104, 105, 106],
        "sample_count": 50,
        "aggregate_peak_rss_bytes": 18_449_874_944,
        "process_peak_rss_bytes": {
            "100": 500_000_000,
            "101": 3_326_586_880,
            "102": 3_100_000_000,
            "103": 100_000_000,
            "104": 100_000_000,
            "105": 100_000_000,
            "106": 100_000_000,
        },
    }
    resource_path.write_text(json.dumps(resource), encoding="utf-8")
    resource_sha256 = hashlib.sha256(resource_path.read_bytes()).hexdigest()
    run_metadata_path = control_dir / f"{run_label}_run_metadata.json"
    run_metadata = {
        "run_label": run_label,
        "batch_id": batch_id,
        "component": "benchmark",
        "producer_resource_contract": {
            "selected_workers": 6,
            "row_group_size": 65_536,
            "queue_depth": 2,
        },
    }
    run_metadata_path.write_text(json.dumps(run_metadata), encoding="utf-8")
    run_metadata_sha256 = hashlib.sha256(run_metadata_path.read_bytes()).hexdigest()
    atomic_write_signed_json(
        control_dir / f"{run_label}_manifest.json",
        {
            "schema_version": "artifact-storage-v2",
            "run_label": run_label,
            "component": "benchmark",
            "status": "partial",
            "evidence_completeness": "partial",
            "artifacts": [
                {
                    "artifact_type": "manifest_metadata",
                    "relative_path": f"control/{run_label}_run_metadata.json",
                    "checksum": run_metadata_sha256,
                },
                {
                    "artifact_type": "resource_summary",
                    "relative_path": f"control/{run_label}_resource_summary.json",
                    "checksum": resource_sha256,
                    "evidence_completeness": "complete",
                },
            ],
        },
    )
    atomic_write_signed_json(
        batch_dir / "batch_manifest.json",
        {
            "schema_version": "stage05.2-batch-manifest-v1",
            "run_label": run_label,
            "batch_id": batch_id,
            "status": "failed",
            "failure_reason": failure_reason,
        },
    )
    atomic_write_signed_json(
        campaign_dir / "campaign_manifest.json",
        {
            "schema_version": "stage05.2-campaign-manifest-v2",
            "run_label": run_label,
            "scope": "formal",
            "status": "failed",
            "failure_reason": failure_reason,
            "selected_workers": 6,
            "producer_resource_contract": {
                "selected_workers": 6,
                "row_group_size": 65_536,
                "queue_depth": 2,
            },
            "batches": [
                {
                    "run_label": run_label,
                    "batch_id": batch_id,
                    "status": "failed",
                    "failure_reason": failure_reason,
                }
            ],
        },
    )
    return resource_path, resource


def test_failed_formal_memory_floor_requires_signed_failure_chain(
    tmp_path: Path,
) -> None:
    campaign_dir = tmp_path / "stage05.2_benchmark_attempt90"
    resource_path, _ = _write_failed_formal_memory_floor(campaign_dir)

    floor = load_failed_formal_memory_floor(
        campaign_dir,
        batch_id="batch0003",
    )

    assert floor == FormalCampaignMemoryFloor(
        workers=6,
        aggregate_peak_rss_bytes=18_449_874_944,
        per_worker_peak_rss_bytes=3_326_586_880,
        row_group_size=65_536,
        queue_depth=2,
        run_label="stage05.2_benchmark_attempt90",
        batch_id="batch0003",
        resource_summary_sha256=hashlib.sha256(resource_path.read_bytes()).hexdigest(),
        campaign_geometry_contribution=0,
    )


def test_failed_formal_memory_floor_accepts_process_rss_guard_failure(
    tmp_path: Path,
) -> None:
    campaign_dir = tmp_path / "stage05.2_benchmark_attempt99"
    resource_path, _ = _write_failed_formal_memory_floor(
        campaign_dir,
        failure_reason=(
            "RuntimeError: worker failure RuntimeError: runtime guard aborted "
            "Stage 5.2 work: process RSS hard limit exceeded: pid=63832 "
            "observed=3993497600 limit=3991904256; process-pool abort failure "
            "RuntimeError: pid=63833: survived terminate and kill"
        ),
    )

    floor = load_failed_formal_memory_floor(
        campaign_dir,
        batch_id="batch0003",
    )

    assert floor.per_worker_peak_rss_bytes == 3_326_586_880
    assert floor.resource_summary_sha256 == hashlib.sha256(
        resource_path.read_bytes()
    ).hexdigest()


def test_failed_formal_memory_floor_accepts_aggregate_rss_guard_failure(
    tmp_path: Path,
) -> None:
    campaign_dir = tmp_path / "stage05.2_benchmark_rerun02"
    resource_path, _ = _write_failed_formal_memory_floor(
        campaign_dir,
        failure_reason=(
            "RuntimeError: worker failure RuntimeError: runtime guard aborted "
            "Stage 5.2 work: aggregate RSS hard limit exceeded: "
            "observed=24073318400 limit=24072732672; process-pool abort failure "
            "RuntimeError: pid=70242: survived terminate and kill"
        ),
    )

    floor = load_failed_formal_memory_floor(
        campaign_dir,
        batch_id="batch0003",
    )

    assert floor.aggregate_peak_rss_bytes == 18_449_874_944
    assert floor.resource_summary_sha256 == hashlib.sha256(
        resource_path.read_bytes()
    ).hexdigest()


def test_failed_formal_memory_floor_rejects_non_memory_worker_failure(
    tmp_path: Path,
) -> None:
    campaign_dir = tmp_path / "stage05.2_benchmark_attempt99"
    _write_failed_formal_memory_floor(
        campaign_dir,
        failure_reason="RuntimeError: worker failure: unexpected exit",
    )

    with pytest.raises(RuntimeError, match="campaign identity"):
        load_failed_formal_memory_floor(campaign_dir, batch_id="batch0003")


def test_failed_formal_memory_floor_rejects_unbound_resource_edit(
    tmp_path: Path,
) -> None:
    campaign_dir = tmp_path / "stage05.2_benchmark_attempt90"
    resource_path, resource = _write_failed_formal_memory_floor(campaign_dir)
    resource["aggregate_peak_rss_bytes"] = 1
    resource_path.write_text(json.dumps(resource), encoding="utf-8")

    try:
        load_failed_formal_memory_floor(campaign_dir, batch_id="batch0003")
    except RuntimeError as error:
        assert "resource binding" in str(error)
    else:
        raise AssertionError("unbound resource edit was accepted")


def test_failed_formal_memory_floor_rejects_noncanonical_run_label(
    tmp_path: Path,
) -> None:
    campaign_dir = tmp_path / "stage05.2_benchmark_attempt90"
    _write_failed_formal_memory_floor(campaign_dir)
    campaign_path = campaign_dir / "campaign_manifest.json"
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    campaign["run_label"] = "stage05.2_benchmark_attempt90/../../escape"
    campaign["batches"][0]["run_label"] = campaign["run_label"]
    atomic_write_signed_json(campaign_path, campaign)

    with pytest.raises(RuntimeError, match="campaign identity"):
        load_failed_formal_memory_floor(campaign_dir, batch_id="batch0003")


def test_failed_formal_memory_floor_rejects_incomplete_pid_binding(
    tmp_path: Path,
) -> None:
    campaign_dir = tmp_path / "stage05.2_benchmark_attempt90"
    resource_path, resource = _write_failed_formal_memory_floor(campaign_dir)
    resource["descendant_pids"] = [101, 102, 103, 104, 105, 106, 107]
    resource_path.write_text(json.dumps(resource), encoding="utf-8")
    resource_sha256 = hashlib.sha256(resource_path.read_bytes()).hexdigest()
    artifact_manifest_path = (
        campaign_dir
        / "batch0003"
        / "control"
        / "stage05.2_benchmark_attempt90_manifest.json"
    )
    artifact_manifest = json.loads(
        artifact_manifest_path.read_text(encoding="utf-8")
    )
    resource_record = next(
        item
        for item in artifact_manifest["artifacts"]
        if item["artifact_type"] == "resource_summary"
    )
    resource_record["checksum"] = resource_sha256
    atomic_write_signed_json(artifact_manifest_path, artifact_manifest)

    with pytest.raises(RuntimeError, match="resource summary"):
        load_failed_formal_memory_floor(campaign_dir, batch_id="batch0003")
