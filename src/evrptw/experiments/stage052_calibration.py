"""Real-shard Stage 5.2 producer and persistence calibration.

The calibration corpus is read-only and never contributes campaign geometry or
readiness counts.  Its only durable output is a signed resource contract that
new Pilot and Formal campaigns freeze into their configuration and manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from math import ceil
from pathlib import Path
from tempfile import TemporaryDirectory

import psutil  # type: ignore[import-untyped]
import pyarrow.parquet as pq

from evrptw.artifacts import (
    ARTIFACT_STORAGE_V2,
    SCREENING_DECISIONS_V3,
    ArtifactStorageConfig,
    _StreamingParquetSink,
    atomic_write_signed_json,
    signed_sidecar_matches,
)
from evrptw.experiments.stage02_route_reduction import (
    load_config as load_stage02_config,
)
from evrptw.experiments.stage04_weights import load_stage04_config
from evrptw.experiments.stage052_performance import (
    Stage052Axis,
    _build_tasks,
    _optimization_profile,
    _run_and_persist_v2_shard,
    _run_v2_tasks,
    _ShardTask,
    load_stage052_config,
)
from evrptw.parser import parse_schneider
from evrptw.repository import repository_root
from evrptw.stage052 import Stage052Component
from evrptw.stage052_evidence import ProcessTreeResourceSampler
from evrptw.stage052_resources import (
    ParquetBenchmark,
    ProducerBenchmark,
    ProducerResourceContract,
    ProducerSelection,
    derive_producer_resource_contract,
    select_parquet_configuration,
    select_producer_configuration,
)

_CALIBRATION_INSTANCES = ("c101_21", "r101_21", "rc101_21")
_CALIBRATION_SEEDS = (2014, 2015)
_PRODUCER_WORKERS = (4, 5, 6)
_PRODUCER_PROBE_WORKERS = (4, 5, 6, 8)
_PARQUET_CONFIGURATIONS = (
    (65_536, 1),
    (65_536, 2),
    (262_144, 1),
    (262_144, 2),
)


def _integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"calibration {field_name} is not an integer")
    return value


@dataclass(frozen=True, slots=True)
class MeasuredProducerCandidate:
    benchmark: ProducerBenchmark
    per_worker_peak_rss_bytes: int

    def __post_init__(self) -> None:
        if self.per_worker_peak_rss_bytes <= 0:
            raise ValueError("per-worker peak RSS must be positive")


@dataclass(frozen=True, slots=True)
class ProducerMemoryFloor:
    four_worker_aggregate_peak_rss_bytes: int
    per_worker_peak_rss_bytes: int
    source_sha256_by_batch: dict[str, str]

    def __post_init__(self) -> None:
        if (
            self.four_worker_aggregate_peak_rss_bytes <= 0
            or self.per_worker_peak_rss_bytes <= 0
            or set(self.source_sha256_by_batch) != {"batch0001", "batch0002"}
        ):
            raise ValueError("producer memory floor is incomplete")

    def projected_aggregate_peak_rss_bytes(self, workers: int) -> int:
        if workers not in _PRODUCER_PROBE_WORKERS:
            raise ValueError("producer memory floor workers must be 4, 5, 6, or 8")
        return ceil(self.four_worker_aggregate_peak_rss_bytes * workers / 4)


def load_attempt73_memory_floor(corpus_dir: Path) -> ProducerMemoryFloor:
    """Load the signed large-shard RSS floor without importing campaign geometry."""

    aggregate_peaks: list[int] = []
    per_worker_peaks: list[int] = []
    source_sha256_by_batch: dict[str, str] = {}
    for batch_id in ("batch0001", "batch0002"):
        batch_dir = corpus_dir / batch_id
        batch_manifest_path = batch_dir / "batch_manifest.json"
        if not signed_sidecar_matches(
            batch_manifest_path,
            batch_manifest_path.with_suffix(".sha256"),
        ):
            raise RuntimeError(f"Attempt73 {batch_id} manifest seal is invalid")
        try:
            batch_manifest = json.loads(batch_manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Attempt73 {batch_id} manifest is unreadable") from error
        resource_path = (
            batch_dir
            / "control"
            / "stage05.2_benchmark_attempt73_resource_summary.json"
        )
        if (
            not isinstance(batch_manifest, dict)
            or batch_manifest.get("run_label") != "stage05.2_benchmark_attempt73"
            or batch_manifest.get("batch_id") != batch_id
            or batch_manifest.get("status") != "archived"
            or batch_manifest.get("resource_summary_sha256") != _sha256(resource_path)
        ):
            raise RuntimeError(f"Attempt73 {batch_id} resource binding is invalid")
        try:
            resource = json.loads(resource_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Attempt73 {batch_id} resource summary is unreadable") from error
        process_peaks = resource.get("process_peak_rss_bytes")
        if (
            not isinstance(resource, dict)
            or resource.get("configured_worker_count") != 4
            or resource.get("status") != "complete"
            or not isinstance(process_peaks, dict)
            or not process_peaks
        ):
            raise RuntimeError(f"Attempt73 {batch_id} resource summary is invalid")
        aggregate_peaks.append(
            _integer(resource.get("aggregate_peak_rss_bytes"), "aggregate peak RSS")
        )
        per_worker_peaks.append(
            max(_integer(value, "process peak RSS") for value in process_peaks.values())
        )
        source_sha256_by_batch[batch_id] = _sha256(resource_path)
    return ProducerMemoryFloor(
        four_worker_aggregate_peak_rss_bytes=max(aggregate_peaks),
        per_worker_peak_rss_bytes=max(per_worker_peaks),
        source_sha256_by_batch=source_sha256_by_batch,
    )


def apply_memory_floor(
    measurements: Sequence[MeasuredProducerCandidate],
    floor: ProducerMemoryFloor,
) -> tuple[MeasuredProducerCandidate, ...]:
    """Apply a conservative linear 4→5/6 projection from sealed long shards."""

    return tuple(
        MeasuredProducerCandidate(
            benchmark=replace(
                measurement.benchmark,
                aggregate_peak_rss_bytes=max(
                    measurement.benchmark.aggregate_peak_rss_bytes,
                    floor.projected_aggregate_peak_rss_bytes(
                        measurement.benchmark.workers
                    ),
                ),
            ),
            per_worker_peak_rss_bytes=max(
                measurement.per_worker_peak_rss_bytes,
                floor.per_worker_peak_rss_bytes,
            ),
        )
        for measurement in measurements
    )


class _PeakRssMonitor:
    def __init__(self, *, interval_seconds: float = 0.02) -> None:
        self._interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._peak = 0
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> int:
        self._stop.set()
        self._thread.join()
        return self._peak

    def _run(self) -> None:
        process = psutil.Process()
        while not self._stop.wait(self._interval_seconds):
            self._peak = max(self._peak, int(process.memory_info().rss))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _source_state(repository: Path) -> dict[str, object]:
    try:
        revision = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty_paths = tuple(
            line
            for line in subprocess.run(
                ("git", "status", "--porcelain"),
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.splitlines()
            if line
        )
    except (OSError, subprocess.CalledProcessError):
        return {"repository_revision": None, "repository_dirty": None}
    return {
        "repository_revision": revision,
        "repository_dirty": bool(dirty_paths),
        "dirty_path_count": len(dirty_paths),
    }


def _calibration_provenance(
    *,
    repository: Path,
    config_path: Path,
    corpus_dir: Path,
) -> dict[str, object]:
    from evrptw import _core

    native_path = Path(str(_core.__file__)).resolve()
    corpus_manifests = tuple(
        path
        for path in (
            corpus_dir / "campaign_manifest.json",
            corpus_dir / "batch0001" / "batch_manifest.json",
            corpus_dir / "batch0002" / "batch_manifest.json",
        )
        if path.is_file()
    )
    return {
        **_source_state(repository),
        "python_abi": sys.implementation.cache_tag,
        "native_extension_sha256": _sha256(native_path),
        "configuration_sha256": _sha256(config_path) if config_path.is_file() else None,
        "corpus_manifest_sha256": {
            path.relative_to(corpus_dir).as_posix(): _sha256(path)
            for path in corpus_manifests
        },
        "producer_scope": {
            "instances": list(_CALIBRATION_INSTANCES),
            "seeds": list(_CALIBRATION_SEEDS),
            "termination": "fixed_work_100_exact_calls",
        },
    }


def _calibration_shard_runner(task: _ShardTask) -> list[dict[str, object]]:
    """Run one deterministic fixed-work shard in a fresh producer child."""

    config = load_stage052_config(task.config_path)
    stage04 = load_stage04_config(task.root / config.stage04_config)
    stage02 = load_stage02_config(task.root / config.stage02_config)
    instance = parse_schneider(
        task.root / config.benchmark_dir / f"{task.instance_name}.txt"
    )
    instance = replace(
        instance,
        distance_backend=_optimization_profile(Stage052Component.BENCHMARK),
    )
    axes = (
        Stage052Axis(
            name="fixed_work_calibration",
            termination_mode="fixed_work",
            time_limit_seconds=120.0,
            exact_call_budget=100,
            instrumentation_enabled=True,
            max_iterations=1_000,
        ),
    )
    return _run_and_persist_v2_shard(
        task,
        writer=None,
        config=config,
        stage04=stage04,
        stage02=stage02,
        instance=instance,
        axes=axes,
        storage=config.v2_storage,
    )


def benchmark_producer_candidate(
    *,
    workers: int,
    root: Path,
    config_path: Path,
    output_root: Path,
    run_label: str = "stage05.2_resource_calibration_attempt01",
) -> MeasuredProducerCandidate:
    """Execute the fixed six-shard high-memory scope with one worker candidate."""

    if workers not in _PRODUCER_PROBE_WORKERS:
        raise ValueError("producer calibration workers must be 4, 5, 6, or 8")
    config = load_stage052_config(config_path)
    candidate_dir = output_root / f"workers{workers}"
    candidate_dir.mkdir(parents=True, exist_ok=False)
    tasks = _build_tasks(
        root=root,
        config_path=config_path,
        run_dir=candidate_dir,
        run_label=run_label,
        component=Stage052Component.BENCHMARK,
        scope="calibration",
        instances=_CALIBRATION_INSTANCES,
        seeds=_CALIBRATION_SEEDS,
        worker_count=workers,
        storage=config.v2_storage,
    )
    sampler = ProcessTreeResourceSampler(
        run_label=run_label,
        component="resource_calibration",
        configured_worker_count=workers,
        interval_seconds=0.02,
    )
    swap_baseline = int(psutil.swap_memory().used)
    swap_peak = swap_baseline
    swap_stop = threading.Event()

    def sample_swap() -> None:
        nonlocal swap_peak
        while not swap_stop.wait(0.02):
            swap_peak = max(swap_peak, int(psutil.swap_memory().used))

    swap_thread = threading.Thread(target=sample_swap, daemon=True)
    sampler.start()
    swap_thread.start()
    started = time.perf_counter()
    try:
        rows = _run_v2_tasks(
            tasks,
            worker_count=workers,
            _task_runner=_calibration_shard_runner,
        )
    finally:
        elapsed = time.perf_counter() - started
        swap_stop.set()
        swap_thread.join()
        resource = sampler.stop()
    rows.sort(key=lambda row: (str(row["instance"]), _integer(row["seed"], "seed")))
    if len(rows) != len(tasks) or any(row.get("failure_status") for row in rows):
        raise RuntimeError("producer calibration did not complete its exact fixed scope")
    aggregate_digest = hashlib.sha256(
        json.dumps(
            [
                {
                    "instance": row["instance"],
                    "seed": row["seed"],
                    "semantic_digest": row["semantic_digest"],
                    "objective": [
                        row["vehicle_count"],
                        row["total_distance"],
                        row["total_charging_time"],
                        row["charging_count"],
                    ],
                    "exact_started_calls": row["exact_started_calls"],
                }
                for row in rows
            ],
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    total_exact_calls = sum(
        _integer(row["exact_started_calls"], "exact_started_calls") for row in rows
    )
    per_worker_peak = max(
        (
            int(rss)
            for pid, rss in resource.process_peak_rss_bytes
            if pid in set(resource.descendant_pids)
        ),
        default=0,
    )
    fallback_count = sum(
        _integer(row["native_fallbacks"], "native_fallbacks")
        + _integer(row["native_protocol_fallbacks"], "native_protocol_fallbacks")
        for row in rows
    )
    return MeasuredProducerCandidate(
        benchmark=ProducerBenchmark(
            workers=workers,
            throughput=total_exact_calls / elapsed,
            aggregate_peak_rss_bytes=resource.aggregate_peak_rss_bytes,
            semantic_digest=aggregate_digest,
            swap_peak_bytes=max(0, swap_peak - swap_baseline),
            fallback_count=fallback_count,
            resource_limit_exceeded=False,
        ),
        per_worker_peak_rss_bytes=per_worker_peak,
    )


def _write_parquet_copy(
    sources: Sequence[Path],
    destination_root: Path,
    *,
    row_group_size: int,
    queue_depth: int,
) -> None:
    storage = ArtifactStorageConfig(
        storage_policy_version=ARTIFACT_STORAGE_V2,
        screening_schema_version=SCREENING_DECISIONS_V3,
        compression_level=1,
        parquet_row_group_size=row_group_size,
        parquet_queue_depth=queue_depth,
    )
    for index, source in enumerate(sources):
        parquet = pq.ParquetFile(source)
        destination = destination_root / f"copy{index:02d}.parquet"
        sink = _StreamingParquetSink(destination, parquet.schema_arrow, storage)
        expected_rows = 0
        for batch in parquet.iter_batches(batch_size=65_536):
            sink.append_batch(batch)
            expected_rows += batch.num_rows
        observed_rows, schema_fingerprint = sink.close()
        if observed_rows != expected_rows or not schema_fingerprint:
            raise RuntimeError("Parquet calibration copy failed semantic row/schema replay")


def benchmark_parquet_configuration(
    *,
    corpus_dir: Path,
    row_group_size: int,
    queue_depth: int,
) -> ParquetBenchmark:
    """Replay the same two sealed large-shard event files through Zstandard 1."""

    sources = tuple(
        sorted(corpus_dir.glob("batch000[12]/c101_21/201[45]/*_events_*.parquet"))
    )
    if len(sources) != 2:
        raise RuntimeError("Attempt73 calibration corpus lacks two sealed c101_21 shards")
    semantic_digest = hashlib.sha256(
        json.dumps(
            [(str(path.relative_to(corpus_dir)), _sha256(path)) for path in sources],
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    with TemporaryDirectory(prefix="stage052-parquet-calibration-") as temporary:
        destination = Path(temporary)
        monitor = _PeakRssMonitor()
        monitor.start()
        started = time.perf_counter()
        _write_parquet_copy(
            sources,
            destination,
            row_group_size=row_group_size,
            queue_depth=queue_depth,
        )
        elapsed = time.perf_counter() - started
        peak_rss = monitor.stop()
    return ParquetBenchmark(
        row_group_size=row_group_size,
        queue_depth=queue_depth,
        persistence_seconds=elapsed,
        aggregate_peak_rss_bytes=peak_rss,
        semantic_digest=semantic_digest,
    )


def create_resource_contract(
    *,
    producer_measurements: Sequence[MeasuredProducerCandidate],
    parquet_measurements: Sequence[ParquetBenchmark],
    available_memory_bytes: int,
) -> tuple[ProducerResourceContract, ProducerSelection, ParquetBenchmark]:
    """Apply all deterministic selection gates to measured observations."""

    producer_results = tuple(item.benchmark for item in producer_measurements)
    selection = select_producer_configuration(
        producer_results,
        available_memory_bytes=available_memory_bytes,
    )
    selected_parquet = select_parquet_configuration(
        parquet_measurements,
        available_memory_bytes=available_memory_bytes,
    )
    selected_measurement = next(
        item
        for item in producer_measurements
        if item.benchmark.workers == selection.selected_workers
    )
    contract = derive_producer_resource_contract(
        results=producer_results,
        selection=selection,
        selected_per_worker_peak_rss_bytes=(
            selected_measurement.per_worker_peak_rss_bytes
        ),
        available_memory_bytes=available_memory_bytes,
        row_group_size=selected_parquet.row_group_size,
        queue_depth=selected_parquet.queue_depth,
    )
    return contract, selection, selected_parquet


def run_stage052_resource_calibration(
    *,
    corpus_dir: Path,
    output_root: Path,
    contract_path: Path,
    root: Path | None = None,
    config_path: Path | None = None,
    run_label: str = "stage05.2_resource_calibration_attempt01",
    require_clean_source: bool = False,
    memory_floor: ProducerMemoryFloor | None = None,
    producer_runner: Callable[..., MeasuredProducerCandidate] = benchmark_producer_candidate,
    parquet_runner: Callable[..., ParquetBenchmark] = benchmark_parquet_configuration,
) -> ProducerResourceContract:
    """Run all candidates, select once, and atomically publish a signed contract."""

    repository = repository_root() if root is None else root.resolve()
    resolved_config = (
        repository / "configs/stage052_performance.toml"
        if config_path is None
        else config_path.resolve()
    )
    source_state = _source_state(repository)
    if require_clean_source and (
        source_state.get("repository_revision") is None
        or source_state.get("repository_dirty") is not False
    ):
        raise RuntimeError("resource calibration requires a clean Git source revision")
    if output_root.exists():
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True)
    available_memory = int(psutil.virtual_memory().available)
    fresh_producer_measurements = tuple(
        producer_runner(
            workers=workers,
            root=repository,
            config_path=resolved_config,
            output_root=output_root,
            run_label=run_label,
        )
        for workers in _PRODUCER_WORKERS
    )
    memory_floor = (
        load_attempt73_memory_floor(corpus_dir)
        if memory_floor is None
        else memory_floor
    )
    producer_measurements = apply_memory_floor(
        fresh_producer_measurements,
        memory_floor,
    )
    parquet_measurements = tuple(
        parquet_runner(
            corpus_dir=corpus_dir,
            row_group_size=row_group_size,
            queue_depth=queue_depth,
        )
        for row_group_size, queue_depth in _PARQUET_CONFIGURATIONS
    )
    contract, selection, selected_parquet = create_resource_contract(
        producer_measurements=producer_measurements,
        parquet_measurements=parquet_measurements,
        available_memory_bytes=available_memory,
    )
    atomic_write_signed_json(contract_path, contract.to_dict())
    if not signed_sidecar_matches(contract_path, contract_path.with_suffix(".sha256")):
        raise RuntimeError("resource calibration contract sealing failed")
    atomic_write_signed_json(
        output_root / "calibration_report.json",
        {
            "schema_version": "stage05.2-resource-calibration-report-v1",
            "corpus_role": "read_only_benchmark_differential_only",
            "corpus_path": str(corpus_dir),
            "campaign_geometry_contribution": 0,
            "run_label": run_label,
            "provenance": _calibration_provenance(
                repository=repository,
                config_path=resolved_config,
                corpus_dir=corpus_dir,
            ),
            "available_memory_bytes": available_memory,
            "producer_measurements": [
                {
                    **asdict(item.benchmark),
                    "per_worker_peak_rss_bytes": item.per_worker_peak_rss_bytes,
                }
                for item in producer_measurements
            ],
            "fresh_producer_measurements": [
                {
                    **asdict(item.benchmark),
                    "per_worker_peak_rss_bytes": item.per_worker_peak_rss_bytes,
                }
                for item in fresh_producer_measurements
            ],
            "attempt73_memory_floor": asdict(memory_floor),
            "parquet_measurements": [asdict(item) for item in parquet_measurements],
            "selection": {
                "selected_workers": selection.selected_workers,
                "rejected_reasons": dict(selection.rejected_reasons),
                "row_group_size": selected_parquet.row_group_size,
                "queue_depth": selected_parquet.queue_depth,
            },
            "contract": contract.to_dict(),
        },
    )
    return contract


def run_stage052_producer_probe(
    *,
    workers: int,
    corpus_dir: Path,
    output_root: Path,
    root: Path | None = None,
    config_path: Path | None = None,
    run_label: str,
) -> dict[str, object]:
    """Run one explicitly non-campaign producer concurrency probe."""

    repository = repository_root() if root is None else root.resolve()
    resolved_config = (
        repository / "configs/stage052_performance.toml"
        if config_path is None
        else config_path.resolve()
    )
    measured = benchmark_producer_candidate(
        workers=workers,
        root=repository,
        config_path=resolved_config,
        output_root=output_root,
        run_label=run_label,
    )
    floor = load_attempt73_memory_floor(corpus_dir)
    payload: dict[str, object] = {
        "schema_version": "stage05.2-producer-probe-v1",
        "corpus_role": "exploratory_only_zero_campaign_geometry",
        "campaign_geometry_contribution": 0,
        "workers": workers,
        "fresh_measurement": asdict(measured),
        "projected_long_shard_aggregate_peak_rss_bytes": (
            floor.projected_aggregate_peak_rss_bytes(workers)
        ),
        "long_shard_per_worker_peak_rss_bytes": floor.per_worker_peak_rss_bytes,
        "attempt73_resource_sha256_by_batch": floor.source_sha256_by_batch,
        "provenance": _calibration_provenance(
            repository=repository,
            config_path=resolved_config,
            corpus_dir=corpus_dir,
        ),
    }
    atomic_write_signed_json(output_root / "probe_report.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate Stage 5.2 producer resources")
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--contract-path", type=Path)
    parser.add_argument("--run-label", default="stage05.2_resource_calibration_attempt01")
    parser.add_argument("--allow-dirty-source", action="store_true")
    parser.add_argument(
        "--probe-workers",
        type=int,
        choices=_PRODUCER_PROBE_WORKERS,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage052_performance.toml"),
    )
    arguments = parser.parse_args()
    if arguments.probe_workers is not None:
        payload = run_stage052_producer_probe(
            workers=arguments.probe_workers,
            corpus_dir=arguments.corpus_dir.resolve(),
            output_root=arguments.output_root.resolve(),
            config_path=arguments.config.resolve(),
            run_label=arguments.run_label,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if arguments.contract_path is None:
        parser.error("--contract-path is required unless --probe-workers is used")
    contract = run_stage052_resource_calibration(
        corpus_dir=arguments.corpus_dir.resolve(),
        output_root=arguments.output_root.resolve(),
        contract_path=arguments.contract_path.resolve(),
        config_path=arguments.config.resolve(),
        run_label=arguments.run_label,
        require_clean_source=not arguments.allow_dirty_source,
    )
    print(json.dumps(contract.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
