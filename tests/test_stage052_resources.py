from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from pathlib import Path

import pytest

import evrptw.stage052_resources as resources_module
from evrptw.stage052_resources import (
    CapabilityRequirement,
    HostCapabilities,
    ParquetBenchmark,
    ProducerBenchmark,
    ProducerResourceContract,
    ReviewMemoryContract,
    RuntimeIdentity,
    derive_producer_resource_contract,
    derive_review_memory_contract,
    load_formal_resource_recalibration_evidence,
    load_producer_resource_contract,
    load_review_memory_contract,
    select_parquet_configuration,
    select_producer_configuration,
    validate_capabilities,
    verify_filesystem_capabilities,
)


def _write_formal_recalibration_report(
    path: Path,
    contract: ProducerResourceContract,
) -> None:
    payload = {
        "schema_version": "stage05.2-resource-calibration-report-v2",
        "run_label": "stage05.2_resource_calibration_attempt06",
        "campaign_geometry_contribution": 0,
        "memory_capacity_bytes": contract.available_memory_bytes,
        "contract": contract.to_dict(),
        "selection": {
            "policy": "user_locked",
            "locked_workers": contract.selected_workers,
            "selected_workers": contract.selected_workers,
            "row_group_size": contract.row_group_size,
            "queue_depth": contract.queue_depth,
            "rejected_reasons": {},
        },
        "formal_campaign_memory_floor": {
            "run_label": "stage05.2_benchmark_attempt99",
            "batch_id": "batch0007",
            "workers": contract.selected_workers,
            "aggregate_peak_rss_bytes": contract.selected_aggregate_peak_rss_bytes,
            "per_worker_peak_rss_bytes": contract.selected_per_worker_peak_rss_bytes,
            "row_group_size": contract.row_group_size,
            "queue_depth": contract.queue_depth,
            "resource_summary_sha256": (
                "cdaa627f53d14ce9a34d0054eb22cbaf4d388c80147980d428842c358599a7e5"
            ),
            "campaign_geometry_contribution": 0,
        },
        "formal_memory_measurement": {
            "benchmark": {
                "workers": contract.selected_workers,
                "throughput": 1.0,
                "aggregate_peak_rss_bytes": (
                    contract.selected_aggregate_peak_rss_bytes
                ),
                "semantic_digest": "9" * 64,
                "swap_peak_bytes": 0,
                "fallback_count": 0,
                "resource_limit_exceeded": False,
            },
            "per_worker_peak_rss_bytes": (
                contract.selected_per_worker_peak_rss_bytes
            ),
        },
        "fresh_producer_measurements": [
            {
                "workers": workers,
                "throughput": 1.0,
                "aggregate_peak_rss_bytes": (
                    contract.selected_aggregate_peak_rss_bytes
                    if workers == contract.selected_workers
                    else contract.selected_aggregate_peak_rss_bytes // 2
                ),
                "per_worker_peak_rss_bytes": (
                    contract.selected_per_worker_peak_rss_bytes
                    if workers == contract.selected_workers
                    else contract.selected_per_worker_peak_rss_bytes // 2
                ),
                "semantic_digest": contract.semantic_digest,
                "swap_peak_bytes": 0,
                "fallback_count": 0,
                "resource_limit_exceeded": False,
            }
            for workers in (4, 5, 6)
        ],
        "producer_measurements": [
            {
                "workers": workers,
                "throughput": 1.0,
                "aggregate_peak_rss_bytes": (
                    contract.selected_aggregate_peak_rss_bytes
                    if workers == contract.selected_workers
                    else contract.selected_aggregate_peak_rss_bytes // 2
                ),
                "per_worker_peak_rss_bytes": (
                    contract.selected_per_worker_peak_rss_bytes
                    if workers == contract.selected_workers
                    else contract.selected_per_worker_peak_rss_bytes // 2
                ),
                "semantic_digest": contract.semantic_digest,
                "swap_peak_bytes": 0,
                "fallback_count": 0,
                "resource_limit_exceeded": False,
            }
            for workers in (4, 5, 6)
        ],
        "provenance": {
            "repository_dirty": False,
            "dirty_path_count": 0,
            "repository_revision": "d" * 40,
            "configuration_sha256": "e" * 64,
            "native_extension_sha256": "f" * 64,
            "python_abi": "cpython-313",
        },
    }
    raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    path.write_bytes(raw)
    path.with_suffix(".sha256").write_text(
        hashlib.sha256(raw).hexdigest() + "\n",
        encoding="utf-8",
    )


def test_formal_resource_recalibration_binds_zero_geometry_memory_floor(
    tmp_path: Path,
) -> None:
    contract = ProducerResourceContract(
        selected_workers=6,
        available_memory_bytes=30_000_000_000,
        selected_aggregate_peak_rss_bytes=20_000_000_000,
        selected_per_worker_peak_rss_bytes=4_000_000_000,
        aggregate_memory_limit_bytes=24_000_000_000,
        per_worker_memory_limit_bytes=4_800_000_000,
        semantic_digest="a" * 64,
        calibration_digest="b" * 64,
        row_group_size=262_144,
        queue_depth=2,
    )
    report_path = tmp_path / "stage052_resource_calibration.local.report.json"
    _write_formal_recalibration_report(report_path, contract)

    evidence = load_formal_resource_recalibration_evidence(report_path, contract)

    assert evidence.report_run_label == "stage05.2_resource_calibration_attempt06"
    assert evidence.predecessor_run_label == "stage05.2_benchmark_attempt99"
    assert evidence.predecessor_batch_id == "batch0007"
    assert evidence.predecessor_resource_summary_sha256 == (
        "cdaa627f53d14ce9a34d0054eb22cbaf4d388c80147980d428842c358599a7e5"
    )
    assert evidence.formal_memory_semantic_digest == "9" * 64
    assert evidence.campaign_geometry_contribution == 0
    assert evidence.to_dict()["report_sha256"] == hashlib.sha256(
        report_path.read_bytes()
    ).hexdigest()


def test_formal_resource_recalibration_rejects_nonzero_geometry(
    tmp_path: Path,
) -> None:
    contract = ProducerResourceContract(
        selected_workers=6,
        available_memory_bytes=30_000_000_000,
        selected_aggregate_peak_rss_bytes=20_000_000_000,
        selected_per_worker_peak_rss_bytes=4_000_000_000,
        aggregate_memory_limit_bytes=24_000_000_000,
        per_worker_memory_limit_bytes=4_800_000_000,
        semantic_digest="a" * 64,
        calibration_digest="b" * 64,
        row_group_size=262_144,
        queue_depth=2,
    )
    report_path = tmp_path / "stage052_resource_calibration.local.report.json"
    _write_formal_recalibration_report(report_path, contract)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["formal_campaign_memory_floor"]["campaign_geometry_contribution"] = 1
    raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    report_path.write_bytes(raw)
    report_path.with_suffix(".sha256").write_text(
        hashlib.sha256(raw).hexdigest() + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="zero readiness geometry"):
        load_formal_resource_recalibration_evidence(report_path, contract)


@pytest.mark.parametrize(
    "field_path",
    [
        ("campaign_geometry_contribution",),
        ("formal_campaign_memory_floor", "campaign_geometry_contribution"),
        ("formal_memory_measurement", "benchmark", "swap_peak_bytes"),
        ("formal_memory_measurement", "benchmark", "fallback_count"),
        ("provenance", "dirty_path_count"),
    ],
)
def test_formal_resource_recalibration_rejects_boolean_zero_evidence(
    tmp_path: Path,
    field_path: tuple[str, ...],
) -> None:
    contract = ProducerResourceContract(
        selected_workers=6,
        available_memory_bytes=30_000_000_000,
        selected_aggregate_peak_rss_bytes=20_000_000_000,
        selected_per_worker_peak_rss_bytes=4_000_000_000,
        aggregate_memory_limit_bytes=24_000_000_000,
        per_worker_memory_limit_bytes=4_800_000_000,
        semantic_digest="a" * 64,
        calibration_digest="b" * 64,
        row_group_size=262_144,
        queue_depth=2,
    )
    report_path = tmp_path / "stage052_resource_calibration.local.report.json"
    _write_formal_recalibration_report(report_path, contract)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    target = payload
    for field_name in field_path[:-1]:
        target = target[field_name]
    target[field_path[-1]] = False
    raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    report_path.write_bytes(raw)
    report_path.with_suffix(".sha256").write_text(
        hashlib.sha256(raw).hexdigest() + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError):
        load_formal_resource_recalibration_evidence(report_path, contract)


def test_formal_resource_recalibration_rejects_swap_or_fallback(
    tmp_path: Path,
) -> None:
    contract = ProducerResourceContract(
        selected_workers=6,
        available_memory_bytes=30_000_000_000,
        selected_aggregate_peak_rss_bytes=20_000_000_000,
        selected_per_worker_peak_rss_bytes=4_000_000_000,
        aggregate_memory_limit_bytes=24_000_000_000,
        per_worker_memory_limit_bytes=4_800_000_000,
        semantic_digest="a" * 64,
        calibration_digest="b" * 64,
        row_group_size=262_144,
        queue_depth=2,
    )
    report_path = tmp_path / "stage052_resource_calibration.local.report.json"
    _write_formal_recalibration_report(report_path, contract)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["formal_memory_measurement"]["benchmark"]["swap_peak_bytes"] = 1
    raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    report_path.write_bytes(raw)
    report_path.with_suffix(".sha256").write_text(
        hashlib.sha256(raw).hexdigest() + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="swap/fallback"):
        load_formal_resource_recalibration_evidence(report_path, contract)


@pytest.mark.parametrize(
    ("field_path", "invalid_value"),
    [
        (("benchmark", "workers"), 5),
        (("benchmark", "aggregate_peak_rss_bytes"), 20_000_000_001),
        (("per_worker_peak_rss_bytes",), 4_000_000_001),
    ],
)
def test_formal_resource_recalibration_rejects_formal_probe_peak_drift(
    tmp_path: Path,
    field_path: tuple[str, ...],
    invalid_value: int,
) -> None:
    contract = ProducerResourceContract(
        selected_workers=6,
        available_memory_bytes=30_000_000_000,
        selected_aggregate_peak_rss_bytes=20_000_000_000,
        selected_per_worker_peak_rss_bytes=4_000_000_000,
        aggregate_memory_limit_bytes=24_000_000_000,
        per_worker_memory_limit_bytes=4_800_000_000,
        semantic_digest="a" * 64,
        calibration_digest="b" * 64,
        row_group_size=262_144,
        queue_depth=2,
    )
    report_path = tmp_path / "stage052_resource_calibration.local.report.json"
    _write_formal_recalibration_report(report_path, contract)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    target = payload["formal_memory_measurement"]
    for field_name in field_path[:-1]:
        target = target[field_name]
    target[field_path[-1]] = invalid_value
    raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    report_path.write_bytes(raw)
    report_path.with_suffix(".sha256").write_text(
        hashlib.sha256(raw).hexdigest() + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="Formal memory measurement/contract mismatch"):
        load_formal_resource_recalibration_evidence(report_path, contract)


def test_formal_resource_recalibration_requires_complete_worker_identity_set(
    tmp_path: Path,
) -> None:
    contract = ProducerResourceContract(
        selected_workers=6,
        available_memory_bytes=30_000_000_000,
        selected_aggregate_peak_rss_bytes=20_000_000_000,
        selected_per_worker_peak_rss_bytes=4_000_000_000,
        aggregate_memory_limit_bytes=24_000_000_000,
        per_worker_memory_limit_bytes=4_800_000_000,
        semantic_digest="a" * 64,
        calibration_digest="b" * 64,
        row_group_size=262_144,
        queue_depth=2,
    )
    report_path = tmp_path / "stage052_resource_calibration.local.report.json"
    _write_formal_recalibration_report(report_path, contract)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["fresh_producer_measurements"] = payload[
        "fresh_producer_measurements"
    ][1:]
    raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    report_path.write_bytes(raw)
    report_path.with_suffix(".sha256").write_text(
        hashlib.sha256(raw).hexdigest() + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="worker identity set"):
        load_formal_resource_recalibration_evidence(report_path, contract)


def test_formal_resource_recalibration_requires_exact_attempt99_failure_identity(
    tmp_path: Path,
) -> None:
    contract = ProducerResourceContract(
        selected_workers=6,
        available_memory_bytes=30_000_000_000,
        selected_aggregate_peak_rss_bytes=20_000_000_000,
        selected_per_worker_peak_rss_bytes=4_000_000_000,
        aggregate_memory_limit_bytes=24_000_000_000,
        per_worker_memory_limit_bytes=4_800_000_000,
        semantic_digest="a" * 64,
        calibration_digest="b" * 64,
        row_group_size=262_144,
        queue_depth=2,
    )
    report_path = tmp_path / "stage052_resource_calibration.local.report.json"
    _write_formal_recalibration_report(report_path, contract)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["formal_campaign_memory_floor"]["batch_id"] = "batch0008"
    raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    report_path.write_bytes(raw)
    report_path.with_suffix(".sha256").write_text(
        hashlib.sha256(raw).hexdigest() + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="Attempt99 batch0007"):
        load_formal_resource_recalibration_evidence(report_path, contract)


def _producer_result(
    workers: int,
    throughput: float,
    *,
    rss_gib: float,
    digest: str = "a" * 64,
    swap_bytes: int = 0,
    fallback_count: int = 0,
) -> ProducerBenchmark:
    return ProducerBenchmark(
        workers=workers,
        throughput=throughput,
        aggregate_peak_rss_bytes=int(rss_gib * 1024**3),
        semantic_digest=digest,
        swap_peak_bytes=swap_bytes,
        fallback_count=fallback_count,
        resource_limit_exceeded=False,
    )


def test_capability_contract_accepts_six_workers_without_hardware_identity_gates() -> None:
    requirement = CapabilityRequirement(
        workers=6,
        minimum_memory_bytes=12 * 1024**3,
        minimum_free_space_bytes=100 * 1024**3,
        backend="native_cpu",
        python_abi="cp313",
        native_extension_sha256="a" * 64,
    )
    capabilities = HostCapabilities(
        logical_cpu_count=24,
        available_memory_bytes=24 * 1024**3,
        free_space_bytes=700 * 1024**3,
        available_backends=frozenset({"native_cpu"}),
        python_abi="cp313",
        native_extension_sha256="a" * 64,
        filesystem_fsync=True,
        filesystem_atomic_replace=True,
    )

    validate_capabilities(capabilities, requirement)


def test_runtime_publication_identity_excludes_nonblocking_telemetry() -> None:
    hard = {
        "source_sha256": "a" * 64,
        "wheel_sha256": "b" * 64,
        "selected_workers": 6,
        "replay_backend": "native_arrow",
    }
    first = RuntimeIdentity(hard_contract=hard, telemetry={"power_source": "AC Power"})
    second = RuntimeIdentity(
        hard_contract=hard,
        telemetry={
            "power_source": "Battery",
            "cpu_model": "different telemetry",
            "disk_serial": "not publication identity",
        },
    )

    assert first.publication_sha256 == second.publication_sha256
    assert first.to_dict()["telemetry"] != second.to_dict()["telemetry"]


def test_producer_selection_applies_memory_speedup_digest_and_tie_rules() -> None:
    results = (
        _producer_result(4, 100.0, rss_gib=8.0),
        _producer_result(5, 118.0, rss_gib=9.0),
        _producer_result(6, 122.0, rss_gib=10.0),
    )

    selected = select_producer_configuration(results, available_memory_bytes=16 * 1024**3)

    # Five and six workers are within five percent, so the smaller candidate wins.
    assert selected.selected_workers == 5
    assert selected.semantic_digest == "a" * 64
    assert selected.candidate_workers == (4, 5, 6)
    assert selected.rejected_reasons == {}


def test_producer_selection_can_compare_an_optional_eight_worker_probe() -> None:
    results = (
        _producer_result(4, 100.0, rss_gib=8.0),
        _producer_result(5, 118.0, rss_gib=9.0),
        _producer_result(6, 122.0, rss_gib=10.0),
        _producer_result(8, 150.0, rss_gib=11.0),
    )

    selected = select_producer_configuration(results, available_memory_bytes=16 * 1024**3)

    assert selected.selected_workers == 8
    assert selected.candidate_workers == (4, 5, 6, 8)


def test_producer_selection_rejects_swap_fallback_digest_drift_and_memory_pressure() -> None:
    results = (
        _producer_result(4, 100.0, rss_gib=8.0),
        _producer_result(5, 140.0, rss_gib=9.0, swap_bytes=1),
        _producer_result(6, 160.0, rss_gib=17.0, digest="b" * 64),
    )

    selected = select_producer_configuration(results, available_memory_bytes=16 * 1024**3)

    assert selected.selected_workers == 4
    assert "swap pressure" in selected.rejected_reasons[5]
    assert "memory budget" in selected.rejected_reasons[6]
    assert "semantic digest" in selected.rejected_reasons[6]


def test_producer_selection_rejects_higher_concurrency_below_fifteen_percent() -> None:
    results = (
        _producer_result(4, 100.0, rss_gib=8.0),
        _producer_result(5, 114.9, rss_gib=9.0),
        _producer_result(6, 114.0, rss_gib=10.0),
    )

    selected = select_producer_configuration(results, available_memory_bytes=16 * 1024**3)

    assert selected.selected_workers == 4
    assert all(
        "15% throughput" in selected.rejected_reasons[workers] for workers in (5, 6)
    )


def test_producer_selection_uses_full_measured_memory_capability() -> None:
    results = (
        _producer_result(4, 100.0, rss_gib=13.0),
        _producer_result(5, 116.0, rss_gib=14.0),
        _producer_result(6, 130.0, rss_gib=15.0),
    )

    selected = select_producer_configuration(
        results,
        available_memory_bytes=16 * 1024**3,
    )

    assert selected.selected_workers == 6
    assert selected.rejected_reasons == {}


def test_producer_resource_contract_is_calibration_derived_and_round_trips() -> None:
    results = (
        _producer_result(4, 100.0, rss_gib=8.0),
        _producer_result(5, 118.0, rss_gib=9.0),
        _producer_result(6, 122.0, rss_gib=10.0),
    )
    selection = select_producer_configuration(
        results,
        available_memory_bytes=16 * 1024**3,
    )

    contract = derive_producer_resource_contract(
        results=results,
        selection=selection,
        selected_per_worker_peak_rss_bytes=3 * 1024**3,
        available_memory_bytes=16 * 1024**3,
        row_group_size=65_536,
        queue_depth=1,
    )

    assert contract.selected_workers == 5
    assert contract.aggregate_memory_limit_bytes == math.ceil(9 * 1024**3 * 1.2)
    assert contract.per_worker_memory_limit_bytes == math.ceil(3 * 1024**3 * 1.2)
    assert contract.aggregate_memory_limit_bytes <= 16 * 1024**3
    assert ProducerResourceContract.from_dict(contract.to_dict()) == contract


def test_producer_resource_contract_keeps_full_headroom_above_seventy_five_percent() -> None:
    results = (
        _producer_result(4, 100.0, rss_gib=8.0),
        _producer_result(5, 118.0, rss_gib=10.5),
        _producer_result(6, 122.0, rss_gib=11.0),
    )
    selection = select_producer_configuration(
        results,
        available_memory_bytes=16 * 1024**3,
    )

    contract = derive_producer_resource_contract(
        results=results,
        selection=selection,
        selected_per_worker_peak_rss_bytes=3 * 1024**3,
        available_memory_bytes=16 * 1024**3,
        row_group_size=65_536,
        queue_depth=1,
    )

    assert contract.selected_aggregate_peak_rss_bytes == int(10.5 * 1024**3)
    assert contract.aggregate_memory_limit_bytes == math.ceil(10.5 * 1024**3 * 1.2)
    assert contract.aggregate_memory_limit_bytes > int(16 * 1024**3 * 0.75)


def test_parquet_tuning_requires_ten_percent_critical_path_improvement() -> None:
    baseline = ParquetBenchmark(
        row_group_size=65_536,
        queue_depth=1,
        persistence_seconds=100.0,
        aggregate_peak_rss_bytes=4 * 1024**3,
        semantic_digest="a" * 64,
    )
    insufficient = ParquetBenchmark(
        row_group_size=262_144,
        queue_depth=2,
        persistence_seconds=91.0,
        aggregate_peak_rss_bytes=5 * 1024**3,
        semantic_digest="a" * 64,
    )
    accepted = ParquetBenchmark(
        row_group_size=262_144,
        queue_depth=1,
        persistence_seconds=89.0,
        aggregate_peak_rss_bytes=5 * 1024**3,
        semantic_digest="a" * 64,
    )

    assert (
        select_parquet_configuration(
            (baseline, insufficient),
            available_memory_bytes=16 * 1024**3,
        )
        == baseline
    )
    assert (
        select_parquet_configuration(
            (baseline, insufficient, accepted),
            available_memory_bytes=16 * 1024**3,
        )
        == accepted
    )


def test_parquet_tuning_uses_full_measured_memory_capability() -> None:
    baseline = ParquetBenchmark(
        row_group_size=65_536,
        queue_depth=1,
        persistence_seconds=100.0,
        aggregate_peak_rss_bytes=8 * 1024**3,
        semantic_digest="a" * 64,
    )
    faster = ParquetBenchmark(
        row_group_size=262_144,
        queue_depth=2,
        persistence_seconds=80.0,
        aggregate_peak_rss_bytes=13 * 1024**3,
        semantic_digest="a" * 64,
    )

    assert (
        select_parquet_configuration(
            (baseline, faster),
            available_memory_bytes=16 * 1024**3,
        )
        == faster
    )


def test_capability_contract_fails_fast_on_missing_atomic_filesystem_support() -> None:
    requirement = CapabilityRequirement(
        workers=4,
        minimum_memory_bytes=1,
        minimum_free_space_bytes=1,
        backend="native_cpu",
        python_abi="cp313",
        native_extension_sha256="a" * 64,
    )
    capabilities = HostCapabilities(
        logical_cpu_count=24,
        available_memory_bytes=24 * 1024**3,
        free_space_bytes=700 * 1024**3,
        available_backends=frozenset({"native_cpu"}),
        python_abi="cp313",
        native_extension_sha256="a" * 64,
        filesystem_fsync=True,
        filesystem_atomic_replace=False,
    )

    with pytest.raises(RuntimeError, match="atomic replace"):
        validate_capabilities(capabilities, requirement)


def test_review_memory_contract_is_pilot_derived_with_operating_headroom() -> None:
    contract = derive_review_memory_contract(
        parent_baseline_rss_bytes=512 * 1024**2,
        per_child_p99_rss_bytes=2 * 1024**3,
        review_workers=4,
        available_memory_bytes=16 * 1024**3,
    )

    assert contract.memory_swap_max_bytes == 0
    assert contract.review_workers == 4
    assert contract.memory_high_bytes < contract.process_guard_bytes
    assert contract.process_guard_bytes < contract.memory_max_bytes
    assert contract.memory_max_bytes <= 16 * 1024**3


def test_review_memory_contract_uses_full_measured_memory_capability() -> None:
    contract = derive_review_memory_contract(
        parent_baseline_rss_bytes=1 * 1024**3,
        per_child_p99_rss_bytes=3 * 1024**3,
        review_workers=4,
        available_memory_bytes=16 * 1024**3,
    )

    assert contract.memory_max_bytes == math.ceil(13 * 1024**3 * 1.2)
    assert contract.memory_max_bytes > int(16 * 1024**3 * 0.75)


def test_review_memory_contract_rejects_headroom_beyond_available_memory() -> None:
    with pytest.raises(RuntimeError, match="available memory"):
        derive_review_memory_contract(
            parent_baseline_rss_bytes=1 * 1024**3,
            per_child_p99_rss_bytes=4 * 1024**3,
            review_workers=4,
            available_memory_bytes=16 * 1024**3,
        )


@pytest.mark.parametrize(
    ("name", "contract", "loader"),
    (
        (
            "producer.json",
            ProducerResourceContract(
                selected_workers=5,
                available_memory_bytes=16 * 1024**3,
                selected_aggregate_peak_rss_bytes=8 * 1024**3,
                selected_per_worker_peak_rss_bytes=1536 * 1024**2,
                aggregate_memory_limit_bytes=10 * 1024**3,
                per_worker_memory_limit_bytes=2 * 1024**3,
                semantic_digest="a" * 64,
                calibration_digest="b" * 64,
                row_group_size=65_536,
                queue_depth=1,
            ),
            load_producer_resource_contract,
        ),
        (
            "review.json",
            ReviewMemoryContract(
                parent_baseline_rss_bytes=512 * 1024**2,
                per_child_p99_rss_bytes=1024**3,
                review_workers=4,
                available_memory_bytes=16 * 1024**3,
                memory_high_bytes=5 * 1024**3,
                process_guard_bytes=6 * 1024**3,
                memory_max_bytes=7 * 1024**3,
                memory_swap_max_bytes=0,
            ),
            load_review_memory_contract,
        ),
    ),
)
def test_signed_resource_contract_loaders_fail_closed_on_tamper(
    tmp_path: Path,
    name: str,
    contract: ProducerResourceContract | ReviewMemoryContract,
    loader: Callable[[Path], ProducerResourceContract | ReviewMemoryContract],
) -> None:
    path = tmp_path / name
    raw = (json.dumps(contract.to_dict(), sort_keys=True) + "\n").encode()
    path.write_bytes(raw)
    path.with_suffix(".sha256").write_text(
        hashlib.sha256(raw).hexdigest() + "\n",
        encoding="utf-8",
    )

    assert loader(path) == contract
    path.write_bytes(raw + b" ")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        loader(path)


def test_filesystem_capability_probe_verifies_fsync_and_atomic_replace(
    tmp_path: Path,
) -> None:
    verify_filesystem_capabilities(tmp_path)
    assert not tuple(tmp_path.iterdir())


def test_filesystem_capability_probe_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_replace(_source: Path, _destination: Path) -> None:
        raise OSError("atomic replace unavailable")

    monkeypatch.setattr(resources_module, "durable_replace", fail_replace)
    with pytest.raises(RuntimeError, match="fsync/atomic-replace"):
        verify_filesystem_capabilities(tmp_path)
