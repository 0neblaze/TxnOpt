from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from evrptw.experiment_lifecycle import ExperimentCatalog
from evrptw.experiments.stage052_performance_calibration import (
    MEMORY_ADMISSION_EVIDENCE_SCHEMA_VERSION,
    MODE_NAMES,
    OBSERVATION_PRODUCER_SCHEMA_VERSION,
    OBSERVATION_SCHEMA_VERSION,
    PERFORMANCE_BUILD_STORAGE_ALIAS,
    REQUIRED_BUILD_PROFILES,
    SUPPORTED_BUILD_PROFILES,
    WHEEL_RECEIPT_SCHEMA_VERSION,
    CalibrationError,
    calibrate_performance_profile,
    calibration_resource_request,
    load_fixed_work_observations,
    load_wheel_receipt,
    load_wheel_receipts,
    run_calibration_cli,
)
from evrptw.experiments.stage052_performance_calibration_review import (
    CalibrationReviewError,
    _resource_pss_components,
    _validate_live_memory_admission,
    _validate_resource_evidence,
)
from evrptw.experiments.stage052_telemetry_overhead import (
    TELEMETRY_SAMPLE_SCHEMA_VERSION,
    write_telemetry_overhead_receipt,
)
from evrptw.stage052_performance import (
    BuildArtifactIdentity,
    ExecutionTopology,
    HostPerformanceEnvelope,
    RuntimeResourceSummaryV2,
    TelemetryOverheadReceipt,
    generate_execution_topologies,
)
from evrptw.storage_governance import StartPermit

ROOT = Path(__file__).resolve().parents[1]


def test_performance_calibration_catalog_declares_stage052_prerequisites() -> None:
    catalog = ExperimentCatalog.from_toml(ROOT / "configs/experiment_catalog.toml")
    spec = catalog.for_run_label(
        "stage05.2_native_architecture_performance_calibration_attempt99"
    )

    assert spec.prerequisite_contracts == (
        "stage051_readiness",
        "historical_migration",
        "campaign_geometry",
    )
    assert spec.prerequisite_paths == {
        "stage051_readiness": (
            "experiments/manifests/stage05.1_best_known_artifact_manifest.json"
        )
    }


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sign(path: Path) -> None:
    Path(f"{path}.sha256").write_text(_sha(path.read_bytes()) + "\n", encoding="ascii")


def _telemetry_evidence(
    fingerprint: str,
    unmonitored: tuple[float, ...],
    monitored: tuple[float, ...],
) -> dict[str, object]:
    base: dict[str, object] = {
        "kind": "representative-fixed-work-axis",
        "mode": "current_stage052",
        "axis": "fixed_work",
        "instance": "c101C5",
        "exact_calls": 20,
        "iterations": 200,
        "batch_size": 128,
        "semantic_telemetry": True,
        "physical_telemetry": True,
        "persistence": True,
        "independent_replay": True,
        "minimal_validator_replay": True,
        "fingerprints_identical": True,
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
    sample_base = {
        key: value
        for key, value in base.items()
        if key
        not in {
            "semantic_telemetry",
            "physical_telemetry",
            "persistence",
            "independent_replay",
            "unmonitored_telemetry_surface",
            "monitored_telemetry_surface",
        }
    }

    def sample(enabled: bool, index: int, elapsed: float) -> dict[str, object]:
        return {
            "enabled": enabled,
            "sample_index": index,
            "elapsed_seconds": elapsed,
            "fingerprint": fingerprint,
            "resource_summary": {},
            "workload_evidence": {
                **sample_base,
                "semantic_telemetry": enabled,
                "physical_telemetry": enabled,
                "persistence": enabled,
                "independent_replay": enabled,
            },
        }

    orders = ("off-on", "on-off", "off-on", "on-off", "off-on")
    base["warm_sample_evidence"] = (
        sample(False, -2, 1.0),
        sample(True, -1, 1.0),
    )
    base["paired_sample_evidence"] = tuple(
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
    return base


def _overhead(tmp_path: Path) -> tuple[TelemetryOverheadReceipt, Path]:
    fingerprint = "f" * 64
    unmonitored = (1.0, 1.0, 1.0, 1.0, 1.0)
    monitored = (1.01, 1.01, 1.01, 1.01, 1.01)
    receipt = TelemetryOverheadReceipt(
        unmonitored_seconds=unmonitored,
        monitored_seconds=monitored,
        pair_orders=("off-on", "on-off", "off-on", "on-off", "off-on"),
        sample_interval_seconds=0.05,
        workload_output_sha256=hashlib.sha256(fingerprint.encode("ascii")).hexdigest(),
        monitored_resource_summaries=tuple({"sample_count": 10} for _ in range(5)),
        workload_evidence=_telemetry_evidence(
            fingerprint,
            unmonitored,
            monitored,
        ),
    )
    path = tmp_path / "telemetry-overhead.json"
    write_telemetry_overhead_receipt(path, receipt)
    return receipt, path


def test_calibration_reviewer_recomputes_parallel_diagnostics() -> None:
    summary = RuntimeResourceSummaryV2(
        elapsed_seconds=6.0,
        user_cpu_seconds=12.0,
        system_cpu_seconds=4.0,
        rss_bytes=400,
        pss_bytes=400,
        replay_seconds=1.5,
        worker_p95_seconds=5.0,
        worker_max_seconds=6.0,
        worker_min_seconds=4.0,
    )
    diagnostics: dict[str, object] = {
        "schema_version": "stage05.2-topology-parallel-diagnostics-v1",
        "shard_count": 4,
        "isolated_axis_end_to_end_seconds": 5.0,
        "sequential_projected_seconds": 20.0,
        "mode_block_end_to_end_seconds": 7.5,
        "speedup": 20.0 / 7.5,
        "parallel_efficiency": 20.0 / 7.5 / 4.0,
        "axes_per_hour": 3600.0 * 4.0 / 7.5,
        "cpu_seconds_per_axis": 4.0,
        "axes_per_cpu_second": 0.25,
        "worker_p95_over_min": 1.25,
        "worker_max_over_min": 1.5,
    }
    provenance = {
        "resource_summaries": [
            {
                "role": "admitted-mode-block",
                "resource_summary": summary.to_dict(),
                "parallel_diagnostics": diagnostics,
            }
        ]
    }

    assert _validate_resource_evidence(provenance) == 1
    diagnostics["parallel_efficiency"] = 1.0
    with pytest.raises(CalibrationReviewError, match="do not replay"):
        _validate_resource_evidence(provenance)


def test_calibration_reviewer_recomputes_live_memory_admission() -> None:
    host = _host()
    topology = ExecutionTopology(
        workload_class="c5",
        shards=((0,), (1,)),
        worker_count=2,
        request_threads=2,
    )
    evidence: dict[str, object] = {
        "schema_version": MEMORY_ADMISSION_EVIDENCE_SCHEMA_VERSION,
        "frozen_effective_memory_limit_bytes": 1_000,
        "live_host": host.to_dict(),
        "resident_pss_bytes": 400,
        "isolated_producer_pss_bytes": 100,
        "isolated_worker_descendant_pss_bytes": 100,
        "projected_concurrent_pss_bytes": 300,
        "incremental_required_bytes": 200,
        "admission": {
            "passed": True,
            "available_bytes": 1_000,
            "required_bytes": 200,
            "headroom_bytes": 40,
            "swap_total_bytes": 0,
            "swap_used_bytes": 0,
            "swap_current_bytes": 0,
            "reason": "live memory and swap admission passed",
        },
    }

    assert _validate_live_memory_admission(
        evidence,
        frozen_host=host,
        topology=topology,
        isolated_pss_bytes=200,
        scheduler_pss_bytes=0,
        producer_pss_bytes=100,
        worker_descendant_pss_bytes=100,
    )
    cast(dict[str, object], evidence["admission"])["available_bytes"] = 999
    with pytest.raises(CalibrationReviewError, match="does not replay"):
        _validate_live_memory_admission(
            evidence,
            frozen_host=host,
            topology=topology,
            isolated_pss_bytes=200,
            scheduler_pss_bytes=0,
            producer_pss_bytes=100,
            worker_descendant_pss_bytes=100,
        )


def test_calibration_reviewer_uses_simultaneous_worker_peak_receipt() -> None:
    process_tree: dict[str, object] = {
        "root_process_id": 10,
        "additional_root_pids": [20],
        "sample_count": 2,
        "peak_aggregate_pss_bytes": 100,
        "peak_worker_descendant_pss_bytes": 80,
        "worker_descendant_pss_peak": {
            "status": "available",
            "peak_bytes": 80,
            "sample_index": 0,
            "processes": [
                {"pid": 30, "create_time": 3.0, "pss_bytes": 80},
            ],
        },
        "process_metrics": [
            {"pid": 10, "create_time": 1.0, "maximum_pss_bytes": 100},
            {"pid": 20, "create_time": 2.0, "maximum_pss_bytes": 40},
            {"pid": 30, "create_time": 3.0, "maximum_pss_bytes": 80},
            {"pid": 31, "create_time": 4.0, "maximum_pss_bytes": 80},
        ],
    }

    assert _resource_pss_components(
        process_tree,
        scheduler_process_id=20,
    ) == (100, 40, 80)
def _start_permit(
    tmp_path: Path,
    *,
    run_label: str,
) -> StartPermit:
    permit_path = tmp_path / "permit.json"
    permit_path.write_text("{}", encoding="utf-8")
    return StartPermit(
        run_label=run_label,
        stage_plan_sha256="a" * 64,
        observation_sha256="b" * 64,
        observation_path=tmp_path / "observation.json",
        maintenance_audit_sha256="c" * 64,
        maintenance_audit_path=tmp_path / "maintenance.json",
        permit_path=permit_path,
    )


def _host(*, swap_used: int = 0) -> HostPerformanceEnvelope:
    return HostPerformanceEnvelope(
        allowed_cpu_ids=(0, 1),
        physical_core_groups=((0,), (1,)),
        memory_total_bytes=1_000,
        memory_available_bytes=1_000,
        swap_total_bytes=100 if swap_used else 0,
        swap_used_bytes=swap_used,
        swap_current_bytes=swap_used,
        topology_source="provided",
    )


def test_calibration_resource_request_scales_with_detected_cpu_set() -> None:
    host = replace(
        _host(),
        allowed_cpu_ids=(0, 2, 4, 6, 8),
        physical_core_groups=((0,), (2,), (4,), (6,), (8,)),
    )

    assert calibration_resource_request(host) == {
        "workers": 5,
        "threads": 11,
        "processes": 18,
    }


def _identity(tmp_path: Path, *, profile: str) -> tuple[BuildArtifactIdentity, dict[str, Path]]:
    contents = {
        "wheel": f"{profile}-wheel".encode(),
        "native": f"{profile}-native".encode(),
        "scheduler": f"{profile}-scheduler".encode(),
    }
    files: dict[str, Path] = {}
    for name, value in contents.items():
        path = tmp_path / f"{profile}-{name}.bin"
        path.write_bytes(value)
        files[name] = path
    flags = {
        "portable-o3": ("-O3",),
        "portable-lto": ("-O3", "-flto"),
        "host-native-lto": ("-O3", "-flto", "-march=native"),
    }[profile]
    return BuildArtifactIdentity(
        git_revision="a" * 40,
        git_tree="b" * 40,
        source_manifest_sha256="c" * 64,
        wheel_sha256=_sha(contents["wheel"]),
        native_sha256=_sha(contents["native"]),
        scheduler_sha256=_sha(contents["scheduler"]),
        compiler_version="gcc 14.2",
        flags=flags,
        cpu_feature_mask=("sse4_2",),
    ), files


def _receipts(
    tmp_path: Path,
    *,
    profiles: tuple[str, ...] = SUPPORTED_BUILD_PROFILES,
) -> list[Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for profile in profiles:
        identity, files = _identity(tmp_path, profile=profile)
        path = tmp_path / f"{profile}.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": WHEEL_RECEIPT_SCHEMA_VERSION,
                    "build_profile": profile,
                    "no_cache_build": True,
                    "source_dirty": False,
                    "development_override": False,
                    "storage_alias": PERFORMANCE_BUILD_STORAGE_ALIAS,
                    "wheel_relative_path": files["wheel"].relative_to(path.parent).as_posix(),
                    "native_relative_path": files["native"].relative_to(path.parent).as_posix(),
                    "scheduler_relative_path": files["scheduler"]
                    .relative_to(path.parent)
                    .as_posix(),
                    "compiler_id": "gcc",
                    "artifact_identity": identity.to_dict(),
                }
            ),
            encoding="utf-8",
        )
        _sign(path)
        paths.append(path)
    return paths


def _topology(workload: str, *, scheduler: bool = False) -> ExecutionTopology:
    return generate_execution_topologies(
        (0, 1),
        workload_class=workload,
        scheduler_cpu_ids=(0,) if scheduler else (),
    )[0]


def _axis(
    *,
    mode: str,
    workload: str,
    topology: ExecutionTopology,
    seconds: float,
    semantic_variant: str = "same",
    pss_bytes: int = 100,
) -> dict[str, object]:
    marker = "route" if semantic_variant == "same" else semantic_variant
    return {
        "mode": mode,
        "workload_class": workload,
        "instance": "c101C5" if workload == "c5" else "c101_21",
        "seed": 2014,
        "topology_id": f"{workload}-two-shard",
        "topology": topology.to_dict(),
        "producer_end_to_end_seconds": seconds * 0.9,
        "independent_replay_seconds": seconds * 0.1,
        "end_to_end_seconds": seconds,
        "pss_bytes": pss_bytes,
        "scheduler_pss_bytes": pss_bytes // 10 if mode == "host_scheduler" else 0,
        "producer_pss_bytes": pss_bytes // 5,
        "worker_descendant_pss_bytes": (
            pss_bytes
            - pss_bytes // 5
            - (pss_bytes // 10 if mode == "host_scheduler" else 0)
        ),
        "objective": {"vehicle_count": 1, "distance": 10.0},
        "routes": [["depot", marker, "depot"]],
        "candidate_trajectory": ["candidate-0", marker],
        "exact_order": ["route-0"],
        "cache_lifecycle": {"lookup": 1, "store": 1},
        "transaction_hashes": ["d" * 64],
        "confidence_interval": None,
    }


def _write_observations(
    tmp_path: Path,
    *,
    mismatch: str | None = None,
    pss_bytes: int = 100,
    profiles: tuple[str, ...] = SUPPORTED_BUILD_PROFILES,
) -> list[Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    parent_run_label = "stage05.2_native_architecture_performance_calibration_attempt99"
    run_root = tmp_path / parent_run_label
    run_root.mkdir()
    permit_path = run_root / "start-permit.json"
    permit_path.write_text(
        json.dumps({"run_label": parent_run_label, "status": "reserved"}),
        encoding="utf-8",
    )
    _sign(permit_path)
    host_path = run_root / "host-envelope.json"
    host_path.write_text(json.dumps(_host().to_dict()), encoding="utf-8")
    _sign(host_path)
    paths: list[Path] = []
    for profile in profiles:
        identity, _ = _identity(tmp_path, profile=profile)
        lifecycle = {
            workload: [
                {
                    "build_profile": profile,
                    "topology_id": f"{workload}-two-shard",
                    "topology": _topology(workload, scheduler=True).to_dict(),
                    "session_isolation_passed": True,
                    "cache_reset_passed": True,
                    "rss_stability_passed": True,
                    "semantic_replay_passed": True,
                    "mode_block_faster": True,
                    "per_wave_end_to_end_seconds": 2.0,
                    "mode_block_end_to_end_seconds": 1.5,
                    "per_wave_scheduler_pss_peak_bytes": 50,
                    "mode_block_scheduler_pss_peak_bytes": 50,
                    "per_wave_process_tree_pss_peak_bytes": 100,
                    "mode_block_process_tree_pss_peak_bytes": 100,
                    "wave_count": 2,
                }
            ]
            for workload in ("c5", "100-customer")
        }
        for repeat in range(3):
            axes: list[dict[str, object]] = []
            for mode in MODE_NAMES:
                for workload in ("c5", "100-customer"):
                    variant = profile if profile == mismatch else "same"
                    axes.append(
                        _axis(
                            mode=mode,
                            workload=workload,
                            topology=_topology(workload, scheduler=mode == "host_scheduler"),
                            seconds={
                                "portable-o3": 10.0,
                                "portable-lto": 9.9,
                                "host-native-lto": 9.8,
                            }[profile],
                            semantic_variant=variant,
                            pss_bytes=pss_bytes,
                        )
                    )
            path = run_root / "observations" / profile / f"repeat{repeat + 1}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            raw_path = run_root / "raw" / profile / f"repeat{repeat + 1}.json"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_text(
                json.dumps({"axes": axes}, sort_keys=True),
                encoding="utf-8",
            )
            _sign(raw_path)
            raw_sidecar = Path(f"{raw_path}.sha256")
            raw_persistence = Path(f"{raw_path}.persistence")
            raw_persistence.write_text("test-persistence-receipt\n", encoding="ascii")
            raw_persistence_sidecar = Path(f"{raw_persistence}.sha256")
            raw_persistence_sidecar.write_text(
                _sha(raw_persistence.read_bytes()) + "\n",
                encoding="ascii",
            )
            path.write_text(
                json.dumps(
                    {
                        "schema_version": OBSERVATION_SCHEMA_VERSION,
                        "build_profile": profile,
                        "repeat": repeat,
                        "warm_start": {"source": "stage033", "identity": "same"},
                        "axes": axes,
                        "host_scheduler_lifecycle_evidence": lifecycle,
                        "build_identity": identity.to_dict(),
                        "fixed_work_budget": {"exact_calls": 100, "iterations": 40},
                        "memory_rejections": [],
                        "producer_provenance": {
                            "schema_version": OBSERVATION_PRODUCER_SCHEMA_VERSION,
                            "repository_revision": "a" * 40,
                            "producer_source_sha256": "e" * 64,
                            "parent_run_label": parent_run_label,
                            "storage_alias": "stage052-performance-calibration-run",
                            "start_permit_relative_path": "start-permit.json",
                            "start_permit_sha256": _sha(permit_path.read_bytes()),
                            "host_envelope_relative_path": "host-envelope.json",
                            "host_envelope_sha256": _sha(host_path.read_bytes()),
                            "raw_axis_inventory": [
                                {
                                    "storage_alias": "stage052-performance-calibration-run",
                                    "relative_path": raw_path.relative_to(run_root).as_posix(),
                                    "sha256": _sha(raw_path.read_bytes()),
                                    "relative_sidecar_path": raw_sidecar.relative_to(
                                        run_root
                                    ).as_posix(),
                                    "sidecar_sha256": _sha(raw_sidecar.read_bytes()),
                                    "role": "test-observation-batch",
                                    "supporting_artifacts": [
                                        {
                                            "relative_path": raw_persistence.relative_to(
                                                run_root
                                            ).as_posix(),
                                            "sha256": _sha(raw_persistence.read_bytes()),
                                            "role": "axis-persistence-receipt",
                                        },
                                        {
                                            "relative_path": raw_persistence_sidecar.relative_to(
                                                run_root
                                            ).as_posix(),
                                            "sha256": _sha(raw_persistence_sidecar.read_bytes()),
                                            "role": "axis-persistence-receipt-sidecar",
                                        },
                                    ],
                                }
                            ],
                            "scheduler_task_receipt_inventory": [],
                            "resource_summaries": [
                                {"schema_version": "test-resource-v1", "sample_count": 1}
                            ],
                        },
                    }
                ),
                encoding="utf-8",
            )
            _sign(path)
            paths.append(path)
    return paths


def test_calibration_freezes_profile_and_signed_outputs(tmp_path: Path) -> None:
    receipts = load_wheel_receipts(_receipts(tmp_path))
    observations = load_fixed_work_observations(_write_observations(tmp_path))
    overhead, overhead_path = _overhead(tmp_path)
    result = calibrate_performance_profile(
        run_label="stage05.2_native_architecture_performance_calibration_attempt01",
        wheel_receipts=receipts,
        observations=observations,
        host=_host(),
        telemetry_overhead=overhead,
        telemetry_overhead_path=overhead_path,
    )
    assert result.selected_build_profile == "portable-lto"
    calibration = result.profile.calibration
    selection_statistics = cast(Mapping[str, object], calibration["build_selection_statistics"])
    assert selection_statistics["method"] == ("paired-exact-bootstrap-median-aggregate-e2e-v1")
    assert selection_statistics["fastest_build"] == "host-native-lto"
    assert selection_statistics["eligible_builds"] == (
        "host-native-lto",
        "portable-lto",
        "portable-o3",
    )
    paired_intervals = cast(
        Mapping[str, object], selection_statistics["paired_relative_ci_vs_fastest"]
    )
    assert cast(tuple[float, float], paired_intervals["portable-lto"])[0] >= 0.0
    required = cast(Mapping[str, object], calibration["memory_required_bytes_by_topology"])
    assert set(required) == {
        f"{mode}:{workload}" for mode in MODE_NAMES for workload in ("c5", "100-customer")
    }
    assert calibration["host_scheduler_lifecycle_by_workload"] == {
        "c5": "mode-block",
        "100-customer": "mode-block",
    }
    result_dir = tmp_path / "results"
    # Publish with the direct writer in this unit test to avoid reading the live host.
    from evrptw.experiments.stage052_performance_calibration import write_calibration_bundle

    outputs = write_calibration_bundle(result, output_root=result_dir)
    assert outputs["profile"].is_file()
    assert outputs["profile_sidecar"].read_text().strip() == _sha(outputs["profile"].read_bytes())
    assert outputs["receipt"].is_file()
    with pytest.raises(FileExistsError):
        write_calibration_bundle(result, output_root=result_dir)


def test_semantic_mismatch_rejects_only_divergent_candidate(tmp_path: Path) -> None:
    receipts = load_wheel_receipts(_receipts(tmp_path))
    observations = load_fixed_work_observations(
        _write_observations(tmp_path / "observations", mismatch="host-native-lto")
    )
    overhead, overhead_path = _overhead(tmp_path)
    result = calibrate_performance_profile(
        run_label="stage05.2_native_architecture_performance_calibration_attempt02",
        wheel_receipts=receipts,
        observations=observations,
        host=_host(),
        telemetry_overhead=overhead,
        telemetry_overhead_path=overhead_path,
    )
    assert "host-native-lto" in result.rejected_builds
    assert result.selected_build_profile == "portable-lto"


def test_host_native_build_is_optional_when_compiler_probe_rejects_it(
    tmp_path: Path,
) -> None:
    receipts = load_wheel_receipts(_receipts(tmp_path, profiles=REQUIRED_BUILD_PROFILES))
    observations = load_fixed_work_observations(
        _write_observations(
            tmp_path / "observations",
            profiles=REQUIRED_BUILD_PROFILES,
        )
    )
    overhead, overhead_path = _overhead(tmp_path)
    result = calibrate_performance_profile(
        run_label="stage05.2_native_architecture_performance_calibration_attempt07",
        wheel_receipts=receipts,
        observations=observations,
        host=_host(),
        telemetry_overhead=overhead,
        telemetry_overhead_path=overhead_path,
    )
    assert result.selected_build_profile == "portable-lto"
    assert {receipt.build_profile for receipt in result.wheel_receipts} == set(
        REQUIRED_BUILD_PROFILES
    )


def test_build_selection_uses_each_builds_best_safe_topology(tmp_path: Path) -> None:
    receipt_paths = _receipts(tmp_path / "inputs")
    observation_paths = _write_observations(tmp_path / "observations")
    for path in observation_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        profile = cast(str, payload["build_profile"])
        extra_axes: list[dict[str, object]] = []
        for raw in payload["axes"]:
            axis = cast(dict[str, object], raw)
            if profile == "host-native-lto":
                axis["producer_end_to_end_seconds"] = 7.2
                axis["independent_replay_seconds"] = 0.8
                axis["end_to_end_seconds"] = 8.0
            mode = cast(str, axis["mode"])
            workload = cast(str, axis["workload_class"])
            topology = (
                ExecutionTopology(
                    workload_class=workload,
                    shards=((0,), (1,)),
                    worker_count=2,
                    scheduler_cpu_ids=(0, 1),
                    request_threads=2,
                    affinity_policy="free_scheduler",
                    allow_affinity_overlap=True,
                )
                if mode == "host_scheduler"
                else ExecutionTopology(
                    workload_class=workload,
                    shards=((0, 1),),
                    worker_count=1,
                    request_threads=1,
                    affinity_policy="uniform",
                )
            )
            extra = dict(axis)
            extra["topology_id"] = f"{workload}-extra-topology"
            extra["topology"] = topology.to_dict()
            extra_seconds = {
                "portable-o3": 100.0,
                "portable-lto": 11.0,
                "host-native-lto": 200.0,
            }[profile]
            extra["producer_end_to_end_seconds"] = extra_seconds * 0.9
            extra["independent_replay_seconds"] = extra_seconds * 0.1
            extra["end_to_end_seconds"] = extra_seconds
            extra_axes.append(extra)
        payload["axes"].extend(extra_axes)
        path.write_text(json.dumps(payload), encoding="utf-8")
        _sign(path)

    receipts = load_wheel_receipts(receipt_paths)
    observations = load_fixed_work_observations(observation_paths)
    overhead, overhead_path = _overhead(tmp_path)
    result = calibrate_performance_profile(
        run_label="stage05.2_native_architecture_performance_calibration_attempt10",
        wheel_receipts=receipts,
        observations=observations,
        host=_host(),
        telemetry_overhead=overhead,
        telemetry_overhead_path=overhead_path,
    )

    assert result.selected_build_profile == "host-native-lto"


def test_swap_is_a_hard_topology_gate(tmp_path: Path) -> None:
    receipts = load_wheel_receipts(_receipts(tmp_path))
    observations = load_fixed_work_observations(_write_observations(tmp_path / "observations"))
    overhead, overhead_path = _overhead(tmp_path)
    with pytest.raises(CalibrationError, match="memory-safe topology"):
        calibrate_performance_profile(
            run_label="stage05.2_native_architecture_performance_calibration_attempt03",
            wheel_receipts=receipts,
            observations=observations,
            host=_host(swap_used=1),
            telemetry_overhead=overhead,
            telemetry_overhead_path=overhead_path,
        )


def test_single_axis_pss_is_projected_over_concurrent_shards(
    tmp_path: Path,
) -> None:
    receipts = load_wheel_receipts(_receipts(tmp_path))
    observations = load_fixed_work_observations(
        _write_observations(
            tmp_path / "observations",
            pss_bytes=500,
        )
    )
    overhead, overhead_path = _overhead(tmp_path)
    with pytest.raises(CalibrationError, match="memory-safe topology"):
        calibrate_performance_profile(
            run_label="stage05.2_native_architecture_performance_calibration_attempt05",
            wheel_receipts=receipts,
            observations=observations,
            host=_host(),
            telemetry_overhead=overhead,
            telemetry_overhead_path=overhead_path,
        )


def test_unknown_json_field_and_duplicate_repeat_fail(tmp_path: Path) -> None:
    paths = _write_observations(tmp_path / "observations")
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    payload["unexpected"] = True
    paths[0].write_text(json.dumps(payload), encoding="utf-8")
    _sign(paths[0])
    with pytest.raises(CalibrationError, match="fields mismatch"):
        load_fixed_work_observations(paths)


def test_cli_requires_valid_inputs_after_lifecycle_admission(tmp_path: Path) -> None:
    run_label = "stage05.2_native_architecture_performance_calibration_attempt04"
    output_root = tmp_path / "results"
    permit = _start_permit(
        tmp_path,
        run_label=run_label,
    )
    with pytest.raises(CalibrationError):
        run_calibration_cli(
            wheel_receipt_paths=[],
            observation_paths=[],
            output_root=output_root,
            run_label=run_label,
            host_envelope_path=None,
            telemetry_overhead_receipt_path=tmp_path / "missing-telemetry.json",
            start_permit=permit,
        )
    assert not (output_root / run_label).exists()
    with pytest.raises(CalibrationError):
        run_calibration_cli(
            wheel_receipt_paths=[],
            observation_paths=[],
            output_root=tmp_path / "results",
            run_label=run_label,
            host_envelope_path=None,
            telemetry_overhead_receipt_path=tmp_path / "missing-telemetry.json",
            start_permit=permit,
        )


def test_observation_children_inherit_the_running_calibration_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from evrptw.experiments import stage052_performance_calibration as calibration

    receipt = load_wheel_receipt(_receipts(tmp_path / "inputs", profiles=("portable-o3",))[0])
    run_label = "stage05.2_native_architecture_performance_calibration_attempt12"
    run_dir = tmp_path / run_label
    run_dir.mkdir()
    permit = _start_permit(tmp_path, run_label=run_label)
    permit.permit_path.write_text("{}", encoding="utf-8")
    host_path = run_dir / "host.json"
    host_path.write_text("{}", encoding="utf-8")
    root = tmp_path / "repository"
    root.mkdir()

    def fake_run(
        command: tuple[str, ...],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        output_index = command.index("--output") + 1
        output = Path(command[output_index])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("{}", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="completed\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    outputs = calibration._produce_observation_children(
        repository_root_path=root,
        receipts=(receipt,),
        run_dir=run_dir,
        host_envelope_path=host_path,
        start_permit=permit,
        warm_start_bundle_path=tmp_path / "warm-starts.json",
        benchmark_dir=tmp_path / "benchmarks",
        continuity_lease_token="lease-token",
    )

    assert len(outputs) == 3
    process_receipt = json.loads(
        (run_dir / "observation_processes/portable-o3/repeat1.json").read_text(encoding="utf-8")
    )
    command = cast(list[str], process_receipt["command"])
    assert command[command.index("--continuity-lease-token") + 1] == "<redacted>"
    assert command[command.index("--start-permit") + 1] == str(permit.permit_path.resolve())
    assert command[command.index("--repository-root") + 1] == str(root)


def test_representative_telemetry_parent_accepts_current_shared_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from evrptw.experiments import stage052_performance_calibration as calibration

    receipt = load_wheel_receipt(_receipts(tmp_path / "inputs", profiles=("portable-o3",))[0])
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    permit = _start_permit(tmp_path, run_label=run_dir.name)
    permit.permit_path.write_text("{}", encoding="utf-8")
    host_path = run_dir / "host.json"
    host_path.write_text("{}", encoding="utf-8")
    root = tmp_path / "repository"
    root.mkdir()

    def fake_run(
        command: tuple[str, ...],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        output = Path(command[command.index("--output") + 1])
        enabled = command[command.index("--telemetry-sample") + 1] == "on"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "schema_version": TELEMETRY_SAMPLE_SCHEMA_VERSION,
                    "enabled": enabled,
                    "fingerprint": "f" * 64,
                    "resource_summary": {},
                    "workload_evidence": {},
                    "elapsed_seconds": 1.0,
                    "replay_seconds": 0.1,
                    "raw_axis_inventory": [],
                }
            ),
            encoding="utf-8",
        )
        _sign(output)
        return subprocess.CompletedProcess(command, 0, stdout="completed\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    sample = calibration._representative_telemetry_runner(
        repository_root_path=root,
        receipt=receipt,
        run_dir=run_dir,
        host_envelope_path=host_path,
        start_permit=permit,
        warm_start_bundle_path=tmp_path / "warm-starts.json",
        benchmark_dir=tmp_path / "benchmarks",
        continuity_lease_token="lease-token",
    )(True, 0)

    assert sample.fingerprint == b"f" * 64
    assert sample.resource_summary["sample_relative_path"] == "telemetry_samples/sample-00-on.json"


def test_performance_calibration_cli_requires_repository_root(tmp_path: Path) -> None:
    from evrptw.experiments.stage052_performance_calibration import main

    with pytest.raises(SystemExit) as raised:
        main(
            [
                "--wheel-receipt",
                str(tmp_path / "wheel.json"),
                "--output-root",
                str(tmp_path / "results"),
                "--run-label",
                "stage05.2_native_architecture_performance_calibration_attempt01",
                "--warm-start-bundle",
                str(tmp_path / "warm.json"),
                "--benchmark-dir",
                str(tmp_path / "benchmarks"),
                "--continuity-lease-token",
                "lease",
            ]
        )
    assert raised.value.code == 2


def test_signed_input_and_telemetry_overhead_are_hard_gates(tmp_path: Path) -> None:
    receipt_paths = _receipts(tmp_path)
    Path(f"{receipt_paths[0]}.sha256").write_text("0" * 64 + "\n", encoding="ascii")
    with pytest.raises(CalibrationError, match="SHA-256 mismatch"):
        load_wheel_receipts(receipt_paths)

    observations_root = tmp_path / "observations"
    receipts = load_wheel_receipts(_receipts(observations_root))
    observations = load_fixed_work_observations(_write_observations(observations_root))
    fingerprint = "f" * 64
    unmonitored = (1.0, 1.0, 1.0, 1.0, 1.0)
    monitored = (1.03, 1.03, 1.03, 1.03, 1.03)
    failed = TelemetryOverheadReceipt(
        unmonitored_seconds=unmonitored,
        monitored_seconds=monitored,
        pair_orders=("off-on", "on-off", "off-on", "on-off", "off-on"),
        sample_interval_seconds=0.05,
        workload_output_sha256=hashlib.sha256(fingerprint.encode("ascii")).hexdigest(),
        monitored_resource_summaries=tuple({"sample_count": 10} for _ in range(5)),
        workload_evidence=_telemetry_evidence(
            fingerprint,
            unmonitored,
            monitored,
        ),
    )
    failed_path = tmp_path / "failed-overhead.json"
    write_telemetry_overhead_receipt(failed_path, failed)
    with pytest.raises(CalibrationError, match="overhead gate failed"):
        calibrate_performance_profile(
            run_label="stage05.2_native_architecture_performance_calibration_attempt06",
            wheel_receipts=receipts,
            observations=observations,
            host=_host(),
            telemetry_overhead=failed,
            telemetry_overhead_path=failed_path,
        )


def test_independent_review_rederives_the_exact_frozen_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from evrptw.experiments import stage052_performance_calibration_review as review
    from evrptw.experiments.stage052_performance_calibration import (
        _terminal_manifest,
        write_calibration_bundle,
    )

    run_label = "stage05.2_native_architecture_performance_calibration_attempt08"
    receipts = load_wheel_receipts(_receipts(tmp_path / "inputs"))
    observations = load_fixed_work_observations(_write_observations(tmp_path / "observations"))
    overhead, overhead_path = _overhead(tmp_path)
    result = calibrate_performance_profile(
        run_label=run_label,
        wheel_receipts=receipts,
        observations=observations,
        host=_host(),
        telemetry_overhead=overhead,
        telemetry_overhead_path=overhead_path,
    )
    outputs = write_calibration_bundle(result, output_root=tmp_path / "results")
    _terminal_manifest(
        output_root=tmp_path / "results",
        run_label=run_label,
        status="complete",
        receipt_path=outputs["receipt"],
    )
    monkeypatch.setattr(
        review,
        "_replay_observation_children",
        lambda *_args, **_kwargs: (72, 30, 0),
    )
    monkeypatch.setattr(
        review,
        "_replay_telemetry_children",
        lambda *_args, **_kwargs: 12,
    )

    payload = review.derive_calibration_review(
        calibration_run_dir=outputs["run_dir"],
        benchmark_dir=tmp_path / "benchmarks",
    )

    assert payload["qualification"] == review.CALIBRATION_REVIEW_QUALIFICATION
    assert payload["profile_canonical_sha256"] == result.profile.canonical_sha256
    assert payload["rederived_profile_canonical_sha256"] == result.profile.canonical_sha256
    assert payload["raw_axis_replay_count"] == 72


def test_calibration_bundle_replays_after_directory_relocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from evrptw.experiments import stage052_performance_calibration_review as review
    from evrptw.experiments.stage052_performance_calibration import (
        _terminal_manifest,
        write_calibration_bundle,
    )

    run_label = "stage05.2_native_architecture_performance_calibration_attempt13"
    input_root = tmp_path / "external-inputs"
    receipts = load_wheel_receipts(_receipts(input_root / "builds"))
    observations = load_fixed_work_observations(_write_observations(input_root / "observations"))
    overhead, overhead_path = _overhead(input_root)
    result = calibrate_performance_profile(
        run_label=run_label,
        wheel_receipts=receipts,
        observations=observations,
        host=_host(),
        telemetry_overhead=overhead,
        telemetry_overhead_path=overhead_path,
    )
    outputs = write_calibration_bundle(result, output_root=tmp_path / "original")
    _terminal_manifest(
        output_root=tmp_path / "original",
        run_label=run_label,
        status="complete",
        receipt_path=outputs["receipt"],
    )
    signed_receipt = outputs["receipt"].read_text(encoding="utf-8")
    assert "operational_input_locations" not in signed_receipt
    assert str(tmp_path) not in signed_receipt

    relocated = tmp_path / "relocated" / run_label
    relocated.parent.mkdir()
    shutil.copytree(outputs["run_dir"], relocated)
    shutil.rmtree(input_root)
    shutil.rmtree(outputs["run_dir"])
    monkeypatch.setattr(
        review,
        "_replay_observation_children",
        lambda *_args, **_kwargs: (72, 30, 0),
    )
    monkeypatch.setattr(
        review,
        "_replay_telemetry_children",
        lambda *_args, **_kwargs: 12,
    )

    payload = review.derive_calibration_review(
        calibration_run_dir=relocated,
        benchmark_dir=tmp_path / "benchmarks",
    )

    assert payload["qualification"] == review.CALIBRATION_REVIEW_QUALIFICATION
    assert payload["rederived_profile_canonical_sha256"] == result.profile.canonical_sha256


def test_independent_review_rejects_tampered_calibration_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from evrptw.experiments import stage052_performance_calibration_review as review
    from evrptw.experiments.stage052_performance_calibration import (
        _terminal_manifest,
        write_calibration_bundle,
    )

    run_label = "stage05.2_native_architecture_performance_calibration_attempt09"
    receipts = load_wheel_receipts(_receipts(tmp_path / "inputs"))
    observations = load_fixed_work_observations(_write_observations(tmp_path / "observations"))
    overhead, overhead_path = _overhead(tmp_path)
    result = calibrate_performance_profile(
        run_label=run_label,
        wheel_receipts=receipts,
        observations=observations,
        host=_host(),
        telemetry_overhead=overhead,
        telemetry_overhead_path=overhead_path,
    )
    outputs = write_calibration_bundle(result, output_root=tmp_path / "results")
    _terminal_manifest(
        output_root=tmp_path / "results",
        run_label=run_label,
        status="complete",
        receipt_path=outputs["receipt"],
    )
    monkeypatch.setattr(
        review,
        "_replay_observation_children",
        lambda *_args, **_kwargs: (72, 30, 0),
    )
    outputs["receipt"].write_text("{}\n", encoding="utf-8")

    with pytest.raises(review.CalibrationReviewError, match="sidecar mismatch"):
        review.derive_calibration_review(
            calibration_run_dir=outputs["run_dir"],
            benchmark_dir=tmp_path / "benchmarks",
        )


def test_calibration_reviewer_routes_sealed_failure_capsule(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from evrptw.experiments import stage052_campaign_review as campaign_review
    from evrptw.experiments import stage052_performance_calibration_review as review

    raw_manifest = tmp_path / "control" / "failure.json"
    review_manifest = tmp_path / "review" / "review_manifest.json"
    repository = tmp_path / "repository"
    calls: list[tuple[Path, Path]] = []

    monkeypatch.setattr(review, "require_clean_repository_root", lambda _path: None)

    def route_failure(*, raw_manifest_path: Path, review_manifest_path: Path) -> dict[str, object]:
        calls.append((raw_manifest_path, review_manifest_path))
        return {"status": "FAILED_UNKNOWN", "run_label": "attempt"}

    monkeypatch.setattr(
        campaign_review,
        "review_lifecycle_failure_capsule",
        route_failure,
    )

    assert (
        review.main(
            [
                "--lifecycle-failure-manifest",
                str(raw_manifest),
                "--failure-review-manifest",
                str(review_manifest),
                "--repository-root",
                str(repository),
            ]
        )
        == 0
    )
    assert calls == [(raw_manifest, review_manifest)]
    assert json.loads(capsys.readouterr().out)["status"] == "FAILED_UNKNOWN"
