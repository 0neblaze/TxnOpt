from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "evrptw" / "stage052_performance.py"
_SPEC = importlib.util.spec_from_file_location("stage052_performance_under_test", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_PERF = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _PERF
_SPEC.loader.exec_module(_PERF)

BuildCandidate = _PERF.BuildCandidate
BuildArtifactIdentity = _PERF.BuildArtifactIdentity
CalibrationMeasurement = _PERF.CalibrationMeasurement
ExecutionTopology = _PERF.ExecutionTopology
FrozenPerformanceProfile = _PERF.FrozenPerformanceProfile
HostPerformanceEnvelope = _PERF.HostPerformanceEnvelope
RuntimeResourceSummaryV2 = _PERF.RuntimeResourceSummaryV2
TelemetryOverheadReceipt = _PERF.TelemetryOverheadReceipt
build_profile_candidates = _PERF.build_profile_candidates
check_memory_admission = _PERF.check_memory_admission
detect_host_performance = _PERF.detect_host_performance
generate_execution_topologies = _PERF.generate_execution_topologies
generate_mode_topology_candidates = _PERF.generate_mode_topology_candidates
execution_topology_id = _PERF.execution_topology_id
load_frozen_profile = _PERF.load_frozen_profile
partition_cpus = _PERF.partition_cpus
physical_core_first_cpu_order = _PERF.physical_core_first_cpu_order
require_clean_repository_root = _PERF.require_clean_repository_root
select_build_candidate = _PERF.select_build_candidate
performance_topology_key = _PERF.performance_topology_key
STAGE052_PERFORMANCE_MODES = _PERF.STAGE052_PERFORMANCE_MODES
STAGE052_WORKLOAD_CLASSES = _PERF.STAGE052_WORKLOAD_CLASSES


def test_explicit_repository_root_requires_exact_clean_git_root(tmp_path: Path) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    subprocess.run(("git", "init", "-q"), cwd=root, check=True)
    nested = root / "nested"
    nested.mkdir()
    (root / "tracked.txt").write_text("source\n", encoding="utf-8")
    (nested / "tracked.txt").write_text("nested source\n", encoding="utf-8")
    subprocess.run(("git", "add", "tracked.txt", "nested/tracked.txt"), cwd=root, check=True)
    subprocess.run(
        (
            "git",
            "-c",
            "user.name=Stage 5.2 Test",
            "-c",
            "user.email=stage052@example.invalid",
            "commit",
            "-q",
            "-m",
            "fixture",
        ),
        cwd=root,
        check=True,
    )

    assert require_clean_repository_root(root) == root.resolve()
    with pytest.raises(RuntimeError, match="worktree root"):
        require_clean_repository_root(nested)
    (root / "untracked.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="clean Git worktree"):
        require_clean_repository_root(root)


def _host(
    *,
    cpus: tuple[int, ...] = (0, 1, 2, 3, 4),
    available: int = 10 * 1024**3,
    swap_total: int = 0,
    swap_used: int = 0,
    swap_current: int | None = None,
) -> HostPerformanceEnvelope:
    return HostPerformanceEnvelope(
        allowed_cpu_ids=cpus,
        memory_total_bytes=16 * 1024**3,
        memory_available_bytes=available,
        swap_total_bytes=swap_total,
        swap_used_bytes=swap_used,
        swap_current_bytes=swap_current,
        topology_source="provided",
    )


def _digest(char: str) -> str:
    return char * 64


def _representative_evidence(
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
            "semantic_telemetry": True,
            "physical_telemetry": True,
            "persistence": True,
            "independent_replay": True,
            "resource_telemetry": False,
        },
        "monitored_telemetry_surface": {
            "semantic_telemetry": True,
            "physical_telemetry": True,
            "persistence": True,
            "independent_replay": True,
            "resource_telemetry": True,
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
                "semantic_telemetry": True,
                "physical_telemetry": True,
                "persistence": True,
                "independent_replay": True,
                "resource_telemetry": enabled,
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


def _runtime_profile() -> tuple[FrozenPerformanceProfile, dict[str, object]]:
    host = _host(cpus=(0, 1, 2, 3))
    identity = BuildArtifactIdentity(
        git_revision="e" * 40,
        git_tree="f" * 40,
        source_manifest_sha256=_digest("a"),
        wheel_sha256=_digest("b"),
        native_sha256=_digest("c"),
        scheduler_sha256=_digest("d"),
        compiler_version="gcc 14",
        flags=("-O3",),
        cpu_feature_mask=(),
    )
    build = BuildCandidate(
        "portable-o3",
        ("-O3",),
        True,
        False,
        False,
        compiler="gcc",
        artifact_identity=identity,
    )
    topologies: dict[str, ExecutionTopology] = {}
    for mode in STAGE052_PERFORMANCE_MODES:
        for workload_class in STAGE052_WORKLOAD_CLASSES:
            topologies[performance_topology_key(mode, workload_class)] = ExecutionTopology(
                workload_class=workload_class,
                shards=((0, 1), (2, 3)),
                worker_count=2,
                scheduler_cpu_ids=(0, 1, 2, 3) if mode == "host_scheduler" else (),
                request_threads=2,
                affinity_policy="free_scheduler" if mode == "host_scheduler" else "uniform",
                allow_affinity_overlap=mode == "host_scheduler",
            )
    telemetry_fingerprint = _digest("9")
    overhead = TelemetryOverheadReceipt(
        unmonitored_seconds=(1.0,) * 5,
        monitored_seconds=(1.01,) * 5,
        pair_orders=("off-on", "on-off", "off-on", "on-off", "off-on"),
        sample_interval_seconds=0.05,
        workload_output_sha256=hashlib.sha256(telemetry_fingerprint.encode("ascii")).hexdigest(),
        monitored_resource_summaries=tuple(
            {"sample_count": 20, "sample_interval_seconds": 0.05}
            for _ in range(5)
        ),
        workload_evidence=_representative_evidence(
            telemetry_fingerprint,
            (1.0,) * 5,
            (1.01,) * 5,
        ),
    )

    def lifecycle_receipt(*, faster: bool) -> dict[str, object]:
        return {
            "session_isolation_passed": True,
            "cache_reset_passed": True,
            "rss_stability_passed": True,
            "semantic_replay_passed": True,
            "mode_block_faster": faster,
            "per_wave_end_to_end_seconds_median": 2.0 if faster else 1.0,
            "mode_block_end_to_end_seconds_median": 1.0 if faster else 2.0,
            "per_wave_scheduler_pss_peak_bytes_max": 100,
            "mode_block_scheduler_pss_peak_bytes_max": 100,
            "sample_count": 6,
            "wave_count": 2,
            "qualified": faster,
        }

    entry = {
        "storage_alias": "calibration-inputs",
        "relative_path": "wheel-receipts/portable-o3/a.json",
        "sha256": _digest("1"),
    }
    calibration = {
        "memory_required_bytes_by_topology": {key: 1024 for key in topologies},
        "telemetry_overhead": overhead.to_dict(),
        "signed_input_inventory": {
            "wheel_receipts": [entry],
            "fixed_work_observations": [
                {
                    **entry,
                    "relative_path": "fixed-work-observations/portable-o3/repeat1/a.json",
                    "sha256": _digest("2"),
                }
            ],
            "telemetry_overhead": {
                **entry,
                "relative_path": "telemetry-overhead/a.json",
                "sha256": _digest("3"),
            },
        },
        "host_scheduler_lifecycle_by_workload": {
            "c5": "mode-block",
            "100-customer": "per-wave",
        },
        "host_scheduler_lifecycle_evidence": {
            "c5": lifecycle_receipt(faster=True),
            "100-customer": lifecycle_receipt(faster=False),
        },
    }
    profile = FrozenPerformanceProfile(
        host=host,
        selected_build=build,
        topologies=topologies,
        calibration=calibration,
        selection_reason="runtime binding test",
    )
    wheel_receipt: dict[str, object] = {
        "build_git_revision": identity.git_revision,
        "build_git_tree": identity.git_tree,
        "build_source_manifest_sha256": identity.source_manifest_sha256,
        "wheel_sha256": identity.wheel_sha256,
        "native_sha256": identity.native_sha256,
        "scheduler_sha256": identity.scheduler_sha256,
        "build_compiler_version": identity.compiler_version,
        "build_performance_profile": build.name,
        "build_compiler_id": build.compiler,
        "build_interprocedural_optimization": build.lto,
        "build_host_native": build.host_native,
    }
    return profile, wheel_receipt


def test_odd_cpu_partition_is_balanced_and_lossless() -> None:
    partitions = partition_cpus((0, 1, 2, 3, 4), 2)
    assert partitions == ((0, 1, 2), (3, 4))
    assert tuple(cpu for shard in partitions for cpu in shard) == (0, 1, 2, 3, 4)
    topologies = generate_execution_topologies((0, 1, 2, 3, 4))
    assert [topology.shard_count for topology in topologies] == [5, 3, 2]
    for topology in topologies:
        assert topology.cpu_ids == (0, 1, 2, 3, 4)
        assert len(set(topology.compute_cpu_ids)) == 5
    physical = generate_execution_topologies(
        (0, 1, 2, 3, 4),
        affinity_policy="physical_core_first",
        physical_core_groups=((0, 2), (1, 3), (4,)),
    )
    assert physical[0].shards == ((0,), (1,), (4,), (2,), (3,))
    explicit = generate_execution_topologies((0, 1, 2, 3, 4), cpu_order=(4, 2, 0, 3, 1))
    assert explicit[0].shards == ((4,), (2,), (0,), (3,), (1,))
    partitioned = generate_execution_topologies(
        (0, 1, 2, 3),
        scheduler_cpu_ids=(0,),
        affinity_policy="physical_core_first",
        physical_core_groups=((0, 2), (1, 3)),
    )
    assert partitioned[0].shards == ((2,), (1,), (3,))


def test_mode_topology_matrix_compares_free_pinned_and_scheduler_partitions() -> None:
    host = HostPerformanceEnvelope(
        allowed_cpu_ids=(0, 1, 2, 3, 4),
        physical_core_groups=((0, 3), (1, 4), (2,)),
        memory_total_bytes=16 * 1024**3,
        memory_available_bytes=10 * 1024**3,
        topology_source="provided",
    )
    assert physical_core_first_cpu_order(host) == (0, 1, 2, 3, 4)
    local = generate_mode_topology_candidates(
        host,
        mode="per_solve_runtime",
        workload_class="100-customer",
    )
    assert {item.affinity_policy for item in local} == {
        "free_scheduler",
        "physical_core_first",
    }
    assert {item.shard_count for item in local} == {2, 3, 5}
    assert len({execution_topology_id(item) for item in local}) == len(local)

    scheduler = generate_mode_topology_candidates(
        host,
        mode="host_scheduler",
        workload_class="c5",
    )
    shared = [item for item in scheduler if item.affinity_policy == "free_scheduler"]
    partitioned = [item for item in scheduler if item.affinity_policy == "scheduler_partition"]
    assert {item.shard_count for item in shared} == {2, 3, 5}
    assert all(item.allow_affinity_overlap for item in shared)
    assert all(item.scheduler_cpu_ids == host.allowed_cpu_ids for item in shared)
    assert {item.shard_count for item in partitioned} == {2, 3}
    assert all(
        set(item.compute_cpu_ids).isdisjoint(item.scheduler_cpu_ids)
        and item.cpu_ids == host.allowed_cpu_ids
        for item in partitioned
    )


@pytest.mark.parametrize("cpu_count", (65, 96))
def test_large_cpu_topologies_cap_request_threads_without_losing_compute_cpus(
    cpu_count: int,
) -> None:
    cpus = tuple(range(cpu_count))
    host = HostPerformanceEnvelope(
        allowed_cpu_ids=cpus,
        memory_total_bytes=512 * 1024**3,
        memory_available_bytes=400 * 1024**3,
        topology_source="provided",
    )
    candidates = generate_mode_topology_candidates(
        host,
        mode="host_scheduler",
        workload_class="100-customer",
    )
    assert candidates
    assert all(1 <= item.request_threads <= 64 for item in candidates)
    shared = [item for item in candidates if item.allow_affinity_overlap]
    assert any(item.shard_count == cpu_count for item in shared)
    assert all(item.scheduler_cpu_ids == cpus for item in shared)


def test_missing_physical_topology_falls_back_to_uniform_partitions(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    proc.mkdir()
    text = chr(10).join(
        (
            "MemTotal:       16384 kB",
            "MemAvailable:    8192 kB",
            "SwapTotal:          0 kB",
            "SwapFree:           0 kB",
        )
    )
    (proc / "meminfo").write_text(text, encoding="utf-8")
    envelope = detect_host_performance(
        allowed_cpu_ids=(1, 3, 5),
        platform_name="linux",
        architecture="x86_64",
        proc_root=proc,
        sysfs_root=tmp_path / "missing-sysfs",
        cgroup_root=tmp_path / "missing-cgroup",
    )
    assert envelope.topology_source == "unavailable"
    assert envelope.physical_core_count is None
    assert generate_execution_topologies(envelope.allowed_cpu_ids)[-1].cpu_ids == (1, 3, 5)


def test_available_power_frequency_and_temperature_are_diagnostic(
    tmp_path: Path,
) -> None:
    proc = tmp_path / "proc"
    proc.mkdir()
    (proc / "meminfo").write_text(
        "MemTotal:       16384 kB\nMemAvailable:    8192 kB\n"
        "SwapTotal:          0 kB\nSwapFree:           0 kB\n",
        encoding="utf-8",
    )
    sysfs = tmp_path / "sys"
    ac = sysfs / "class" / "power_supply" / "AC1"
    ac.mkdir(parents=True)
    (ac / "type").write_text("Mains\n", encoding="utf-8")
    (ac / "online").write_text("1\n", encoding="utf-8")
    frequency = sysfs / "devices" / "system" / "cpu" / "cpu0" / "cpufreq"
    frequency.mkdir(parents=True)
    (frequency / "scaling_cur_freq").write_text("3200000\n", encoding="utf-8")
    thermal = sysfs / "class" / "thermal" / "thermal_zone0"
    thermal.mkdir(parents=True)
    (thermal / "type").write_text("x86_pkg_temp\n", encoding="utf-8")
    (thermal / "temp").write_text("55000\n", encoding="utf-8")
    profile = sysfs / "firmware" / "acpi"
    profile.mkdir(parents=True)
    (profile / "platform_profile").write_text("performance\n", encoding="utf-8")

    envelope = detect_host_performance(
        allowed_cpu_ids=(0,),
        platform_name="linux",
        architecture="x86_64",
        proc_root=proc,
        sysfs_root=sysfs,
        cgroup_root=tmp_path / "missing-cgroup",
    )
    power = envelope.telemetry["power_state"]
    assert isinstance(power, Mapping)
    assert power["status"] == "online"
    assert envelope.telemetry["power_plan"] == "performance"
    assert envelope.telemetry["frequency"] == {
        "unit": "kHz",
        "sample_count": 1,
        "minimum": 3200000,
        "median": 3200000,
        "maximum": 3200000,
    }
    temperature = envelope.telemetry["temperature"]
    assert isinstance(temperature, Mapping)
    assert temperature["sensors"][0]["millidegree_celsius"] == 55000


def test_cgroup_memory_fields_are_independent_when_swap_current_is_missing(
    tmp_path: Path,
) -> None:
    proc = tmp_path / "proc"
    proc.mkdir()
    (proc / "meminfo").write_text(
        "MemTotal:       16384 kB\nMemAvailable:    8192 kB\n"
        "SwapTotal:       8192 kB\nSwapFree:        8192 kB\n",
        encoding="utf-8",
    )
    cgroup = tmp_path / "cgroup"
    cgroup.mkdir()
    (cgroup / "memory.max").write_text("8388608\n", encoding="utf-8")
    (cgroup / "memory.current").write_text("1048576\n", encoding="utf-8")
    (cgroup / "memory.swap.max").write_text("16777216\n", encoding="utf-8")
    envelope = detect_host_performance(
        allowed_cpu_ids=(0,),
        platform_name="linux",
        architecture="x86_64",
        proc_root=proc,
        sysfs_root=tmp_path / "missing-sysfs",
        cgroup_root=cgroup,
    )
    assert envelope.memory_limit_bytes == 8388608
    assert envelope.memory_current_bytes == 1048576
    assert envelope.swap_limit_bytes == 16777216
    assert envelope.swap_current_bytes is None


def test_nested_cgroup_v2_uses_current_membership_and_strictest_ancestor(
    tmp_path: Path,
) -> None:
    proc = tmp_path / "proc"
    self_proc = proc / "self"
    self_proc.mkdir(parents=True)
    (proc / "meminfo").write_text(
        "MemTotal:       65536 kB\nMemAvailable:   49152 kB\n"
        "SwapTotal:       8192 kB\nSwapFree:        8192 kB\n",
        encoding="utf-8",
    )
    (self_proc / "cgroup").write_text("0::/init.scope/worker\n", encoding="utf-8")
    cgroup = tmp_path / "cgroup"
    worker = cgroup / "init.scope" / "worker"
    worker.mkdir(parents=True)
    (self_proc / "mountinfo").write_text(
        f"1 0 0:1 / {cgroup} rw - cgroup2 cgroup rw\n",
        encoding="utf-8",
    )
    (cgroup / "memory.max").write_text("max\n", encoding="utf-8")
    (cgroup / "memory.current").write_text("1048576\n", encoding="utf-8")
    (cgroup / "memory.swap.current").write_text("0\n", encoding="utf-8")
    parent = cgroup / "init.scope"
    (parent / "memory.max").write_text("33554432\n", encoding="utf-8")
    (parent / "memory.current").write_text("16777216\n", encoding="utf-8")
    (parent / "memory.swap.max").write_text("8388608\n", encoding="utf-8")
    (parent / "memory.swap.current").write_text("0\n", encoding="utf-8")
    (worker / "memory.max").write_text("25165824\n", encoding="utf-8")
    (worker / "memory.current").write_text("4194304\n", encoding="utf-8")
    (worker / "memory.swap.max").write_text("max\n", encoding="utf-8")
    (worker / "memory.swap.current").write_text("0\n", encoding="utf-8")

    envelope = detect_host_performance(
        allowed_cpu_ids=(0,),
        platform_name="linux",
        architecture="x86_64",
        proc_root=proc,
        sysfs_root=tmp_path / "missing-sysfs",
        cgroup_root=cgroup,
    )

    assert envelope.memory_limit_bytes == 33554432
    assert envelope.memory_current_bytes == 16777216
    assert envelope.swap_limit_bytes == 8388608
    assert envelope.swap_current_bytes == 0
    assert envelope.telemetry["cgroup_v2_path"] == str(worker)
    assert envelope.telemetry["cgroup_v2_ancestor_count"] == 3


def test_memory_admission_requires_twenty_percent_headroom_and_zero_swap() -> None:
    passing = check_memory_admission(_host(available=12 * 1024**3), 10 * 1024**3)
    assert passing.passed
    assert passing.headroom_bytes == 2 * 1024**3
    insufficient = check_memory_admission(_host(available=11 * 1024**3), 10 * 1024**3)
    assert not insufficient.passed
    assert "below required" in insufficient.reason
    capacity_only = check_memory_admission(_host(available=12 * 1024**3, swap_total=1), 1)
    assert capacity_only.passed
    swapped = check_memory_admission(_host(available=12 * 1024**3, swap_total=1, swap_used=1), 1)
    assert not swapped.passed
    assert "swap" in swapped.reason
    cgroup_swapped = check_memory_admission(
        _host(available=12 * 1024**3, swap_total=1, swap_current=1), 1
    )
    assert not cgroup_swapped.passed


def test_runtime_binding_owns_live_host_build_and_calibration_validation() -> None:
    profile, wheel_receipt = _runtime_profile()
    binding = profile.bind_runtime(wheel_receipt, observed_host=profile.host)
    assert binding.profile_sha256 == profile.canonical_sha256
    assert binding.allowed_cpu_ids == (0, 1, 2, 3)
    assert binding.max_executor_workers == 2
    assert binding.scheduler_lifecycle_for("c5") == "mode-block"
    assert binding.scheduler_lifecycle_for("100-customer") == "per-wave"
    assert binding.topology_for("host_scheduler", "c5").scheduler_cpu_ids == (0, 1, 2, 3)
    receipt = binding.host_receipt()
    admission = receipt["memory_admission"]
    assert isinstance(admission, dict)
    assert all(row["passed"] is True for row in admission.values())

    changed_build = dict(wheel_receipt)
    changed_build["native_sha256"] = _digest("0")
    with pytest.raises(RuntimeError, match="native_sha256"):
        profile.bind_runtime(changed_build, observed_host=profile.host)

    changed_host = _host(cpus=(4, 5, 6, 7))
    with pytest.raises(RuntimeError, match="allowed CPU set"):
        profile.bind_runtime(wheel_receipt, observed_host=changed_host)


def test_build_tie_prefers_portable_lower_memory_candidate() -> None:
    candidates = build_profile_candidates(
        _host(),
        compiler="gcc (Ubuntu)",
        host_native_supported=True,
    )
    measurements = [
        CalibrationMeasurement("portable-o3", 9.2, 100, _digest("a"), (9.0, 9.4)),
        CalibrationMeasurement("portable-o3", 9.2, 100, _digest("a"), (9.0, 9.4)),
        CalibrationMeasurement("portable-lto", 9.1, 120, _digest("a"), (8.9, 9.3)),
        CalibrationMeasurement("portable-lto", 9.1, 120, _digest("a"), (8.9, 9.3)),
        CalibrationMeasurement("host-native-lto", 9.0, 200, _digest("a"), (8.9, 9.1)),
        CalibrationMeasurement("host-native-lto", 9.0, 200, _digest("a"), (8.9, 9.1)),
    ]
    assert select_build_candidate(candidates, measurements).name == "portable-o3"


def test_semantic_mismatch_removes_candidate() -> None:
    candidates = build_profile_candidates(host_native_supported=False)
    measurements = [
        CalibrationMeasurement("portable-o3", 1.0, 100, _digest("a")),
        CalibrationMeasurement("portable-lto", 0.1, 100, _digest("b")),
    ]
    assert select_build_candidate(candidates, measurements).name == "portable-o3"


def test_frozen_profile_round_trip_and_tamper_detection(tmp_path: Path) -> None:
    host = _host(cpus=(0, 1, 2, 3))
    build = BuildCandidate(
        "portable-o3",
        ("-O3",),
        True,
        False,
        False,
        artifact_identity=BuildArtifactIdentity(
            git_revision="e" * 40,
            git_tree="f" * 40,
            source_manifest_sha256=_digest("a"),
            wheel_sha256=_digest("b"),
            native_sha256=_digest("c"),
            scheduler_sha256=_digest("d"),
            compiler_version="gcc 14",
            flags=("-O3",),
            cpu_feature_mask=(),
        ),
    )
    topology = generate_execution_topologies(host.allowed_cpu_ids)[0]
    profile = FrozenPerformanceProfile(
        host=host,
        selected_build=build,
        topologies={"general": topology},
        calibration={"median_seconds": 1.0},
        selection_reason="test",
    )
    path = tmp_path / "profile.json"
    profile.save(path)
    with pytest.raises(FileExistsError):
        profile.save(path)
    loaded = load_frozen_profile(path)
    assert loaded.canonical_sha256 == profile.canonical_sha256
    assert loaded.to_dict() == profile.to_dict()
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["selection_reason"] = "tampered"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        FrozenPerformanceProfile.load(path)


def test_profile_hash_excludes_live_host_diagnostics() -> None:
    profile, _ = _runtime_profile()
    changed_host = HostPerformanceEnvelope(
        allowed_cpu_ids=profile.host.allowed_cpu_ids,
        physical_core_groups=profile.host.physical_core_groups,
        memory_total_bytes=profile.host.memory_total_bytes,
        memory_available_bytes=profile.host.memory_available_bytes // 2,
        memory_limit_bytes=profile.host.memory_limit_bytes,
        memory_current_bytes=4096,
        swap_total_bytes=profile.host.swap_total_bytes,
        swap_used_bytes=profile.host.swap_used_bytes,
        swap_current_bytes=profile.host.swap_current_bytes,
        swap_limit_bytes=profile.host.swap_limit_bytes,
        cpu_features=profile.host.cpu_features,
        compiler=profile.host.compiler,
        topology_source=profile.host.topology_source,
        telemetry={"frequency": {"median": 9999999}, "temperature": "unavailable"},
    )
    changed = FrozenPerformanceProfile(
        host=changed_host,
        selected_build=profile.selected_build,
        topologies=profile.topologies,
        calibration=profile.calibration,
        selection_reason=profile.selection_reason,
    )
    assert changed.canonical_sha256 == profile.canonical_sha256
    assert changed.to_dict()["host"] != profile.to_dict()["host"]


def test_profile_recursively_freezes_nested_json_inputs() -> None:
    profile, _ = _runtime_profile()
    telemetry: dict[str, object] = {"sensor": {"samples": [1, 2]}}
    calibration = cast(dict[str, object], profile.to_dict()["calibration"])
    calibration["nested"] = {"values": [1, 2]}
    host = HostPerformanceEnvelope(
        allowed_cpu_ids=profile.host.allowed_cpu_ids,
        memory_total_bytes=profile.host.memory_total_bytes,
        memory_available_bytes=profile.host.memory_available_bytes,
        topology_source="provided",
        telemetry=telemetry,
    )
    frozen = FrozenPerformanceProfile(
        host=host,
        selected_build=profile.selected_build,
        topologies=profile.topologies,
        calibration=calibration,
        selection_reason=profile.selection_reason,
    )
    before = frozen.to_dict()
    before_hash = frozen.canonical_sha256
    cast(dict[str, object], telemetry["sensor"])["samples"] = [9]
    cast(dict[str, object], calibration["nested"])["values"] = [9]
    assert frozen.to_dict() == before
    assert frozen.canonical_sha256 == before_hash
    with pytest.raises(TypeError):
        cast(dict[str, object], frozen.calibration)["new"] = "forbidden"


def test_build_identity_rejects_non_sha_git_identity() -> None:
    payload = BuildArtifactIdentity(
        git_revision="e" * 40,
        git_tree="f" * 40,
        source_manifest_sha256=_digest("a"),
        wheel_sha256=_digest("b"),
        native_sha256=_digest("c"),
        scheduler_sha256=_digest("d"),
        compiler_version="gcc 14",
        flags=("-O3",),
        cpu_feature_mask=(),
    ).to_dict()
    payload["git_revision"] = "not-a-git-sha"
    with pytest.raises(ValueError, match="40-character Git SHA-1"):
        BuildArtifactIdentity.from_dict(payload)
    with pytest.raises(ValueError, match="artifact flags must be an array"):
        BuildArtifactIdentity(
            git_revision="e" * 40,
            git_tree="f" * 40,
            source_manifest_sha256=_digest("a"),
            wheel_sha256=_digest("b"),
            native_sha256=_digest("c"),
            scheduler_sha256=_digest("d"),
            compiler_version="gcc 14",
            flags="-O3",
            cpu_feature_mask=(),
        )


def test_topology_rejects_overlap_and_runtime_summary_round_trips() -> None:
    with pytest.raises(ValueError, match="overlap"):
        ExecutionTopology(shards=((0, 1), (1, 2)), worker_count=2, request_threads=2)
    summary = RuntimeResourceSummaryV2(
        elapsed_seconds=2.0,
        effective_cores=3.5,
        cpu_utilization_fraction=1.75,
        rss_bytes=123,
        p95_end_to_end_seconds=1.2,
        max_end_to_end_seconds=1.4,
    )
    assert RuntimeResourceSummaryV2.from_dict(summary.to_dict()) == summary


def test_telemetry_overhead_receipt_enforces_alternating_two_percent_gate() -> None:
    fingerprint = _digest("a")
    unmonitored = (1.0, 1.0, 1.0, 1.0, 1.0)
    monitored = (1.01, 1.02, 1.0, 1.01, 1.02)
    receipt = TelemetryOverheadReceipt(
        unmonitored_seconds=unmonitored,
        monitored_seconds=monitored,
        pair_orders=("off-on", "on-off", "off-on", "on-off", "off-on"),
        sample_interval_seconds=0.05,
        workload_output_sha256=hashlib.sha256(fingerprint.encode("ascii")).hexdigest(),
        monitored_resource_summaries=tuple(
            {"sample_count": 20, "sample_interval_seconds": 0.05}
            for _ in range(5)
        ),
        workload_evidence=_representative_evidence(
            fingerprint,
            unmonitored,
            monitored,
        ),
    )
    assert receipt.passed
    assert receipt.median_overhead_fraction == pytest.approx(0.01)
    assert TelemetryOverheadReceipt.from_dict(receipt.to_dict()) == receipt

    failed_monitored = (1.03, 1.03, 1.03, 1.03, 1.03)
    failed = TelemetryOverheadReceipt(
        unmonitored_seconds=unmonitored,
        monitored_seconds=failed_monitored,
        pair_orders=("off-on", "on-off", "off-on", "on-off", "off-on"),
        sample_interval_seconds=0.05,
        workload_output_sha256=hashlib.sha256(fingerprint.encode("ascii")).hexdigest(),
        monitored_resource_summaries=tuple(
            {"sample_count": 20, "sample_interval_seconds": 0.05}
            for _ in range(5)
        ),
        workload_evidence=_representative_evidence(
            fingerprint,
            unmonitored,
            failed_monitored,
        ),
    )
    with pytest.raises(ValueError, match="exceeds"):
        failed.require_passed()
    tampered = receipt.to_dict()
    tampered["median_overhead_fraction"] = 0.0
    with pytest.raises(ValueError, match="derived field diverged"):
        TelemetryOverheadReceipt.from_dict(tampered)
