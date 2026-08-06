"""Five-mode Stage 5.2 native-architecture comparison runner."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import resource
import subprocess
import time
import zipfile
from collections.abc import Iterable, Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Protocol, TypedDict, cast
from urllib.parse import unquote, urlparse

import numpy as np

from evrptw.alns import ALNSResult, solve_alns
from evrptw.cache_incremental import CacheIncrementalConfig
from evrptw.candidate_control import CandidateControlConfig
from evrptw.candidate_transaction import NativeCandidateTransactionConfig
from evrptw.exact_deadline import ExactDeadlineConfig
from evrptw.experiments.stage02_route_reduction import FORMAL_INSTANCES
from evrptw.measurement import CheapScreeningConfig, MeasurementConfig
from evrptw.models import Instance, NodeType
from evrptw.native_execution import Stage052NativeExecutionConfig
from evrptw.native_kernels import NativeKernelConfig
from evrptw.native_scheduler import NativeHostScheduler
from evrptw.objective import SolutionObjective
from evrptw.parser import parse_schneider
from evrptw.repository import repository_root
from evrptw.runtime_envelope import ProcessTreeMonitor
from evrptw.stage04 import Stage04Config
from evrptw.stage052_continuity_lease import require_owned
from evrptw.validation import validate_routes
from evrptw.warm_start import (
    WarmStartValidationConfig,
    canonical_customer_sequences_sha256,
)
from tools.native_build_attestation import (
    committed_source_attestation,
    committed_wheel_project_entry_sha256,
    validate_scheduler_build_attestation,
)

SCHEMA_VERSION = "stage05.2-native-architecture-comparison-v6"
SEEDS = (2014, 2015, 2016)
PAIRED_INSTANCES = ("c101C5", "c101_21", "r101_21", "rc101_21")
AXIS_NAMES = ("fixed_work", "wall_clock_30")
SHARD_PROCESSES = 6
THREADS_PER_SHARD = 4
TOTAL_COMPUTE_THREADS = 24
WARM_START_SCHEMA_VERSION = "stage05.2-native-architecture-warm-start-v2"
NATIVE_ARCHITECTURE_CAPABILITY_NAMES = (
    "host_candidate_transaction_scheduler",
    "whole_search_gil_released",
    "single_host_24_thread_compute_pool",
    "runtime_semantic_event_journal",
)

WarmStartIdentity = tuple[str, int]
WarmStartRecord = tuple[tuple[tuple[str, ...], ...], dict[str, object]]


class NativeBuildAttestation(Protocol):
    __build_git_revision__: str
    __build_git_tree__: str
    __build_source_manifest_sha256__: str
    __build_tracked_file_count__: int
    __build_source_dirty__: bool
    __build_development_override__: bool
    __build_cpp_source_kind__: str
    __build_source_attestation_version__: int


class WheelReceipt(TypedDict):
    wheel_path: str
    wheel_sha256: str
    direct_url_path: str
    package_path: str
    native_path: str
    native_sha256: str
    scheduler_path: str
    scheduler_sha256: str
    build_git_revision: str
    build_git_tree: str
    build_source_manifest_sha256: str
    build_tracked_file_count: int
    build_source_dirty: bool
    build_development_override: bool
    build_cpp_source_kind: str
    build_source_attestation_version: int
    wheel_entry_sha256: dict[str, str]
    native_wheel_entry: str
    scheduler_wheel_entry: str
    runner_wheel_entry: str
    scheduler_build_attestation: dict[str, object]


class ArchitectureMode(StrEnum):
    CURRENT_STAGE052 = "current_stage052"
    PYTHON_CANDIDATE_CONTROL = "python_candidate_control"
    PER_SOLVE_RUNTIME = "per_solve_runtime"
    FULL_NATIVE_ALNS = "full_native_alns"
    HOST_SCHEDULER = "host_scheduler"


MODES = tuple(ArchitectureMode)


@dataclass(frozen=True, slots=True)
class ArchitectureAxisTask:
    scope: str
    repeat: int
    axis: str
    instance_name: str
    seed: int
    benchmark_dir: Path
    output_root: Path
    run_labels: dict[str, str]
    scheduler_socket_path: str
    wheel_sha256: str
    native_sha256: str
    scheduler_sha256: str
    revision: str
    initial_customer_sequences: tuple[tuple[str, ...], ...]
    initial_solution_provenance: dict[str, object]
    scheduler_process_id: int | None = None


def run_labels_for_scope(scope: str, attempt: int) -> dict[str, str]:
    if scope not in {"paired", "pilot"} or attempt <= 0:
        raise ValueError("native architecture scope/attempt is invalid")
    return {
        mode.value: (
            f"stage05.2_native_architecture_{mode.value}_{scope}_attempt{attempt:02d}"
        )
        for mode in MODES
    }


def build_axis_plan(
    scope: str,
    *,
    attempt: int,
    benchmark_dir: Path,
    output_root: Path,
    scheduler_socket_path: str,
    wheel_sha256: str,
    native_sha256: str,
    scheduler_sha256: str,
    revision: str,
    warm_starts: dict[WarmStartIdentity, WarmStartRecord],
) -> tuple[ArchitectureAxisTask, ...]:
    instances: tuple[str, ...]
    axes: tuple[str, ...]
    if scope == "paired":
        instances = PAIRED_INSTANCES
        repeats = range(3)
        axes = AXIS_NAMES
    elif scope == "pilot":
        instances = tuple(FORMAL_INSTANCES)
        repeats = range(1)
        axes = ("wall_clock_30",)
    else:
        raise ValueError("scope must be paired or pilot")
    labels = run_labels_for_scope(scope, attempt)
    expected_identities = {
        (instance_name, seed)
        for instance_name in instances
        for seed in SEEDS
    }
    if set(warm_starts) != expected_identities:
        missing = sorted(expected_identities - set(warm_starts))
        extra = sorted(set(warm_starts) - expected_identities)
        raise ValueError(
            f"warm-start identity set mismatch: missing={missing}, extra={extra}"
        )
    return tuple(
        ArchitectureAxisTask(
            scope=scope,
            repeat=repeat,
            axis=axis,
            instance_name=instance_name,
            seed=seed,
            benchmark_dir=benchmark_dir,
            output_root=output_root,
            run_labels=labels,
            scheduler_socket_path=scheduler_socket_path,
            wheel_sha256=wheel_sha256,
            native_sha256=native_sha256,
            scheduler_sha256=scheduler_sha256,
            revision=revision,
            initial_customer_sequences=warm_starts[(instance_name, seed)][0],
            initial_solution_provenance=dict(warm_starts[(instance_name, seed)][1]),
        )
        for repeat in repeats
        for axis in axes
        for instance_name in instances
        for seed in SEEDS
    )


def rotated_modes(task: ArchitectureAxisTask) -> tuple[ArchitectureMode, ...]:
    instance_order = (
        PAIRED_INSTANCES
        if task.scope == "paired"
        else tuple(FORMAL_INSTANCES)
    )
    axis_order = AXIS_NAMES if task.scope == "paired" else ("wall_clock_30",)
    rotation = (
        task.repeat * len(axis_order) * len(instance_order) * len(SEEDS)
        + axis_order.index(task.axis) * len(instance_order) * len(SEEDS)
        + instance_order.index(task.instance_name) * len(SEEDS)
        + SEEDS.index(task.seed)
    ) % len(MODES)
    return MODES[rotation:] + MODES[:rotation]


def expected_axis_count(scope: str) -> int:
    if scope == "paired":
        return len(PAIRED_INSTANCES) * len(SEEDS) * 3 * len(AXIS_NAMES) * len(MODES)
    if scope == "pilot":
        return len(FORMAL_INSTANCES) * len(SEEDS) * len(MODES)
    raise ValueError("scope must be paired or pilot")


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_warm_start_bundle(
    path: Path,
    *,
    benchmark_dir: Path,
) -> dict[WarmStartIdentity, WarmStartRecord]:
    """Verify and decode one immutable cross-mode warm-start input bundle."""

    if not path.is_file():
        raise FileNotFoundError(f"warm-start bundle does not exist: {path}")
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file():
        raise RuntimeError("warm-start bundle SHA-256 sidecar is missing")
    bundle_sha256 = _sha256_path(path)
    expected_sha256 = sidecar.read_text(encoding="ascii").strip()
    if expected_sha256 != bundle_sha256:
        raise RuntimeError("warm-start bundle SHA-256 mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != (
        WARM_START_SCHEMA_VERSION
    ):
        raise RuntimeError("warm-start bundle schema is invalid")
    raw_records = payload.get("records")
    if not isinstance(raw_records, list) or not raw_records:
        raise RuntimeError("warm-start bundle records are missing")
    output: dict[WarmStartIdentity, WarmStartRecord] = {}
    instance_cache: dict[str, Instance] = {}
    for raw in raw_records:
        if not isinstance(raw, dict):
            raise RuntimeError("warm-start bundle contains a non-object record")
        instance_name = raw.get("instance")
        seed = raw.get("seed")
        routes = raw.get("customer_sequences")
        if (
            not isinstance(instance_name, str)
            or isinstance(seed, bool)
            or not isinstance(seed, int)
            or not isinstance(routes, list)
            or not routes
        ):
            raise RuntimeError("warm-start bundle record identity/routes are invalid")
        identity = (instance_name, seed)
        if identity in output:
            raise RuntimeError(f"duplicate warm-start identity: {identity}")
        instance = instance_cache.get(instance_name)
        if instance is None:
            instance = parse_schneider(benchmark_dir / f"{instance_name}.txt")
            instance_cache[instance_name] = instance
        sequences: list[tuple[str, ...]] = []
        for route in routes:
            if not isinstance(route, list) or not route or not all(
                isinstance(name, str) for name in route
            ):
                raise RuntimeError(f"warm-start route is invalid for {identity}")
            sequences.append(tuple(cast(str, name) for name in route))
        expected_customers = sorted(customer.name for customer in instance.customers)
        supplied_customers = sorted(name for route in sequences for name in route)
        if supplied_customers != expected_customers:
            raise RuntimeError(f"warm-start customer coverage mismatch for {identity}")
        source_sha256 = raw.get("source_solution_sha256")
        if (
            not isinstance(source_sha256, str)
            or len(source_sha256) != 64
            or any(character not in "0123456789abcdef" for character in source_sha256)
        ):
            raise RuntimeError(f"warm-start source hash is invalid for {identity}")
        source_path_value = raw.get("source_solution_path")
        source_axis = raw.get("source_axis")
        if (
            not isinstance(source_path_value, str)
            or not source_path_value
            or not isinstance(source_axis, str)
            or not source_axis
        ):
            raise RuntimeError(f"warm-start source path/axis is invalid for {identity}")
        source_path = Path(source_path_value)
        if not source_path.is_absolute():
            source_path = path.parent / source_path
        source_path = source_path.resolve()
        if not source_path.is_file() or _sha256_path(source_path) != source_sha256:
            raise RuntimeError(f"warm-start source solution hash mismatch for {identity}")
        try:
            source_payload = json.loads(source_path.read_bytes())
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"warm-start source solution is unreadable for {identity}"
            ) from error
        source_axes = source_payload.get("axes") if isinstance(source_payload, dict) else None
        source_record = source_axes.get(source_axis) if isinstance(source_axes, dict) else None
        if not isinstance(source_record, dict):
            raise RuntimeError(f"warm-start source axis is missing for {identity}")
        source_routes = source_record.get("routes")
        source_objective = source_record.get("objective_key")
        if not isinstance(source_routes, list) or not all(
            isinstance(route, list) and all(isinstance(name, str) for name in route)
            for route in source_routes
        ):
            raise RuntimeError(f"warm-start source routes are invalid for {identity}")
        source_sequences = tuple(
            tuple(
                cast(str, name)
                for name in route
                if cast(str, name) in instance.by_name
                and instance.by_name[cast(str, name)].kind is NodeType.CUSTOMER
            )
            for route in source_routes
        )
        if source_sequences != tuple(sequences):
            raise RuntimeError(f"warm-start routes diverge from source for {identity}")
        source_report = validate_routes(instance, source_routes)
        if not source_report.feasible:
            raise RuntimeError(f"warm-start source routes are infeasible for {identity}")
        recomputed_objective = list(
            SolutionObjective.from_report(instance, source_report).key
        )
        if source_objective != recomputed_objective:
            raise RuntimeError(f"warm-start source objective mismatch for {identity}")
        declared_objective = raw.get("source_objective_key")
        if declared_objective != recomputed_objective:
            raise RuntimeError(f"warm-start declared objective mismatch for {identity}")
        sequence_sha256 = canonical_customer_sequences_sha256(tuple(sequences))
        output[identity] = (
            tuple(sequences),
            {
                "source_stage": raw.get("source_stage", "comparison_warm_start"),
                "source_run_label": raw.get("source_run_label", ""),
                "source_instance": instance_name,
                "source_seed": seed,
                "source_solution_sha256": source_sha256,
                "source_solution_path": str(source_path),
                "source_axis": source_axis,
                "source_customer_sequences_sha256": sequence_sha256,
                "source_objective_key": recomputed_objective,
                "warm_start_bundle_sha256": bundle_sha256,
                "warm_start_bundle_path": str(path.resolve()),
            },
        )
    return output


def _validate_native_build_attestation(
    native_core: NativeBuildAttestation,
    *,
    expected_revision: str,
    expected_tree: str,
    expected_source_manifest_sha256: str,
    expected_tracked_file_count: int,
) -> None:
    def is_lower_hex(value: object, length: int) -> bool:
        return (
            isinstance(value, str)
            and len(value) == length
            and all(character in "0123456789abcdef" for character in value)
        )

    try:
        build_revision = native_core.__build_git_revision__
        build_tree = native_core.__build_git_tree__
        source_manifest_sha256 = native_core.__build_source_manifest_sha256__
        tracked_file_count = native_core.__build_tracked_file_count__
        source_dirty = native_core.__build_source_dirty__
        development_override = native_core.__build_development_override__
        cpp_source_kind = native_core.__build_cpp_source_kind__
        attestation_version = native_core.__build_source_attestation_version__
    except AttributeError as error:
        raise RuntimeError("installed native wheel lacks source attestation") from error
    if not is_lower_hex(build_revision, 40) or build_revision != expected_revision:
        raise RuntimeError("installed native wheel was not built from the recorded revision")
    if not is_lower_hex(build_tree, 40) or build_tree != expected_tree:
        raise RuntimeError("installed native wheel Git tree does not match the checkout")
    if type(attestation_version) is not int or attestation_version != 1:
        raise RuntimeError("installed native wheel has an unknown source attestation")
    if (
        not is_lower_hex(source_manifest_sha256, 64)
        or source_manifest_sha256 != expected_source_manifest_sha256
    ):
        raise RuntimeError("installed native wheel source manifest does not match Git")
    if type(tracked_file_count) is not int or tracked_file_count <= 0:
        raise RuntimeError("installed native wheel has an invalid tracked-file count")
    if tracked_file_count != expected_tracked_file_count:
        raise RuntimeError("installed native wheel tracked-file count does not match Git")
    if type(source_dirty) is not bool:
        raise RuntimeError("installed native wheel has an invalid dirty-source flag")
    if source_dirty:
        raise RuntimeError("installed native wheel was built from a dirty source tree")
    if type(development_override) is not bool:
        raise RuntimeError("installed native wheel has an invalid development override")
    if development_override:
        raise RuntimeError("installed native wheel used the development build override")
    if cpp_source_kind != "git_blob_snapshot":
        raise RuntimeError("installed native wheel did not use a Git-blob C++ snapshot")


def _verify_installed_project_files(
    wheel_path: Path,
    *,
    site_packages: Path,
    required_entry_sha256: Mapping[str, str] | None = None,
    expected_source_entries: Mapping[str, str] | None = None,
    generated_entries: set[str] | None = None,
) -> dict[str, str]:
    wheel_entry_sha256: dict[str, str] = {}
    with zipfile.ZipFile(wheel_path) as archive:
        project_entries = tuple(
            entry
            for entry in archive.infolist()
            if not entry.is_dir()
            and entry.filename.startswith(("evrptw/", "tools/"))
        )
        if not project_entries:
            raise RuntimeError("supplied wheel contains no project files")
        for entry in project_entries:
            entry_path = PurePosixPath(entry.filename)
            if (
                entry_path.is_absolute()
                or ".." in entry_path.parts
                or "\\" in entry.filename
                or entry.filename in wheel_entry_sha256
            ):
                raise RuntimeError("supplied wheel has an unsafe project entry")
            installed_path = site_packages / entry.filename
            if not installed_path.is_file():
                raise RuntimeError(
                    f"installed wheel file is missing: {entry.filename}"
                )
            expected_sha256 = hashlib.sha256(archive.read(entry)).hexdigest()
            if _sha256_path(installed_path) != expected_sha256:
                raise RuntimeError(
                    f"installed wheel file differs from its archive: {entry.filename}"
                )
            wheel_entry_sha256[entry.filename] = expected_sha256
    installed_project_files = {
        str(path.relative_to(site_packages)).replace(os.sep, "/")
        for package_name in ("evrptw", "tools")
        for path in (site_packages / package_name).rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    unexpected_files = sorted(installed_project_files - set(wheel_entry_sha256))
    if unexpected_files:
        raise RuntimeError(
            "installed project files are absent from the supplied wheel: "
            + ", ".join(unexpected_files[:8])
        )
    if expected_source_entries is not None:
        source_entries = {
            entry
            for entry in wheel_entry_sha256
            if entry not in (generated_entries or set())
        }
        expected_source_entry_hashes = dict(expected_source_entries)
        if source_entries != set(expected_source_entry_hashes) or any(
            wheel_entry_sha256[entry] != expected_sha256
            for entry, expected_sha256 in expected_source_entry_hashes.items()
        ):
            raise RuntimeError("wheel project source inventory does not match Git")
    for required_entry, expected_sha256 in (required_entry_sha256 or {}).items():
        if wheel_entry_sha256.get(required_entry) != expected_sha256:
            raise RuntimeError(
                f"required wheel entry is not hash-bound: {required_entry}"
            )
    return dict(sorted(wheel_entry_sha256.items()))


def _verify_installed_wheel(
    wheel_path: Path,
    *,
    expected_revision: str,
    expected_tree: str,
    expected_source_manifest_sha256: str,
    expected_tracked_file_count: int,
    expected_source_entries: Mapping[str, str],
) -> WheelReceipt:
    """Prove that the executing distribution was installed from the supplied wheel."""

    resolved_wheel = wheel_path.resolve()
    wheel_sha256 = _sha256_path(resolved_wheel)
    distribution = importlib.metadata.distribution("reproducible-evrptw")
    direct_url_entry = next(
        (
            entry
            for entry in distribution.files or ()
            if str(entry).endswith(".dist-info/direct_url.json")
        ),
        None,
    )
    if direct_url_entry is None:
        raise RuntimeError("installed distribution has no direct_url.json wheel receipt")
    direct_url_path = Path(str(distribution.locate_file(direct_url_entry)))
    if not direct_url_path.is_file():
        raise RuntimeError("installed distribution has no direct_url.json wheel receipt")
    direct_url = json.loads(direct_url_path.read_text(encoding="utf-8"))
    if not isinstance(direct_url, dict):
        raise RuntimeError("installed wheel receipt has an invalid schema")
    archive_info = direct_url.get("archive_info")
    source_url = direct_url.get("url")
    if not isinstance(archive_info, dict) or not isinstance(source_url, str):
        raise RuntimeError("executing distribution is not a non-editable wheel install")
    parsed = urlparse(source_url)
    installed_source = Path(unquote(parsed.path)).resolve()
    if parsed.scheme != "file" or installed_source != resolved_wheel:
        raise RuntimeError("executing distribution was installed from a different wheel")
    receipt_hash = archive_info.get("hash")
    expected_receipt_hash = f"sha256={wheel_sha256}"
    if receipt_hash != expected_receipt_hash:
        raise RuntimeError("installed wheel receipt SHA-256 does not match supplied wheel")
    import evrptw
    from evrptw import _core as native_core

    site_packages = direct_url_path.parent.parent.resolve()
    package_path = Path(str(evrptw.__file__)).resolve()
    native_path = Path(str(native_core.__file__)).resolve()
    scheduler_path = native_path.with_name("_native_host_scheduler")
    if (
        not package_path.is_relative_to(site_packages)
        or not native_path.is_relative_to(site_packages)
        or not scheduler_path.is_relative_to(site_packages)
    ):
        raise RuntimeError("comparison runner imported source outside the installed wheel")
    if not scheduler_path.is_file() or not os.access(scheduler_path, os.X_OK):
        raise RuntimeError("installed wheel has no executable native host scheduler")
    _validate_native_build_attestation(
        native_core,
        expected_revision=expected_revision,
        expected_tree=expected_tree,
        expected_source_manifest_sha256=expected_source_manifest_sha256,
        expected_tracked_file_count=expected_tracked_file_count,
    )
    native_sha256 = _sha256_path(native_path)
    scheduler_sha256 = _sha256_path(scheduler_path)
    runner_path = Path(__file__).resolve()
    if not runner_path.is_relative_to(site_packages):
        raise RuntimeError("comparison runner was imported outside the installed wheel")
    native_wheel_entry = native_path.relative_to(site_packages).as_posix()
    scheduler_wheel_entry = scheduler_path.relative_to(site_packages).as_posix()
    runner_wheel_entry = runner_path.relative_to(site_packages).as_posix()
    wheel_entry_sha256 = _verify_installed_project_files(
        resolved_wheel,
        site_packages=site_packages,
        required_entry_sha256={
            native_wheel_entry: native_sha256,
            scheduler_wheel_entry: scheduler_sha256,
            runner_wheel_entry: _sha256_path(runner_path),
        },
        expected_source_entries=expected_source_entries,
        generated_entries={native_wheel_entry, scheduler_wheel_entry},
    )
    scheduler_attestation_raw = subprocess.run(
        [str(scheduler_path), "--build-attestation"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    try:
        scheduler_attestation = json.loads(scheduler_attestation_raw)
    except json.JSONDecodeError as error:
        raise RuntimeError("native scheduler build attestation is invalid") from error
    if not isinstance(scheduler_attestation, dict):
        raise RuntimeError("native scheduler build attestation is invalid")
    validate_scheduler_build_attestation(
        scheduler_attestation,
        revision=native_core.__build_git_revision__,
        git_tree=native_core.__build_git_tree__,
        source_manifest_sha256=native_core.__build_source_manifest_sha256__,
        tracked_file_count=native_core.__build_tracked_file_count__,
    )
    return {
        "wheel_path": str(resolved_wheel),
        "wheel_sha256": wheel_sha256,
        "direct_url_path": str(direct_url_path.resolve()),
        "package_path": str(package_path),
        "native_path": str(native_path),
        "native_sha256": native_sha256,
        "scheduler_path": str(scheduler_path),
        "scheduler_sha256": scheduler_sha256,
        "build_git_revision": native_core.__build_git_revision__,
        "build_git_tree": native_core.__build_git_tree__,
        "build_source_manifest_sha256": (
            native_core.__build_source_manifest_sha256__
        ),
        "build_tracked_file_count": native_core.__build_tracked_file_count__,
        "build_source_dirty": False,
        "build_development_override": False,
        "build_cpp_source_kind": native_core.__build_cpp_source_kind__,
        "build_source_attestation_version": 1,
        "wheel_entry_sha256": wheel_entry_sha256,
        "native_wheel_entry": native_wheel_entry,
        "scheduler_wheel_entry": scheduler_wheel_entry,
        "runner_wheel_entry": runner_wheel_entry,
        "scheduler_build_attestation": dict(scheduler_attestation),
    }


def _require_native_architecture_capabilities() -> dict[str, bool]:
    """Fail before any run label exists when the native design is incomplete."""

    from evrptw import _core as native_core

    raw = native_core.stage052_native_architecture_capabilities_v2()
    if (
        not isinstance(raw, np.ndarray)
        or raw.dtype != np.dtype(np.int64)
        or raw.shape != (len(NATIVE_ARCHITECTURE_CAPABILITY_NAMES),)
        or not raw.flags.c_contiguous
        or np.any((raw != 0) & (raw != 1))
    ):
        raise RuntimeError("native architecture capability receipt is invalid")
    capabilities = {
        name: bool(raw[index])
        for index, name in enumerate(NATIVE_ARCHITECTURE_CAPABILITY_NAMES)
    }
    missing = [name for name, available in capabilities.items() if not available]
    if missing:
        raise RuntimeError(
            "native architecture campaign is blocked by incomplete capabilities: "
            + ", ".join(missing)
        )
    return capabilities


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _write_signed_json(path: Path, payload: object) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _canonical_bytes(payload) + b"\n"
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_bytes(data)
    temporary.replace(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    sidecar.write_text(hashlib.sha256(data).hexdigest() + "\n", encoding="ascii")
    return len(data) + sidecar.stat().st_size


def _set_artifact_size(payload: dict[str, object]) -> None:
    payload["artifact_bytes"] = 0
    for _ in range(4):
        size = len(_canonical_bytes(payload)) + 1 + 65
        if payload["artifact_bytes"] == size:
            return
        payload["artifact_bytes"] = size
    raise RuntimeError("artifact byte count did not converge")


def _native_config(
    mode: ArchitectureMode,
    *,
    scheduler_socket_path: str | None = None,
) -> Stage052NativeExecutionConfig:
    if mode not in {
        ArchitectureMode.PER_SOLVE_RUNTIME,
        ArchitectureMode.FULL_NATIVE_ALNS,
        ArchitectureMode.HOST_SCHEDULER,
    }:
        raise ValueError("mode does not use the explicit native execution protocol")
    return Stage052NativeExecutionConfig(
        mode=mode.value,  # type: ignore[arg-type]
        native_kernel_config=NativeKernelConfig(),
        candidate_transaction_config=NativeCandidateTransactionConfig(),
        candidate_control_config=CandidateControlConfig(worker_count=THREADS_PER_SHARD),
        shard_processes=SHARD_PROCESSES,
        compute_threads_per_shard=THREADS_PER_SHARD,
        scheduler_socket_path=scheduler_socket_path,
    )


def _thread_count() -> int:
    return len(tuple((Path("/proc") / str(os.getpid()) / "task").iterdir()))


def _rss_bytes() -> int:
    fields = Path("/proc/self/statm").read_text(encoding="ascii").split()
    if len(fields) < 2:
        raise RuntimeError("cannot read process RSS from /proc/self/statm")
    return int(fields[1]) * os.sysconf("SC_PAGE_SIZE")


def _metric_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return value


def _row_evidence(rows: Iterable[object]) -> dict[str, object]:
    digest = hashlib.sha256(b"stage05.2-row-evidence-v1\0")
    count = 0
    for row in rows:
        encoded = _canonical_bytes(_evidence_json_value(row))
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        count += 1
    return {"count": count, "sha256": digest.hexdigest()}


def _evidence_json_value(value: object) -> object:
    if isinstance(value, np.generic):
        return _evidence_json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return {"nonfinite_float": "nan"}
        return {"nonfinite_float": "positive_inf" if value > 0.0 else "negative_inf"}
    if isinstance(value, dict):
        return {str(key): _evidence_json_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_evidence_json_value(item) for item in value]
    return value


def _canonical_trace_event(event: dict[str, object]) -> dict[str, object]:
    """Remove wall-clock telemetry from one replayable semantic event."""

    canonical = {
        key: value
        for key, value in event.items()
        if key not in {"timestamp_seconds", "duration_seconds"}
    }
    if canonical.get("event_type") in {
        "candidate_cache_transaction",
        "candidate_screening_aggregate",
    }:
        return {}
    if canonical.get("event_type") in {
        "deadline_boundary",
        "exact_budget_boundary",
        "candidate_control_boundary",
    }:
        if canonical.get("termination_boundary") is not True:
            return {}
        canonical.pop("termination_boundary", None)
    if canonical.get("event_type") == "screening_decision":
        raw_checks = canonical.get("checks", [])
        if isinstance(raw_checks, list):
            normalized_checks: list[dict[str, object]] = []
            boolean_checks = {
                "route_structure",
                "single_segment_battery_reachability",
            }
            for raw_check in raw_checks:
                if not isinstance(raw_check, dict):
                    continue
                check_name = str(raw_check.get("check", ""))
                check_value = raw_check.get("value")
                normalized_checks.append(
                    {
                        "check": check_name,
                        "status": raw_check.get("status"),
                        "value": (
                            bool(check_value)
                            if check_name in boolean_checks
                            else check_value
                        ),
                    }
                )
            canonical["checks"] = normalized_checks
        canonical = {
            key: value
            for key, value in canonical.items()
            if key
            in {
                "event_type",
                "route_key",
                "lane",
                "iteration",
                "operator",
                "status",
                "first_failed_check",
                "reason",
                "checks",
                "demand",
                "min_time_window_slack",
                "distance_lower_bound",
                "distance_increment_lower_bound",
                "single_segment_reachable",
                "structural_energy_lower_bound",
                "negative_cache_hit",
                "exact_call_blocked",
                "semantic_event_id",
            }
        }
    if canonical.get("event_type") == "candidate_control_budget":
        canonical.pop("accounting", None)
        context = canonical.get("context")
        if isinstance(context, str) and context.endswith(":native_candidate_round"):
            canonical["context"] = context.removesuffix(
                ":native_candidate_round"
            ) + ":candidate_pool"
    if canonical.get("event_type") == "exact_batch_started":
        canonical.pop("transaction_sha256", None)
    if canonical.get("event_type") == "exact_route_result":
        canonical.pop("evaluation_id", None)
    if canonical.get("event_type") == "candidate_plan_decision":
        canonical.pop("batch_ordinal", None)
        canonical.pop("transaction_status_code", None)
    if canonical.get("event_type") == "cache_event":
        operation = canonical.get("operation")
        if operation in {"lookup", "store", "evict", "reconcile", "oversize_not_cached"}:
            return {}
        if operation in {"hit", "candidate_pending_hit", "miss"}:
            canonical = {
                key: value
                for key, value in canonical.items()
                if key
                in {
                    "event_type",
                    "route_key",
                    "lane",
                    "iteration",
                    "operator",
                    "semantic_event_id",
                }
            }
            canonical["event_type"] = "cache_lookup_result"
            canonical["status"] = "miss" if operation == "miss" else "hit"
    if canonical.get("event_type") == "parallel_batch":
        canonical["event_type"] = "candidate_batch_complete"
        canonical["status"] = "complete"
        for key in (
            "worker_count",
            "worker_protocol",
            "submission_order",
            "completion_order",
            "chunk_sizes",
            "completed_indices",
            "batch_ordinal",
        ):
            canonical.pop(key, None)
    if canonical.get("event_type") == "candidate_state":
        route_keys = canonical.get("candidate_route_keys")
        if not isinstance(route_keys, list | tuple) or not all(
            isinstance(key, str) for key in route_keys
        ):
            raise ValueError(
                "candidate_state requires candidate_route_keys as a string array"
            )
        full_route_keys = canonical.get("candidate_full_route_keys", [])
        if not isinstance(full_route_keys, list | tuple) or not all(
            isinstance(key, str) for key in full_route_keys
        ):
            raise ValueError(
                "candidate_full_route_keys must be a string array when present"
            )
        canonical = {
            key: canonical[key]
            for key in (
                "event_type",
                "lane",
                "iteration",
                "operator",
                "candidate_objective_key",
                "candidate_feasible",
                "accepted",
                "status",
                "reason",
                "semantic_event_id",
            )
            if key in canonical
        }
        canonical["candidate_route_keys"] = list(route_keys)
        canonical["candidate_full_route_keys"] = list(full_route_keys)
        identity = {
            "candidate_route_keys": list(route_keys),
            "candidate_full_route_keys": list(full_route_keys),
        }
        canonical["candidate_id"] = hashlib.sha256(
            _canonical_bytes(identity)
        ).hexdigest()
    return canonical


def _semantic_candidate_trajectory(result: ALNSResult) -> list[dict[str, object]]:
    quality_operators = {
        "relocate",
        "swap",
        "two_opt_star",
        "route_segment_destroy",
        "ejection_chain",
    }
    implementation_telemetry_statuses = {
        "pair_prefilter_rejected_aggregate",
        "prefilter_rejected_aggregate",
    }
    trajectory: list[dict[str, object]] = []
    for raw_event in result.neighborhood_events:
        if raw_event.get("status") in implementation_telemetry_statuses:
            # Prefilter aggregates describe implementation-specific work
            # avoided before the candidate transaction. They remain in raw
            # neighborhood evidence (pair pruning also has a dedicated native-
            # ablation stream), but are not a search decision and therefore
            # cannot shift canonical ordinals.
            continue
        ordinal = len(trajectory)
        event = {
            key: value
            for key, value in raw_event.items()
            if not str(key).startswith("_")
            and not (key == "aggregate_count" and value == 1)
            and not (key == "candidate_pool_hash" and not value)
        }
        operator = str(event.get("operator", ""))
        track = str(event.get("track", ""))
        lane = (
            "constraint_lane"
            if track == "constraint_lane"
            else "quality_shadow"
            if operator in quality_operators
            else "legacy"
        )
        identity = {
            "lane": lane,
            "iteration": event.get("iteration"),
            "operator": operator,
            "status": event.get("status"),
            "candidate_route_sequences": event.get(
                "candidate_route_sequences", ()
            ),
            "candidate_objective_key": event.get("candidate_objective_key", ()),
            "ordinal": ordinal,
        }
        trajectory.append(
            {
                **event,
                "lane": lane,
                "candidate_id": hashlib.sha256(
                    _canonical_bytes(identity)
                ).hexdigest(),
            }
        )
    return trajectory


def _canonical_event_rows(rows: Iterable[object]) -> list[dict[str, object]]:
    canonical_rows: list[dict[str, object]] = []
    for ordinal, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError("canonical semantic streams require event objects")
        canonical = _evidence_json_value(_canonical_trace_event(dict(row)))
        if not isinstance(canonical, dict):
            raise AssertionError("canonical event normalization lost object identity")
        canonical_rows.append({**canonical, "stream_ordinal": ordinal})
    return canonical_rows


def _canonical_semantic_streams(result: ALNSResult) -> dict[str, list[dict[str, object]]]:
    trace = result.measurement_trace
    if trace is None:
        raise ValueError("canonical semantic streams require a measurement trace")
    stream_names = (
        "candidate_state",
        "operator",
        "stage04",
        "candidate_transaction",
        "exact_work",
        "exact_result",
        "cache",
        "screening",
        "deadline",
        "termination",
        "native_failure",
    )
    streams: dict[str, list[dict[str, object]]] = {
        name: [] for name in stream_names
    }
    runtime_events = trace.runtime_semantic_events
    runtime_ids = [event.get("semantic_event_id") for event in runtime_events]
    if runtime_ids != list(range(1, len(runtime_events) + 1)):
        raise ValueError("runtime semantic trace IDs are not contiguous")
    semantic_event_id = 0
    for raw_event in runtime_events:
        stream_name = raw_event.get("semantic_stream")
        if not isinstance(stream_name, str) or stream_name not in streams:
            raise ValueError("runtime semantic event names an unknown stream")
        source_runtime_event_id = raw_event.get("runtime_causal_event_id")
        if (
            isinstance(source_runtime_event_id, bool)
            or not isinstance(source_runtime_event_id, int)
            or source_runtime_event_id != raw_event.get("semantic_event_id")
        ):
            raise ValueError(
                "runtime semantic event lost its causal source identity"
            )
        event = {
            key: value
            for key, value in raw_event.items()
            if key
            not in {
                "semantic_stream",
                "runtime_causal_event_id",
                "runtime_native_lane_id",
                "runtime_native_operator_id",
                "runtime_native_iteration",
                "runtime_native_transaction_id",
                "runtime_native_subject_id",
                "runtime_native_status_code",
                "runtime_native_flags",
            }
        }
        canonical = _evidence_json_value(_canonical_trace_event(event))
        if not isinstance(canonical, dict):
            raise AssertionError("canonical event normalization lost object identity")
        if not canonical:
            continue
        runtime_event_id = canonical.pop("semantic_event_id", None)
        if not isinstance(runtime_event_id, int) or isinstance(runtime_event_id, bool):
            raise ValueError("runtime semantic event lost its event ID")
        semantic_event_id += 1
        streams[stream_name].append(
            {
                **canonical,
                "runtime_event_id": runtime_event_id,
                "semantic_event_id": semantic_event_id,
                "stream_ordinal": len(streams[stream_name]),
            }
        )
    required_nonempty = {
        "stage04",
        "exact_work",
        "exact_result",
        "cache",
        "screening",
        "termination",
    }
    if result.effective_iterations > 0:
        required_nonempty.update({"candidate_state", "operator"})
    if result.candidate_control_statistics.get("enabled") is True:
        required_nonempty.add("candidate_transaction")
    missing = sorted(name for name in required_nonempty if not streams[name])
    if missing:
        raise ValueError(
            "runtime semantic journal is incomplete: " + ", ".join(missing)
        )
    termination = streams["termination"]
    if result.objective is None:
        raise ValueError("runtime semantic termination requires an objective")
    if (
        len(termination) != 1
        or termination[0].get("event_type") != "termination"
        or termination[0].get("status") != result.termination_reason
        or termination[0].get("iterations") != result.iterations
        or termination[0].get("effective_iterations")
        != result.effective_iterations
        or termination[0].get("exact_started_calls")
        != result.exact_started_calls
        or termination[0].get("exact_completed_calls")
        != result.exact_completed_calls
        or termination[0].get("exact_interrupted_calls")
        != result.exact_interrupted_calls
        or termination[0].get("objective_key") != list(result.objective.key)
        or termination[0].get("semantic_event_id") != semantic_event_id
    ):
        raise ValueError("runtime semantic termination does not reconcile")
    if result.termination_reason != "iteration_limit" and not streams["deadline"]:
        raise ValueError("runtime semantic deadline boundary is missing")
    return streams


def _canonical_semantic_event_sequence(
    streams: dict[str, list[dict[str, object]]],
) -> list[dict[str, object]]:
    """Merge runtime-stamped journals without inventing cross-stream order."""

    expected_streams = {
        "candidate_state",
        "operator",
        "stage04",
        "candidate_transaction",
        "exact_work",
        "exact_result",
        "cache",
        "screening",
        "deadline",
        "termination",
        "native_failure",
    }
    if set(streams) != expected_streams:
        raise ValueError("canonical semantic stream set is incomplete")
    if not any(streams.values()):
        raise ValueError("canonical semantic runtime journal cannot be empty")
    events: list[dict[str, object]] = []
    for stream_name, rows in streams.items():
        previous_event_id = 0
        for row in rows:
            event_id = row.get("semantic_event_id")
            if (
                isinstance(event_id, bool)
                or not isinstance(event_id, int)
                or event_id <= previous_event_id
            ):
                raise ValueError(
                    f"canonical semantic stream {stream_name} lacks ordered runtime event IDs"
                )
            previous_event_id = event_id
            events.append({**row, "semantic_stream": stream_name})
    def runtime_event_id(event: dict[str, object]) -> int:
        value = event["semantic_event_id"]
        if isinstance(value, bool) or not isinstance(value, int):
            raise AssertionError("validated runtime event ID changed type")
        return value

    events.sort(key=runtime_event_id)
    event_ids = [runtime_event_id(event) for event in events]
    if event_ids != list(range(1, len(events) + 1)):
        raise ValueError(
            "canonical semantic runtime event IDs must be unique and contiguous"
        )
    return [
        {**event, "semantic_sequence": sequence}
        for sequence, event in enumerate(events)
    ]


def _semantic_operator_statistics(result: ALNSResult) -> dict[str, dict[str, object]]:
    telemetry_fields = {
        "failure_reasons",
        "prefilter_passed",
        "prefilter_rejected",
    }
    return {
        operator: {
            key: value
            for key, value in statistics.items()
            if key not in telemetry_fields
        }
        for operator, statistics in sorted(result.neighborhood_statistics.items())
    }


def _measurement_evidence(result: ALNSResult) -> dict[str, object]:
    trace = result.measurement_trace
    if trace is None:
        semantic = {
            "present": False,
            "exact_route_order": _row_evidence(()),
            "cache_lifecycle": _row_evidence(()),
            "deadline_boundaries": _row_evidence(()),
        }
    else:
        exact_route_order = tuple(
            {
                "batch_ordinal": ordinal,
                "lane": event.get("lane"),
                "iteration": event.get("iteration"),
                "operator": event.get("operator"),
                "sequences": event.get("sequences"),
            }
            for ordinal, event in enumerate(result.candidate_work_events)
        )
        semantic_cache_rows: list[dict[str, object]] = []
        for batch_ordinal, event in enumerate(result.candidate_work_events):
            for route_ordinal, sequence in enumerate(
                cast(list[list[str]], event.get("sequences", []))
            ):
                semantic_cache_rows.append(
                    {
                        "batch_ordinal": batch_ordinal,
                        "route_ordinal": route_ordinal,
                        "lane": event.get("lane"),
                        "iteration": event.get("iteration"),
                        "operator": event.get("operator"),
                        "customer_sequence": sequence,
                        "transition": "exact_miss_to_committed_store",
                    }
                )
        cache_statistics = result.cache_incremental_statistics
        semantic_cache_rows.append(
            {
                "transition": "final_cache_state",
                **{
                    field: cache_statistics.get(field, 0)
                    for field in (
                        "cache_stores",
                        "cache_evictions",
                        "cache_oversize_not_cached",
                        "entries_current",
                        "entries_peak",
                        "bytes_current",
                        "bytes_peak",
                        "unique_route_evaluations",
                    )
                },
            }
        )
        canonical_routes = sorted(
            {
                tuple(sequence)
                for event in result.candidate_work_events
                for sequence in cast(list[list[str]], event.get("sequences", []))
            }
        )
        deadline_boundaries = (
            {
                "evaluation_id": row.evaluation_id,
                "route_key": row.route_key,
                "deadline_boundary": row.deadline_boundary,
                "exact_started": row.exact_started,
                "exact_completed": row.exact_completed,
                "status": row.status,
            }
            for row in trace.route_evaluations
            if row.deadline_boundary
        )
        semantic = {
            "present": True,
            "exact_route_order": _row_evidence(exact_route_order),
            "exact_route_results": _row_evidence(result.route_result_events),
            "cache_lifecycle": _row_evidence(semantic_cache_rows),
            "deadline_boundaries": _row_evidence(deadline_boundaries),
            "route_dictionary": _row_evidence(
                {"route": list(route)} for route in canonical_routes
            ),
            # Only cross-adapter semantics participate in the transaction
            # digest. Native screening/pruning and incremental telemetry are
            # audited below but are allowed to use different implementations.
            "events": _row_evidence(_semantic_candidate_trajectory(result)),
            "candidate_trajectory": _row_evidence(
                _semantic_candidate_trajectory(result)
            ),
        }
        semantic["native_telemetry"] = {
            "screening_decisions": _row_evidence(
                asdict(row) for row in trace.screening_decisions
            ),
            "incremental_propagations": _row_evidence(
                dict(row) for row in trace.incremental_propagations
            ),
        }
    digest_payload = {
        key: value for key, value in semantic.items() if key != "native_telemetry"
    }
    semantic["sha256"] = hashlib.sha256(_canonical_bytes(digest_payload)).hexdigest()
    return semantic


def _solve_mode(
    mode: ArchitectureMode,
    task: ArchitectureAxisTask,
) -> tuple[ALNSResult, float, dict[str, object]]:
    instance = replace(
        parse_schneider(task.benchmark_dir / f"{task.instance_name}.txt"),
        distance_backend="native",
    )
    fixed_work = task.axis == "fixed_work"
    exact_deadline = (
        ExactDeadlineConfig.fixed_exact_calls(100, watchdog_seconds=120.0)
        if fixed_work
        else ExactDeadlineConfig.wall_clock()
    )
    threads_before = _thread_count()
    common: dict[str, object] = {
        "seed": task.seed,
        "max_iterations": 1000,
        "time_limit_seconds": 120.0 if fixed_work else 30.0,
        "operator_profile": "stage02_constraint_guided",
        "measurement_config": MeasurementConfig(
            record_runtime_semantic_events=True,
        ),
        "screening_config": CheapScreeningConfig(),
        "cache_incremental_config": CacheIncrementalConfig(enabled=True),
        "backend": "cpu_batch",
        "batch_size": 128,
        "termination_mode": "fixed_work" if fixed_work else "wall_clock",
        "exact_deadline_config": exact_deadline,
        "stage04_config": Stage04Config(),
        "initial_customer_sequences": task.initial_customer_sequences,
        "initial_solution_provenance": task.initial_solution_provenance,
        "warm_start_validation_config": WarmStartValidationConfig(),
    }
    scheduler_roots = (
        (task.scheduler_process_id,)
        if mode is ArchitectureMode.HOST_SCHEDULER
        and task.scheduler_process_id is not None
        else ()
    )
    # Per-axis telemetry includes the shared scheduler so an individual raw
    # bundle never omits part of its execution process tree.  These shared-root
    # values must not be summed across concurrent shard axes; the parent
    # mode-wave observation is the aggregate accounting source.
    with ProcessTreeMonitor(additional_root_pids=scheduler_roots) as resource_monitor:
        started = time.perf_counter()
        if mode is ArchitectureMode.CURRENT_STAGE052:
            result = solve_alns(
                instance,
                **common,  # type: ignore[arg-type]
                native_kernel_config=NativeKernelConfig(),
                candidate_transaction_config=NativeCandidateTransactionConfig(),
            )
        elif mode is ArchitectureMode.PYTHON_CANDIDATE_CONTROL:
            result = solve_alns(
                instance,
                **common,  # type: ignore[arg-type]
                candidate_control_config=CandidateControlConfig(
                    worker_count=THREADS_PER_SHARD
                ),
            )
        else:
            result = solve_alns(
                instance,
                **common,  # type: ignore[arg-type]
                native_execution_config=_native_config(
                    mode,
                    scheduler_socket_path=(
                        task.scheduler_socket_path
                        if mode is ArchitectureMode.HOST_SCHEDULER
                        else None
                    ),
                ),
            )
        solver_seconds = time.perf_counter() - started
    report = validate_routes(instance, [list(route) for route in result.routes])
    if not report.feasible or result.objective is None:
        raise RuntimeError("architecture axis returned an invalid or objective-less solution")
    if result.objective.key != SolutionObjective.from_report(instance, report).key:
        raise RuntimeError("architecture axis objective does not replay")
    topology: dict[str, object] = {
        "shard_processes": SHARD_PROCESSES,
        "threads_per_shard": THREADS_PER_SHARD,
        "compute_thread_limit": TOTAL_COMPUTE_THREADS,
        "scheduler_threads": 24 if mode is ArchitectureMode.HOST_SCHEDULER else 0,
        "effective_native_search_threads": (
            24 if mode is ArchitectureMode.HOST_SCHEDULER else THREADS_PER_SHARD
        ),
        "shared_native_work_pool": result.native_execution_statistics.get(
            "shared_native_work_pool", False
        ),
        "process_id": os.getpid(),
        "scheduler_process_id": task.scheduler_process_id,
        "shared_scheduler_resource_attribution": (
            "mode_wave_primary_axis_values_overlap"
            if mode is ArchitectureMode.HOST_SCHEDULER
            else "not_applicable"
        ),
        "threads_before": threads_before,
        "threads_after": _thread_count(),
        "rss_bytes": _rss_bytes(),
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "shared_scheduler_accounting": (
            "mode_wave" if mode is ArchitectureMode.HOST_SCHEDULER else "none"
        ),
        **resource_monitor.statistics(
            elapsed_seconds=solver_seconds,
            compute_thread_limit=TOTAL_COMPUTE_THREADS,
        ),
    }
    return result, solver_seconds, topology


def _result_payload(
    task: ArchitectureAxisTask,
    mode: ArchitectureMode,
    result: ALNSResult,
    solver_seconds: float,
    topology: dict[str, object],
) -> dict[str, object]:
    assert result.objective is not None
    backend = result.backend_metrics
    screening = result.screening_statistics
    candidate_transactions = result.candidate_transaction_statistics
    measurement_evidence = _measurement_evidence(result)
    native_fallback = result.native_execution_statistics.get("fallback_count", 0)
    if isinstance(native_fallback, bool) or not isinstance(native_fallback, int):
        raise RuntimeError("native fallback evidence has an invalid schema")
    native_full_mode = mode in {
        ArchitectureMode.FULL_NATIVE_ALNS,
        ArchitectureMode.HOST_SCHEDULER,
    }
    candidate_control_complete = result.native_execution_statistics.get(
        "candidate_control_semantics_complete"
    )
    stage04_complete = result.native_execution_statistics.get(
        "stage04_semantics_complete"
    )
    instrumentation_complete = result.native_execution_statistics.get(
        "instrumentation_complete"
    )
    canonical_semantic_streams = _canonical_semantic_streams(result)
    canonical_semantic_events = _canonical_semantic_event_sequence(
        canonical_semantic_streams
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "run_label": task.run_labels[mode.value],
        "scope": task.scope,
        "repeat": task.repeat,
        "axis": task.axis,
        "mode": mode.value,
        "instance": task.instance_name,
        "seed": task.seed,
        "status": "completed",
        "revision": task.revision,
        "wheel_sha256": task.wheel_sha256,
        "native_sha256": task.native_sha256,
        "scheduler_sha256": task.scheduler_sha256,
        "solver_seconds": solver_seconds,
        "objective": list(result.objective.key),
        "routes": [list(route) for route in result.routes],
        "customer_sequences": [list(route) for route in result.customer_sequences],
        "validator_passed": True,
        "iterations": result.iterations,
        "effective_iterations": result.effective_iterations,
        "accepted_moves": result.accepted_moves,
        "rejected_moves": result.rejected_moves,
        "exact_started_calls": result.exact_started_calls,
        "exact_completed_calls": result.exact_completed_calls,
        "exact_interrupted_calls": result.exact_interrupted_calls,
        "candidate_work_hash": result.candidate_work_hash,
        "route_result_hash": result.route_result_hash,
        "fallback_count": native_fallback,
        "semantic_trajectory": _semantic_candidate_trajectory(result),
        "canonical_semantic_streams": canonical_semantic_streams,
        "canonical_semantic_events": canonical_semantic_events,
        "trajectory": _row_evidence(
            dict(event) for event in result.neighborhood_events
        ),
        "operator_statistics": result.neighborhood_statistics,
        "operator_semantic_statistics": _semantic_operator_statistics(result),
        "stage04_statistics": result.stage04_statistics,
        "stage04_events": _row_evidence(
            dict(event) for event in result.stage04_event_log
        ),
        "candidate_transaction_events": _row_evidence(
            dict(event) for event in result.candidate_transaction_events
        ),
        "candidate_control_statistics": result.candidate_control_statistics,
        "candidate_transaction_statistics": candidate_transactions,
        "native_execution_statistics": result.native_execution_statistics,
        "backend_metrics": backend,
        "screening_statistics": screening,
        "cache_incremental_statistics": result.cache_incremental_statistics,
        "measurement_evidence": measurement_evidence,
        "semantic_completeness": {
            "candidate_control": (
                candidate_control_complete is True if native_full_mode else True
            ),
            "stage04": stage04_complete is True if native_full_mode else True,
            "measurement_trace": bool(measurement_evidence["present"])
            and (instrumentation_complete is True if native_full_mode else True),
        },
        "topology": topology,
        "throughput": {
            "effective_iterations_per_second": result.effective_iterations
            / max(solver_seconds, 1e-12),
            "exact_started_per_second": result.exact_started_calls
            / max(solver_seconds, 1e-12),
            "candidate_transactions_per_second": (
                _metric_int(candidate_transactions.get("transactions", 0))
                + _metric_int(
                    candidate_transactions.get("native_candidate_transactions", 0)
                )
            )
            / max(solver_seconds, 1e-12),
            "screened_routes_per_second": (
                _metric_int(screening.get("total_routes", 0))
                or _metric_int(screening.get("screening_calls", 0))
            )
            / max(solver_seconds, 1e-12),
        },
        "cache_memory_bytes": _metric_int(
            result.cache_incremental_statistics.get("bytes_peak", 0)
        ),
    }


def _axis_path(task: ArchitectureAxisTask, mode: ArchitectureMode) -> Path:
    return (
        task.output_root
        / task.run_labels[mode.value]
        / "axes"
        / f"repeat{task.repeat + 1}"
        / task.axis
        / task.instance_name
        / f"{task.seed}.json"
    )


def _run_mode(task: ArchitectureAxisTask, mode: ArchitectureMode) -> str:
    path = _axis_path(task, mode)
    try:
        result, solver_seconds, topology = _solve_mode(mode, task)
        payload = _result_payload(task, mode, result, solver_seconds, topology)
    except BaseException as error:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "run_label": task.run_labels[mode.value],
            "scope": task.scope,
            "repeat": task.repeat,
            "axis": task.axis,
            "mode": mode.value,
            "instance": task.instance_name,
            "seed": task.seed,
            "status": "failed",
            "revision": task.revision,
            "wheel_sha256": task.wheel_sha256,
            "native_sha256": task.native_sha256,
            "scheduler_sha256": task.scheduler_sha256,
            "error_type": type(error).__name__,
            "error": str(error),
        }
    # Persistence is measured with a same-directory probe after solving;
    # solver time must never leak into this field. The final signed artifact
    # records the probe duration, avoiding a self-referential rewrite loop.
    payload["persistence_seconds"] = 0.0
    _set_artifact_size(payload)
    probe_path = path.with_suffix(path.suffix + f".persistence-probe-{os.getpid()}")
    probe_sidecar = probe_path.with_suffix(probe_path.suffix + ".sha256")
    persistence_started = time.perf_counter()
    try:
        _write_signed_json(probe_path, payload)
        payload["persistence_seconds"] = time.perf_counter() - persistence_started
    finally:
        probe_path.unlink(missing_ok=True)
        probe_sidecar.unlink(missing_ok=True)
    _set_artifact_size(payload)
    observed_bytes = _write_signed_json(path, payload)
    if observed_bytes != payload["artifact_bytes"]:
        raise RuntimeError("artifact byte count does not reconcile")
    return str(path)


def _run_group(task: ArchitectureAxisTask) -> list[str]:
    """Run one legacy test group; production campaigns use global mode waves."""

    return [_run_mode(task, mode) for mode in rotated_modes(task)]


def _mode_wave_batches(
    plan: tuple[ArchitectureAxisTask, ...],
) -> tuple[tuple[ArchitectureAxisTask, ...], ...]:
    return tuple(
        tuple(plan[offset : offset + SHARD_PROCESSES])
        for offset in range(0, len(plan), SHARD_PROCESSES)
    )


def _configure_compute_envelope() -> dict[str, object]:
    available = sorted(os.sched_getaffinity(0))
    if len(available) < TOTAL_COMPUTE_THREADS:
        raise RuntimeError("Stage 5.2 comparison host exposes fewer than 24 logical CPUs")
    selected = available[:TOTAL_COMPUTE_THREADS]
    os.sched_setaffinity(0, selected)
    thread_environment = {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    os.environ.update(thread_environment)
    return {
        "available_logical_cpus": available,
        "selected_logical_cpus": selected,
        "thread_environment": thread_environment,
    }


def _require_campaign_identity(
    root: Path,
    *,
    continuity_lease_token: str,
    expected_revision: str,
) -> None:
    """Revalidate the single writer and frozen checkout around every mode wave."""

    require_owned(
        root,
        token=continuity_lease_token,
        allowed_phases=frozenset({"paired-campaign", "pilot-campaign"}),
    )
    observed_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if observed_revision != expected_revision:
        raise RuntimeError("native architecture campaign Git revision changed")
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise RuntimeError("native architecture campaign worktree changed")


def run_experiment(
    scope: str,
    *,
    attempt: int,
    output_root: Path,
    wheel_path: Path,
    warm_start_bundle_path: Path,
    continuity_lease_token: str,
    max_workers: int = SHARD_PROCESSES,
) -> dict[str, object]:
    root = repository_root()
    require_owned(
        root,
        token=continuity_lease_token,
        allowed_phases=frozenset({"paired-campaign", "pilot-campaign"}),
    )
    if max_workers != SHARD_PROCESSES:
        raise ValueError("Stage 5.2 comparison requires exactly six shard processes")
    compute_envelope = _configure_compute_envelope()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise RuntimeError("native architecture experiment requires a clean worktree")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    committed_attestation = committed_source_attestation(root, revision)
    expected_source_entries = committed_wheel_project_entry_sha256(root, revision)
    source_manifest_sha256 = cast(
        str,
        committed_attestation["source_manifest_sha256"],
    )
    tracked_file_count = cast(int, committed_attestation["tracked_file_count"])
    if not wheel_path.is_file():
        raise FileNotFoundError("the frozen comparison wheel does not exist")
    wheel_receipt = _verify_installed_wheel(
        wheel_path,
        expected_revision=revision,
        expected_tree=tree,
        expected_source_manifest_sha256=source_manifest_sha256,
        expected_tracked_file_count=tracked_file_count,
        expected_source_entries=expected_source_entries,
    )
    native_capabilities = _require_native_architecture_capabilities()
    native_path = Path(wheel_receipt["native_path"])
    warm_starts = load_warm_start_bundle(
        warm_start_bundle_path,
        benchmark_dir=root / "data" / "schneider",
    )
    labels = run_labels_for_scope(scope, attempt)
    for label in labels.values():
        if (output_root / label).exists():
            raise FileExistsError(f"run label already exists and cannot be reused: {label}")
    scheduler_path = output_root / f".native-scheduler-{scope}-attempt{attempt:02d}.sock"
    plan = build_axis_plan(
        scope,
        attempt=attempt,
        benchmark_dir=root / "data" / "schneider",
        output_root=output_root,
        scheduler_socket_path=str(scheduler_path),
        wheel_sha256=wheel_receipt["wheel_sha256"],
        native_sha256=wheel_receipt["native_sha256"],
        scheduler_sha256=wheel_receipt["scheduler_sha256"],
        revision=revision,
        warm_starts=warm_starts,
    )
    for label in labels.values():
        (output_root / label).mkdir(parents=True)
    started = time.time()
    written: list[str] = []
    scheduler_observations: list[dict[str, int]] = []
    mode_wave_resources: list[dict[str, object]] = []
    scheduler_startup_seconds = 0.0
    scheduler_shutdown_seconds = 0.0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for batch_index, batch in enumerate(_mode_wave_batches(plan)):
            offset = batch_index % len(MODES)
            mode_order = MODES[offset:] + MODES[:offset]
            for mode in mode_order:
                _require_campaign_identity(
                    root,
                    continuity_lease_token=continuity_lease_token,
                    expected_revision=revision,
                )
                scheduler: NativeHostScheduler | None = None
                scheduler_process_id: int | None = None
                wave_scheduler_startup = 0.0
                wave_scheduler_shutdown = 0.0
                if mode is ArchitectureMode.HOST_SCHEDULER:
                    scheduler = NativeHostScheduler(
                        scheduler_path, worker_threads=24
                    )
                    scheduler_start_started = time.perf_counter()
                    scheduler.start()
                    wave_scheduler_startup = (
                        time.perf_counter() - scheduler_start_started
                    )
                    scheduler_startup_seconds += wave_scheduler_startup
                    scheduler_process_id = scheduler.process_id
                    scheduler_observations.append(
                        {
                            "batch_index": batch_index,
                            "process_id": scheduler_process_id,
                            "observed_thread_count": (
                                scheduler.observed_thread_count()
                            ),
                            "configured_worker_threads": scheduler.worker_threads,
                        }
                    )
                try:
                    wave_started = time.perf_counter()
                    with ProcessTreeMonitor() as wave_monitor:
                        if scheduler_process_id is not None:
                            futures = [
                                executor.submit(
                                    _run_mode,
                                    replace(
                                        task,
                                        scheduler_process_id=scheduler_process_id,
                                    ),
                                    mode,
                                )
                                for task in batch
                            ]
                        else:
                            futures = [
                                executor.submit(_run_mode, task, mode)
                                for task in batch
                            ]
                        for future in as_completed(futures):
                            written.append(future.result())
                    _require_campaign_identity(
                        root,
                        continuity_lease_token=continuity_lease_token,
                        expected_revision=revision,
                    )
                    wave_elapsed = time.perf_counter() - wave_started
                finally:
                    if scheduler is not None:
                        scheduler_shutdown_started = time.perf_counter()
                        scheduler.close()
                        wave_scheduler_shutdown = (
                            time.perf_counter() - scheduler_shutdown_started
                        )
                        scheduler_shutdown_seconds += wave_scheduler_shutdown
                mode_wave_resources.append(
                    {
                        "batch_index": batch_index,
                        "mode": mode.value,
                        "axis_count": len(batch),
                        "scheduler_process_id": scheduler_process_id,
                        "scheduler_startup_seconds": wave_scheduler_startup,
                        "scheduler_shutdown_seconds": wave_scheduler_shutdown,
                        "identities": [
                            {
                                "repeat": task.repeat,
                                "axis": task.axis,
                                "instance": task.instance_name,
                                "seed": task.seed,
                            }
                            for task in batch
                        ],
                        "elapsed_seconds": wave_elapsed,
                        **wave_monitor.statistics(
                            elapsed_seconds=wave_elapsed,
                            compute_thread_limit=TOTAL_COMPUTE_THREADS,
                        ),
                    }
                )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "scope": scope,
        "attempt": attempt,
        "run_labels": labels,
        "revision": revision,
        "git_tree": tree,
        "wheel_path": wheel_receipt["wheel_path"],
        "wheel_sha256": wheel_receipt["wheel_sha256"],
        "wheel_receipt": wheel_receipt,
        "native_architecture_capabilities": native_capabilities,
        "native_path": str(native_path.resolve()),
        "native_sha256": wheel_receipt["native_sha256"],
        "scheduler_path": wheel_receipt["scheduler_path"],
        "scheduler_sha256": wheel_receipt["scheduler_sha256"],
        "warm_start_bundle_path": str(warm_start_bundle_path.resolve()),
        "warm_start_bundle_sha256": _sha256_path(warm_start_bundle_path),
        "axis_count": len(written),
        "expected_axis_count": expected_axis_count(scope),
        "started_unix": started,
        "completed_unix": time.time(),
        "topology": {
            "shard_processes": SHARD_PROCESSES,
            "threads_per_shard": THREADS_PER_SHARD,
            "host_scheduler_threads": 24,
            "compute_thread_limit": TOTAL_COMPUTE_THREADS,
            "compute_envelope": compute_envelope,
            "scheduler_observed": scheduler_observations,
            "scheduler_startup_seconds": scheduler_startup_seconds,
            "scheduler_shutdown_seconds": scheduler_shutdown_seconds,
            "mode_wave_resources": mode_wave_resources,
        },
        "mode_order_policy": "six_axis_global_mode_waves_rotated_by_batch",
        "formal_started": False,
    }
    if manifest["axis_count"] != manifest["expected_axis_count"]:
        raise RuntimeError("native architecture experiment axis count is incomplete")
    for label in labels.values():
        _write_signed_json(output_root / label / "run_manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scope", choices=("paired", "pilot"))
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--warm-start-bundle", type=Path, required=True)
    parser.add_argument("--continuity-lease-token", required=True)
    arguments = parser.parse_args(argv)
    manifest = run_experiment(
        arguments.scope,
        attempt=arguments.attempt,
        output_root=arguments.output_root,
        wheel_path=arguments.wheel,
        warm_start_bundle_path=arguments.warm_start_bundle,
        continuity_lease_token=arguments.continuity_lease_token,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "ArchitectureAxisTask",
    "ArchitectureMode",
    "AXIS_NAMES",
    "MODES",
    "PAIRED_INSTANCES",
    "SCHEMA_VERSION",
    "SEEDS",
    "build_axis_plan",
    "expected_axis_count",
    "load_warm_start_bundle",
    "NATIVE_ARCHITECTURE_CAPABILITY_NAMES",
    "rotated_modes",
    "run_experiment",
    "run_labels_for_scope",
)
