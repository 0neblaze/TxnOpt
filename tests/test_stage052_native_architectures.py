from __future__ import annotations

import hashlib
import json
import struct
import subprocess
import tracemalloc
import zipfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from evrptw.charging import solve_exact_charging
from evrptw.experiments.stage052_native_architecture_review import (
    ReviewRecord,
    _canonical_semantic_events,
    _common_prefix,
    _comparison_semantic_events,
    _describe_first_divergence,
    _load_axis_record,
    _raw_axis_inventory,
    _replay_initial_state_receipt,
    _replay_record,
    _review_build_attestation,
    _scheduler_screening_occupancy,
    _semantic_trajectory,
    load_records,
    render_report,
    review_records,
    write_review,
)
from evrptw.experiments.stage052_native_architectures import (
    MODES,
    PAIRED_INSTANCES,
    SCHEMA_VERSION,
    SEEDS,
    WARM_START_SCHEMA_VERSION,
    ArchitectureAxisTask,
    ArchitectureMode,
    _canonical_semantic_event_sequence,
    _canonical_trace_event,
    _require_campaign_identity,
    _require_native_architecture_capabilities,
    _run_group,
    _validate_native_build_attestation,
    _verify_installed_project_files,
    _write_signed_json,
    build_axis_plan,
    expected_axis_count,
    load_warm_start_bundle,
    rotated_modes,
    run_experiment,
    run_labels_for_scope,
)
from evrptw.native_scheduler import NativeHostScheduler
from evrptw.objective import SolutionObjective
from evrptw.parser import parse_schneider
from evrptw.validation import validate_routes
from evrptw.warm_start import canonical_customer_sequences_sha256
from tools.native_build_attestation import (
    committed_source_attestation,
    committed_wheel_project_entries,
    committed_wheel_project_entry_sha256,
)


def test_axis_loader_releases_large_replay_only_fields_between_records(
    tmp_path: Path,
) -> None:
    paths = []
    for index in range(6):
        path = tmp_path / f"large-axis-{index}.json"
        data = json.dumps(
            {
                "canonical_semantic_events": "x" * (12 * 1024 * 1024),
                "canonical_semantic_streams": "y" * (12 * 1024 * 1024),
                "status": "completed",
            },
            sort_keys=True,
        ).encode("utf-8")
        path.write_bytes(data)
        path.with_suffix(".json.sha256").write_text(
            hashlib.sha256(data).hexdigest() + "\n",
            encoding="ascii",
        )
        paths.append(path)
    del data

    tracemalloc.start()
    records = tuple(_load_axis_record(path) for path in paths)
    current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert all(record.payload == {"status": "completed"} for record in records)
    assert current_bytes < 4 * 1024 * 1024
    assert peak_bytes < 400 * 1024 * 1024


def test_signed_axis_hash_and_parse_share_one_byte_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "axis.json"
    data = json.dumps({"status": "completed"}, sort_keys=True).encode("utf-8")
    path.write_bytes(data)
    path.with_suffix(".json.sha256").write_text(
        hashlib.sha256(data).hexdigest() + "\n",
        encoding="ascii",
    )
    original_read_bytes = Path.read_bytes
    reads = 0

    def counted_read_bytes(selected: Path) -> bytes:
        nonlocal reads
        if selected == path:
            reads += 1
        return original_read_bytes(selected)

    monkeypatch.setattr(Path, "read_bytes", counted_read_bytes)

    record = _load_axis_record(path)

    assert record.payload == {"status": "completed"}
    assert reads == 1


def test_native_campaign_gate_accepts_complete_architecture_capabilities() -> None:
    assert _require_native_architecture_capabilities() == {
        "host_candidate_transaction_scheduler": True,
        "whole_search_gil_released": True,
        "single_host_24_thread_compute_pool": True,
        "runtime_semantic_event_journal": True,
    }


def test_native_capability_receipt_confirms_single_host_compute_pool() -> None:
    from evrptw import _core as native_core

    assert native_core.stage052_native_architecture_capabilities_v2().tolist() == [
        1,
        1,
        1,
        1,
    ]


def test_native_build_attestation_rejects_dirty_or_mismatched_source() -> None:
    clean = SimpleNamespace(
        __build_git_revision__="a" * 40,
        __build_git_tree__="b" * 40,
        __build_source_manifest_sha256__="c" * 64,
        __build_tracked_file_count__=878,
        __build_source_dirty__=False,
        __build_development_override__=False,
        __build_cpp_source_kind__="git_blob_snapshot",
        __build_source_attestation_version__=1,
    )
    assert _validate_native_build_attestation(
        clean,
        expected_revision="a" * 40,
        expected_tree="b" * 40,
        expected_source_manifest_sha256="c" * 64,
        expected_tracked_file_count=878,
    ) is None

    for field, value, message in (
        ("__build_source_dirty__", True, "dirty source tree"),
        ("__build_source_dirty__", 0, "invalid dirty-source flag"),
        ("__build_development_override__", True, "development build override"),
        ("__build_development_override__", 0, "invalid development override"),
        ("__build_cpp_source_kind__", "working_tree_override", "Git-blob C\\+\\+ snapshot"),
        ("__build_git_revision__", "d" * 40, "recorded revision"),
        ("__build_git_tree__", "d" * 40, "Git tree"),
        ("__build_source_attestation_version__", 2, "unknown source attestation"),
        ("__build_source_attestation_version__", True, "unknown source attestation"),
        ("__build_source_manifest_sha256__", "g" * 64, "does not match Git"),
        ("__build_tracked_file_count__", True, "invalid tracked-file count"),
    ):
        invalid = SimpleNamespace(**vars(clean))
        setattr(invalid, field, value)
        with pytest.raises(RuntimeError, match=message):
            _validate_native_build_attestation(
                invalid,
                expected_revision="a" * 40,
                expected_tree="b" * 40,
                expected_source_manifest_sha256="c" * 64,
                expected_tracked_file_count=878,
            )

    missing = SimpleNamespace(**vars(clean))
    del missing.__build_git_tree__
    with pytest.raises(RuntimeError, match="lacks source attestation"):
        _validate_native_build_attestation(
            missing,
            expected_revision="a" * 40,
            expected_tree="b" * 40,
            expected_source_manifest_sha256="c" * 64,
            expected_tracked_file_count=878,
        )


def test_wheel_receipt_rejects_installed_project_file_tampering(
    tmp_path: Path,
) -> None:
    wheel = tmp_path / "comparison.whl"
    site_packages = tmp_path / "site-packages"
    package = site_packages / "evrptw"
    package.mkdir(parents=True)
    installed = package / "runtime.py"
    installed.write_bytes(b"reviewed-runtime")
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("evrptw/runtime.py", b"reviewed-runtime")

    receipt = _verify_installed_project_files(wheel, site_packages=site_packages)
    assert receipt == {
        "evrptw/runtime.py": hashlib.sha256(b"reviewed-runtime").hexdigest()
    }

    installed.write_bytes(b"tampered-runtime")
    with pytest.raises(RuntimeError, match="differs from its archive"):
        _verify_installed_project_files(wheel, site_packages=site_packages)

    installed.write_bytes(b"reviewed-runtime")
    (package / "stale_runtime.py").write_bytes(b"stale-runtime")
    with pytest.raises(RuntimeError, match="absent from the supplied wheel"):
        _verify_installed_project_files(wheel, site_packages=site_packages)


def test_wheel_receipt_requires_hash_bound_entries_and_safe_paths(
    tmp_path: Path,
) -> None:
    wheel = tmp_path / "comparison.whl"
    site_packages = tmp_path / "site-packages"
    package = site_packages / "evrptw"
    package.mkdir(parents=True)
    installed = package / "runtime.py"
    installed.write_bytes(b"reviewed-runtime")
    runtime_sha256 = hashlib.sha256(b"reviewed-runtime").hexdigest()
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("evrptw/runtime.py", b"reviewed-runtime")

    with pytest.raises(RuntimeError, match="required wheel entry"):
        _verify_installed_project_files(
            wheel,
            site_packages=site_packages,
            required_entry_sha256={"evrptw/missing.py": runtime_sha256},
        )
    with pytest.raises(RuntimeError, match="source inventory does not match Git"):
        _verify_installed_project_files(
            wheel,
            site_packages=site_packages,
            expected_source_entries={"evrptw/other.py": runtime_sha256},
        )
    with pytest.raises(RuntimeError, match="source inventory does not match Git"):
        _verify_installed_project_files(
            wheel,
            site_packages=site_packages,
            expected_source_entries={"evrptw/runtime.py": "0" * 64},
        )

    unsafe_wheel = tmp_path / "unsafe.whl"
    with zipfile.ZipFile(unsafe_wheel, "w") as archive:
        archive.writestr("evrptw/../runtime.py", b"reviewed-runtime")
    with pytest.raises(RuntimeError, match="unsafe project entry"):
        _verify_installed_project_files(
            unsafe_wheel,
            site_packages=site_packages,
        )

    extra_core = package / "_core.extra.so"
    extra_core.write_bytes(b"unloaded-native")
    extra_core_wheel = tmp_path / "extra-core.whl"
    with zipfile.ZipFile(extra_core_wheel, "w") as archive:
        archive.writestr("evrptw/runtime.py", b"reviewed-runtime")
        archive.writestr("evrptw/_core.extra.so", b"unloaded-native")
    with pytest.raises(RuntimeError, match="source inventory does not match Git"):
        _verify_installed_project_files(
            extra_core_wheel,
            site_packages=site_packages,
            expected_source_entries={"evrptw/runtime.py": runtime_sha256},
            generated_entries={"evrptw/_core.primary.so"},
        )


def _initial_four_lane_projection(
    *, request_sha256: str
) -> tuple[dict[str, object], str, str]:
    route_offsets = [0, 1]
    route_indices = [1]
    path_offsets = [0, 3]
    path_indices = [0, 1, 0]
    statuses = [0]
    reasons = [0]
    metrics = [[2.0, 2.0, 0.0, 0.0]]
    labels = [[1, 1, 0]]
    batch_counters = [1, 1, 1, 0, 1, 0, 0, 0, 1, 128]
    objective_integer = [1, 0]
    objective_float = [2.0, 0.0]
    accounting = [1, 1, 0, 0]
    rng_seeds = [2014, 2014 ^ 0x5EED23]
    node_kind = [0, 1]
    exact_batch_size = 128
    lane_evidence = bytearray(b"stage05.2-native-lane-state-v2")
    integer_columns = (
        route_offsets,
        route_indices,
        path_offsets,
        path_indices,
        statuses,
        reasons,
    )
    for values in integer_columns:
        lane_evidence.extend(struct.pack("<Q", len(values)))
        lane_evidence.extend(struct.pack(f"<{len(values)}q", *values))
    flattened_metrics = [value for row in metrics for value in row]
    lane_evidence.extend(struct.pack("<Q", len(flattened_metrics)))
    lane_evidence.extend(struct.pack(f"<{len(flattened_metrics)}d", *flattened_metrics))
    for values in (
        [value for row in labels for value in row],
        batch_counters,
        objective_integer,
    ):
        lane_evidence.extend(struct.pack("<Q", len(values)))
        lane_evidence.extend(struct.pack(f"<{len(values)}q", *values))
    lane_evidence.extend(struct.pack("<Q", len(objective_float)))
    lane_evidence.extend(struct.pack("<2d", *objective_float))
    lane_sha256 = hashlib.sha256(lane_evidence).hexdigest()
    initial_state_evidence = bytearray(
        b"stage05.2-native-initial-search-state-v2"
    )
    initial_state_evidence.extend(request_sha256.encode("ascii"))
    for values in (path_offsets, path_indices, statuses, reasons):
        initial_state_evidence.extend(struct.pack("<Q", len(values)))
        initial_state_evidence.extend(struct.pack(f"<{len(values)}q", *values))
    initial_state_evidence.extend(struct.pack("<Q", len(flattened_metrics)))
    initial_state_evidence.extend(
        struct.pack(f"<{len(flattened_metrics)}d", *flattened_metrics)
    )
    for values in (
        [value for row in labels for value in row],
        batch_counters,
        objective_integer,
    ):
        initial_state_evidence.extend(struct.pack("<Q", len(values)))
        initial_state_evidence.extend(struct.pack(f"<{len(values)}q", *values))
    initial_state_evidence.extend(struct.pack("<Q", len(objective_float)))
    initial_state_evidence.extend(struct.pack("<2d", *objective_float))
    initial_state_evidence.extend(struct.pack("<Q", len(accounting)))
    initial_state_evidence.extend(struct.pack("<4q", *accounting))
    initial_state_sha256 = hashlib.sha256(initial_state_evidence).hexdigest()
    state_evidence = bytearray(b"stage05.2-native-initial-four-lane-state-v2")
    state_evidence.extend(request_sha256.encode("ascii"))
    state_evidence.extend(initial_state_sha256.encode("ascii"))
    state_evidence.extend(lane_sha256.encode("ascii") * 4)
    for values in (accounting, rng_seeds):
        state_evidence.extend(struct.pack("<Q", len(values)))
        state_evidence.extend(struct.pack(f"<{len(values)}q", *values))
    state_evidence.extend(struct.pack("<q", 0))
    state_evidence.extend(struct.pack("<Q", len(node_kind)))
    state_evidence.extend(struct.pack(f"<{len(node_kind)}q", *node_kind))
    state_evidence.extend(struct.pack("<q", exact_batch_size))
    state_sha256 = hashlib.sha256(state_evidence).hexdigest()
    return (
        {
            "schema_version": "stage05.2-native-initial-four-lane-projection-v1",
            "route_offsets": route_offsets,
            "route_indices": route_indices,
            "path_offsets": path_offsets,
            "path_indices": path_indices,
            "statuses": statuses,
            "reasons": reasons,
            "metrics": metrics,
            "label_counters": labels,
            "batch_counters": batch_counters,
            "objective_integer": objective_integer,
            "objective_float": objective_float,
            "accounting": accounting,
            "rng_seeds": rng_seeds,
            "next_iteration": 0,
            "lane_count": 4,
            "node_kind": node_kind,
            "exact_batch_size": exact_batch_size,
            "lane_sha256": lane_sha256,
            "state_sha256": state_sha256,
        },
        initial_state_sha256,
        state_sha256,
    )


def test_reviewer_independently_replays_persisted_initial_state_receipt() -> None:
    request_sha256 = "a" * 64
    (
        projection,
        state_sha256,
        initial_four_lane_state_sha256,
    ) = _initial_four_lane_projection(
        request_sha256=request_sha256,
    )
    evidence = bytearray(b"stage05.2-native-initial-state-receipt-v3")
    evidence.extend(struct.pack("<qq", 1, 1))
    evidence.extend(request_sha256.encode("ascii"))
    evidence.extend(state_sha256.encode("ascii"))
    evidence.extend(initial_four_lane_state_sha256.encode("ascii"))
    receipt_sha256 = hashlib.sha256(evidence).hexdigest()
    payload: dict[str, object] = {
        "seed": 2014,
        "native_execution_statistics": {
            "initial_state_request_count": 1,
            "initial_state_receipt": {
                "schema_version": "stage05.2-native-initial-state-receipt-v3",
                "host_owned": True,
                "operation_count": 1,
                "request_sha256": request_sha256,
                "state_sha256": state_sha256,
                "initial_four_lane_state_sha256": (initial_four_lane_state_sha256),
                "initial_four_lane_projection": projection,
                "transaction_sha256": receipt_sha256,
            },
        }
    }

    assert _replay_initial_state_receipt(
        payload,
        ArchitectureMode.HOST_SCHEDULER,
        expected_node_kind=(0, 1),
        expected_exact_batch_size=128,
    ) is None
    native = payload["native_execution_statistics"]
    assert isinstance(native, dict)
    initial = native["initial_state_receipt"]
    assert isinstance(initial, dict)
    initial["state_sha256"] = "d" * 64
    assert (
        _replay_initial_state_receipt(
            payload,
            ArchitectureMode.HOST_SCHEDULER,
            expected_node_kind=(0, 1),
            expected_exact_batch_size=128,
        )
        == "initial-state ownership receipt hash mismatch"
    )
    initial["state_sha256"] = state_sha256
    native["initial_state_request_count"] = True
    assert _replay_initial_state_receipt(
        payload,
        ArchitectureMode.HOST_SCHEDULER,
        expected_node_kind=(0, 1),
        expected_exact_batch_size=128,
    ) == "initial-state ownership receipt does not reconcile"


def test_reviewer_rejects_rehashed_false_initial_state_identity() -> None:
    request_sha256 = "a" * 64
    projection, _, _ = _initial_four_lane_projection(
        request_sha256=request_sha256,
    )
    false_initial_sha256 = "d" * 64
    lane_sha256 = str(projection["lane_sha256"])
    accounting = list(projection["accounting"])
    rng_seeds = list(projection["rng_seeds"])
    node_kind = list(projection["node_kind"])
    exact_batch_size = int(projection["exact_batch_size"])
    state_evidence = bytearray(b"stage05.2-native-initial-four-lane-state-v2")
    state_evidence.extend(request_sha256.encode("ascii"))
    state_evidence.extend(false_initial_sha256.encode("ascii"))
    state_evidence.extend(lane_sha256.encode("ascii") * 4)
    for values in (accounting, rng_seeds):
        state_evidence.extend(struct.pack("<Q", len(values)))
        state_evidence.extend(struct.pack(f"<{len(values)}q", *values))
    state_evidence.extend(struct.pack("<q", 0))
    state_evidence.extend(struct.pack("<Q", len(node_kind)))
    state_evidence.extend(struct.pack(f"<{len(node_kind)}q", *node_kind))
    state_evidence.extend(struct.pack("<q", exact_batch_size))
    false_four_lane_sha256 = hashlib.sha256(state_evidence).hexdigest()
    projection["state_sha256"] = false_four_lane_sha256
    receipt_evidence = bytearray(b"stage05.2-native-initial-state-receipt-v3")
    receipt_evidence.extend(struct.pack("<qq", 1, 1))
    receipt_evidence.extend(request_sha256.encode("ascii"))
    receipt_evidence.extend(false_initial_sha256.encode("ascii"))
    receipt_evidence.extend(false_four_lane_sha256.encode("ascii"))
    payload: dict[str, object] = {
        "seed": 2014,
        "native_execution_statistics": {
            "initial_state_request_count": 1,
            "initial_state_receipt": {
                "schema_version": "stage05.2-native-initial-state-receipt-v3",
                "host_owned": True,
                "operation_count": 1,
                "request_sha256": request_sha256,
                "state_sha256": false_initial_sha256,
                "initial_four_lane_state_sha256": false_four_lane_sha256,
                "initial_four_lane_projection": projection,
                "transaction_sha256": hashlib.sha256(
                    receipt_evidence
                ).hexdigest(),
            },
        },
    }

    assert _replay_initial_state_receipt(
        payload,
        ArchitectureMode.HOST_SCHEDULER,
        expected_node_kind=(0, 1),
        expected_exact_batch_size=128,
    ) == "initial-state projection hash mismatch"


@pytest.mark.parametrize(
    "field",
    [
        "route_offsets",
        "metrics",
        "objective_float",
        "rng_seeds",
        "next_iteration",
        "lane_count",
        "batch_counters",
        "node_kind",
        "exact_batch_size",
    ],
)
def test_reviewer_rejects_invalid_initial_projection_values(
    field: str,
) -> None:
    request_sha256 = "a" * 64
    (
        projection,
        state_sha256,
        initial_four_lane_state_sha256,
    ) = _initial_four_lane_projection(
        request_sha256=request_sha256,
    )
    if field == "route_offsets":
        projection[field] = [0, 1 << 80]
    elif field == "metrics":
        projection[field] = [[1 << 2000, 2.0, 0.0, 0.0]]
    elif field == "objective_float":
        projection[field] = ["2.0", 0.0]
    elif field == "rng_seeds":
        projection[field] = [2015, 2015 ^ 0x5EED23]
    elif field == "next_iteration":
        projection[field] = 0.0
    elif field == "lane_count":
        projection[field] = 4.0
    elif field == "batch_counters":
        projection[field] = [1, 1, 1, 0, 1, 0, 0, 0, 1, 127]
    elif field == "node_kind":
        projection[field] = [0, 2]
    else:
        projection[field] = 127
    evidence = bytearray(b"stage05.2-native-initial-state-receipt-v3")
    evidence.extend(struct.pack("<qq", 1, 1))
    evidence.extend(request_sha256.encode("ascii"))
    evidence.extend(state_sha256.encode("ascii"))
    evidence.extend(initial_four_lane_state_sha256.encode("ascii"))
    payload: dict[str, object] = {
        "seed": 2014,
        "native_execution_statistics": {
            "initial_state_request_count": 1,
            "initial_state_receipt": {
                "schema_version": "stage05.2-native-initial-state-receipt-v3",
                "host_owned": True,
                "operation_count": 1,
                "request_sha256": request_sha256,
                "state_sha256": state_sha256,
                "initial_four_lane_state_sha256": (initial_four_lane_state_sha256),
                "initial_four_lane_projection": projection,
                "transaction_sha256": hashlib.sha256(evidence).hexdigest(),
            },
        }
    }

    assert (
        _replay_initial_state_receipt(
            payload,
            ArchitectureMode.HOST_SCHEDULER,
            expected_node_kind=(0, 1),
            expected_exact_batch_size=128,
        )
        is not None
    )


def test_reviewer_rejects_initial_projection_customer_omission() -> None:
    request_sha256 = "a" * 64
    (
        projection,
        state_sha256,
        initial_four_lane_state_sha256,
    ) = _initial_four_lane_projection(
        request_sha256=request_sha256,
    )
    projection["node_kind"] = [0, 1, 1]
    evidence = bytearray(b"stage05.2-native-initial-state-receipt-v3")
    evidence.extend(struct.pack("<qq", 1, 1))
    evidence.extend(request_sha256.encode("ascii"))
    evidence.extend(state_sha256.encode("ascii"))
    evidence.extend(initial_four_lane_state_sha256.encode("ascii"))
    payload: dict[str, object] = {
        "seed": 2014,
        "native_execution_statistics": {
            "initial_state_request_count": 1,
            "initial_state_receipt": {
                "schema_version": "stage05.2-native-initial-state-receipt-v3",
                "host_owned": True,
                "operation_count": 1,
                "request_sha256": request_sha256,
                "state_sha256": state_sha256,
                "initial_four_lane_state_sha256": (
                    initial_four_lane_state_sha256
                ),
                "initial_four_lane_projection": projection,
                "transaction_sha256": hashlib.sha256(evidence).hexdigest(),
            },
        },
    }

    assert _replay_initial_state_receipt(
        payload,
        ArchitectureMode.HOST_SCHEDULER,
        expected_node_kind=(0, 1, 1),
        expected_exact_batch_size=128,
    ) == "initial four-lane projection values do not reconcile"


def test_native_attempt04_is_blocked_before_capability_incomplete_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A zero capability receipt must stop attempt04 before any run label exists."""

    import evrptw.experiments.stage052_native_architectures as native_architectures
    from evrptw import _core as native_core

    root = tmp_path / "checkout"
    root.mkdir()
    output_root = tmp_path / "results"
    wheel_path = tmp_path / "comparison.whl"
    wheel_path.write_bytes(b"frozen-wheel")

    class Receipt:
        def __init__(self, stdout: str) -> None:
            self.stdout = stdout

    monkeypatch.setattr(native_architectures, "repository_root", lambda: root)
    monkeypatch.setattr(native_architectures, "require_owned", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(native_architectures, "_configure_compute_envelope", lambda: {})
    monkeypatch.setattr(
        native_architectures.subprocess,
        "run",
        lambda command, **_kwargs: Receipt(
            "c" * 40 + "\n" if command[1] == "rev-parse" else ""
        ),
    )
    monkeypatch.setattr(
        native_architectures,
        "_verify_installed_wheel",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        native_architectures,
        "committed_source_attestation",
        lambda *_args, **_kwargs: {
            "source_manifest_sha256": "d" * 64,
            "tracked_file_count": 1,
        },
    )
    monkeypatch.setattr(
        native_architectures,
        "committed_wheel_project_entry_sha256",
        lambda *_args, **_kwargs: {"evrptw/runtime.py": "d" * 64},
    )
    monkeypatch.setattr(
        native_core,
        "stage052_native_architecture_capabilities_v2",
        lambda: np.zeros(4, dtype=np.int64),
    )

    with pytest.raises(RuntimeError, match="incomplete capabilities"):
        run_experiment(
            "pilot",
            attempt=4,
            output_root=output_root,
            wheel_path=wheel_path,
            warm_start_bundle_path=tmp_path / "warm-starts.json",
            continuity_lease_token="lease-token",
        )

    assert not tuple(output_root.glob("stage05.2_native_architecture_*/"))


_SEMANTIC_STREAM_NAMES = (
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
_REQUIRED_CAUSAL_STREAMS = frozenset(
    {
        "operator",
        "stage04",
        "candidate_transaction",
        "exact_work",
        "exact_result",
        "cache",
        "screening",
        "termination",
    }
)
_BOUNDARY_CAUSAL_STREAMS = _REQUIRED_CAUSAL_STREAMS | {"deadline"}


def _complete_causal_streams(
    *,
    exclude: str | None = None,
) -> dict[str, list[dict[str, object]]]:
    event_types = {
        "candidate_state": "candidate_state",
        "operator": "operator_call",
        "stage04": "stage04_segment_update",
        "candidate_transaction": "candidate_control_budget",
        "exact_work": "exact_batch_started",
        "exact_result": "exact_route_result",
        "cache": "cache_lookup_result",
        "screening": "screening_decision",
        "deadline": "deadline_boundary",
        "termination": "termination",
    }
    streams = {name: [] for name in _SEMANTIC_STREAM_NAMES}
    event_id = 0
    for stream_name in event_types:
        if stream_name == exclude:
            continue
        event_id += 1
        streams[stream_name].append(
            {
                "event_type": event_types[stream_name],
                "runtime_event_id": event_id,
                "semantic_event_id": event_id,
                "stream_ordinal": 0,
                **(
                    {"status": "iteration_limit"}
                    if stream_name == "termination"
                    else {}
                ),
            }
        )
    return streams


def test_unified_causal_ids_are_monotonic_and_cover_all_required_domains() -> None:
    streams = _complete_causal_streams()
    events = _canonical_semantic_event_sequence(streams)

    assert [event["semantic_event_id"] for event in events] == list(
        range(1, len(events) + 1)
    )
    assert [event["runtime_event_id"] for event in events] == list(
        range(1, len(events) + 1)
    )
    assert {str(event["semantic_stream"]) for event in events} >= (
        _BOUNDARY_CAUSAL_STREAMS
    )

    reviewed = _canonical_semantic_events(
        {
            "mode": "full_native_alns",
            "canonical_semantic_streams": streams,
            "canonical_semantic_events": events,
        }
    )
    assert {str(event["semantic_stream"]) for event in reviewed} >= (
        _BOUNDARY_CAUSAL_STREAMS
    )


@pytest.mark.parametrize("missing_stream", sorted(_REQUIRED_CAUSAL_STREAMS))
def test_unified_causal_ids_reject_missing_required_domain(
    missing_stream: str,
) -> None:
    streams = _complete_causal_streams(exclude=missing_stream)
    events = _canonical_semantic_event_sequence(streams)

    with pytest.raises(ValueError):
        _canonical_semantic_events(
            {
                "mode": "full_native_alns",
                "canonical_semantic_streams": streams,
                "canonical_semantic_events": events,
            }
        )


@pytest.mark.parametrize("corruption", ("missing", "duplicate", "out_of_order"))
def test_unified_causal_ids_reject_missing_duplicate_or_out_of_order_events(
    corruption: str,
) -> None:
    streams = _complete_causal_streams()
    events = _canonical_semantic_event_sequence(streams)
    corrupted = [dict(event) for event in events]
    if corruption == "missing":
        corrupted.pop()
    elif corruption == "duplicate":
        corrupted[1] = dict(corrupted[0])
    else:
        corrupted[0], corrupted[1] = corrupted[1], corrupted[0]

    with pytest.raises(ValueError):
        _canonical_semantic_events(
            {
                "mode": "full_native_alns",
                "canonical_semantic_streams": streams,
                "canonical_semantic_events": corrupted,
            }
        )


def test_campaign_identity_revalidates_lease_revision_and_clean_tree(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    lease_calls: list[tuple[Path, str, frozenset[str]]] = []

    def fake_require_owned(
        root: Path,
        *,
        token: str,
        allowed_phases: frozenset[str],
    ) -> None:
        lease_calls.append((root, token, allowed_phases))

    class Receipt:
        def __init__(self, stdout: str) -> None:
            self.stdout = stdout

    def fake_run(command: list[str], **_: object) -> Receipt:
        return Receipt("a" * 40 + "\n" if command[1] == "rev-parse" else "")

    monkeypatch.setattr(
        "evrptw.experiments.stage052_native_architectures.require_owned",
        fake_require_owned,
    )
    monkeypatch.setattr(
        "evrptw.experiments.stage052_native_architectures.subprocess.run",
        fake_run,
    )

    _require_campaign_identity(
        tmp_path,
        continuity_lease_token="lease-token",
        expected_revision="a" * 40,
    )

    assert lease_calls == [
        (
            tmp_path,
            "lease-token",
            frozenset({"paired-campaign", "pilot-campaign"}),
        )
    ]


@pytest.mark.parametrize(
    ("revision", "status", "message"),
    [
        ("b" * 40, "", "Git revision changed"),
        ("a" * 40, " M source.py\n", "worktree changed"),
    ],
)
def test_campaign_identity_rejects_checkout_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    revision: str,
    status: str,
    message: str,
) -> None:
    class Receipt:
        def __init__(self, stdout: str) -> None:
            self.stdout = stdout

    monkeypatch.setattr(
        "evrptw.experiments.stage052_native_architectures.require_owned",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "evrptw.experiments.stage052_native_architectures.subprocess.run",
        lambda command, **_kwargs: Receipt(
            revision + "\n" if command[1] == "rev-parse" else status
        ),
    )

    with pytest.raises(RuntimeError, match=message):
        _require_campaign_identity(
            tmp_path,
            continuity_lease_token="lease-token",
            expected_revision="a" * 40,
        )


def _plan(scope: str, tmp_path: Path):  # type: ignore[no-untyped-def]
    from evrptw.experiments.stage02_route_reduction import FORMAL_INSTANCES

    instances = PAIRED_INSTANCES if scope == "paired" else tuple(FORMAL_INSTANCES)
    warm_starts = {
        (instance_name, seed): (
            (("C1",),),
            {"source_solution_sha256": "d" * 64},
        )
        for instance_name in instances
        for seed in SEEDS
    }
    return build_axis_plan(
        scope,
        attempt=1,
        benchmark_dir=tmp_path / "benchmarks",
        output_root=tmp_path / "results",
        scheduler_socket_path=str(tmp_path / "scheduler.sock"),
        wheel_sha256="a" * 64,
        native_sha256="b" * 64,
        scheduler_sha256="f" * 64,
        revision="c" * 40,
        warm_starts=warm_starts,
    )


def _c5_warm_start(root: Path) -> tuple[tuple[str, ...], ...]:
    instance = parse_schneider(root / "data/schneider/c101C5.txt")
    return tuple((customer.name,) for customer in instance.customers)


def _c5_source_provenance(root: Path, destination: Path) -> dict[str, object]:
    instance = parse_schneider(root / "data/schneider/c101C5.txt")
    sequences = _c5_warm_start(root)
    routes = [list(solve_exact_charging(instance, sequence).route) for sequence in sequences]
    objective_key = list(
        SolutionObjective.from_report(instance, validate_routes(instance, routes)).key
    )
    payload = {"axes": {"wall_clock": {"routes": routes, "objective_key": objective_key}}}
    destination.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return {
        "source_solution_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "source_solution_path": str(destination),
        "source_axis": "wall_clock",
        "source_customer_sequences_sha256": canonical_customer_sequences_sha256(
            sequences
        ),
        "source_objective_key": objective_key,
    }


def test_warm_start_bundle_binds_hash_identity_and_customer_coverage(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    routes = _c5_warm_start(root)
    source_path = tmp_path / "source-solution.json"
    source_provenance = _c5_source_provenance(root, source_path)
    bundle = tmp_path / "warm-starts.json"
    payload = {
        "schema_version": WARM_START_SCHEMA_VERSION,
        "records": [
            {
                "instance": "c101C5",
                "seed": 2014,
                "customer_sequences": [list(route) for route in routes],
                **source_provenance,
            }
        ],
    }
    data = json.dumps(payload, sort_keys=True).encode("utf-8")
    bundle.write_bytes(data)
    bundle.with_suffix(".json.sha256").write_text(
        hashlib.sha256(data).hexdigest(),
        encoding="ascii",
    )

    loaded = load_warm_start_bundle(
        bundle,
        benchmark_dir=root / "data/schneider",
    )

    assert loaded[("c101C5", 2014)][0] == routes
    assert loaded[("c101C5", 2014)][1]["warm_start_bundle_sha256"] == (
        hashlib.sha256(data).hexdigest()
    )

    bundle.with_suffix(".json.sha256").write_text("0" * 64, encoding="ascii")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        load_warm_start_bundle(bundle, benchmark_dir=root / "data/schneider")

    bundle.with_suffix(".json.sha256").write_text(
        hashlib.sha256(data).hexdigest(),
        encoding="ascii",
    )
    source_path.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="source solution hash mismatch"):
        load_warm_start_bundle(bundle, benchmark_dir=root / "data/schneider")


def test_paired_plan_has_360_axes_and_rotates_all_five_modes(tmp_path: Path) -> None:
    plan = _plan("paired", tmp_path)

    assert len(plan) * len(MODES) == expected_axis_count("paired") == 360
    assert all(set(rotated_modes(task)) == set(MODES) for task in plan)
    first_mode_counts = Counter(rotated_modes(task)[0] for task in plan)
    assert max(first_mode_counts.values()) - min(first_mode_counts.values()) <= 1


def test_pilot_plan_has_180_wall_clock_axes_and_independent_labels(
    tmp_path: Path,
) -> None:
    plan = _plan("pilot", tmp_path)
    labels = run_labels_for_scope("pilot", 1)

    assert len(plan) * len(MODES) == expected_axis_count("pilot") == 180
    assert {task.axis for task in plan} == {"wall_clock_30"}
    assert len(set(labels.values())) == len(MODES)
    assert all("_pilot_attempt01" in label for label in labels.values())


def _review_fixture_records(
    root: Path, evidence_root: Path
) -> tuple[ReviewRecord, ...]:
    instance = parse_schneider(root / "data/schneider/c101C5.txt")
    charging = [solve_exact_charging(instance, (customer.name,)) for customer in instance.customers]
    assert all(result.feasible for result in charging)
    routes = [list(result.route) for result in charging]
    report = validate_routes(instance, routes)
    assert report.feasible
    objective = SolutionObjective.from_report(instance, report)
    records = []
    empty_rows = {
        "count": 0,
        "sha256": hashlib.sha256(b"stage05.2-row-evidence-v1\0").hexdigest(),
    }
    measurement_evidence: dict[str, Any] = {
        "present": True,
        "exact_route_order": empty_rows,
        "cache_lifecycle": empty_rows,
        "deadline_boundaries": empty_rows,
    }
    measurement_evidence["sha256"] = hashlib.sha256(
        json.dumps(
            measurement_evidence,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    for mode in MODES:
        payload: dict[str, Any] = {
            "schema_version": "stage05.2-native-architecture-comparison-v3",
            "status": "completed",
            "mode": mode.value,
            "repeat": 0,
            "axis": "fixed_work",
            "instance": "c101C5",
            "seed": 2014,
            "revision": "c" * 40,
            "wheel_sha256": "d" * 64,
            "native_sha256": "e" * 64,
            "run_label": f"stage05.2_native_architecture_{mode.value}_test_attempt01",
            "routes": routes,
            "objective": list(objective.key),
            "solver_seconds": 1.0,
            "effective_iterations": 50,
            "exact_started_calls": 10,
            "exact_completed_calls": 10,
            "candidate_work_hash": "a" * 64,
            "route_result_hash": "b" * 64,
            "trajectory": empty_rows,
            "operator_statistics": {},
            "stage04_statistics": {},
            "stage04_events": empty_rows,
            "candidate_transaction_events": empty_rows,
            "measurement_evidence": measurement_evidence,
            "semantic_completeness": {
                "candidate_control": True,
                "stage04": True,
                "measurement_trace": True,
            },
            "fallback_count": 0,
            "native_execution_statistics": {"fallback_count": 0},
            "backend_metrics": {"launch_occupancies": [4]},
            "topology": {
                "rss_bytes": 1024,
                "cpu_utilization_percent_of_one_core": 100.0,
            },
            "cache_memory_bytes": 512,
            "artifact_bytes": 2048,
            "persistence_seconds": 0.01,
            "throughput": {
                "effective_iterations_per_second": 50.0,
                "candidate_transactions_per_second": 10.0,
                "screened_routes_per_second": 10.0,
                "exact_started_per_second": 10.0,
            },
        }
        path = evidence_root / f"{mode.value}.json"
        data = json.dumps(payload, sort_keys=True).encode("utf-8")
        path.write_bytes(data)
        path.with_suffix(path.suffix + ".sha256").write_text(
            hashlib.sha256(data).hexdigest(), encoding="ascii"
        )
        records.append(ReviewRecord(path, payload))
    return tuple(records)


def test_independent_review_replays_routes_and_accepts_equal_fixed_work(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    review = review_records(
        _review_fixture_records(root, tmp_path),
        scope="paired",
        benchmark_dir=root / "data/schneider",
    )

    assert review["axis_replay_passed"] is True
    gates = review["differential_gates"]
    assert isinstance(gates, dict)
    assert all(bool(gates[mode.value]["passed"]) for mode in MODES[2:])
    assert "五模式事实表" in render_report(review)


def test_semantic_trajectory_reports_the_real_first_divergence() -> None:
    baseline = [
        {"lane": "legacy", "iteration": 0, "operator": "relocate"},
        {"lane": "constraint", "iteration": 0, "operator": "station_pressure"},
        {"lane": "legacy", "iteration": 1, "operator": "swap"},
    ]
    candidate = [
        baseline[0],
        {"lane": "constraint", "iteration": 0, "operator": "shaw_related"},
        baseline[2],
    ]

    assert _common_prefix(baseline, candidate) == 1


def test_v5_reviewer_rejects_scheduler_overlap_outside_host_wave(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    revision = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    git_tree = subprocess.run(
        ("git", "rev-parse", "HEAD^{tree}"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    source_attestation = committed_source_attestation(root, revision)
    source_manifest_sha256 = source_attestation["source_manifest_sha256"]
    tracked_file_count = source_attestation["tracked_file_count"]
    wheel_entries = committed_wheel_project_entry_sha256(root, revision)
    wheel_entries.update(
        {
            "evrptw/_core.cpython-313-x86_64-linux-gnu.so": "c" * 64,
            "evrptw/_native_host_scheduler": "d" * 64,
        }
    )
    wheel_receipt = {
        "build_git_revision": revision,
        "build_git_tree": git_tree,
        "wheel_sha256": "b" * 64,
        "native_sha256": "c" * 64,
        "scheduler_sha256": "d" * 64,
        "build_source_manifest_sha256": source_manifest_sha256,
        "build_tracked_file_count": tracked_file_count,
        "build_source_dirty": False,
        "build_development_override": False,
        "build_cpp_source_kind": "git_blob_snapshot",
        "build_source_attestation_version": 1,
        "wheel_entry_sha256": wheel_entries,
        "native_wheel_entry": "evrptw/_core.cpython-313-x86_64-linux-gnu.so",
        "scheduler_wheel_entry": "evrptw/_native_host_scheduler",
        "runner_wheel_entry": "evrptw/experiments/stage052_native_architectures.py",
        "scheduler_build_attestation": {
            "schema_version": 1,
            "revision": revision,
            "git_tree": git_tree,
            "source_manifest_sha256": source_manifest_sha256,
            "tracked_file_count": tracked_file_count,
            "source_dirty": False,
            "development_override": False,
            "cpp_source_kind": "git_blob_snapshot",
        },
    }
    labels = run_labels_for_scope("paired", 91)
    run_dir = tmp_path / labels["current_stage052"]
    _write_signed_json(
        run_dir / "run_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "revision": revision,
            "git_tree": git_tree,
            "wheel_sha256": "b" * 64,
            "native_sha256": "c" * 64,
            "scheduler_sha256": "d" * 64,
            "wheel_receipt": wheel_receipt,
            "topology": {
                "scheduler_startup_seconds": 0.1,
                "scheduler_shutdown_seconds": 0.1,
                "scheduler_observed": [],
                "mode_wave_resources": [
                    {
                        "mode": "current_stage052",
                        "elapsed_seconds": 1.0,
                        "process_tree_cpu_seconds": 1.0,
                        "peak_aggregate_rss_bytes": 1,
                        "peak_aggregate_pss_bytes": 1,
                        "scheduler_process_id": 123,
                        "scheduler_startup_seconds": 0.0,
                        "scheduler_shutdown_seconds": 0.0,
                    }
                ],
            },
        },
    )

    with pytest.raises(RuntimeError, match="exclusive mode wave"):
        load_records("paired", attempt=91, results_root=tmp_path)


def test_reviewer_rejects_dirty_or_mismatched_build_attestation() -> None:
    root = Path(__file__).resolve().parents[1]
    revision = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    git_tree = subprocess.run(
        ("git", "rev-parse", "HEAD^{tree}"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    source_attestation = committed_source_attestation(root, revision)
    source_manifest_sha256 = source_attestation["source_manifest_sha256"]
    tracked_file_count = source_attestation["tracked_file_count"]
    wheel_entries = committed_wheel_project_entry_sha256(root, revision)
    wheel_entries.update(
        {
            "evrptw/_core.cpython-313-x86_64-linux-gnu.so": "c" * 64,
            "evrptw/_native_host_scheduler": "d" * 64,
        }
    )
    receipt: dict[str, object] = {
        "build_git_revision": revision,
        "build_git_tree": git_tree,
        "wheel_sha256": "b" * 64,
        "native_sha256": "c" * 64,
        "scheduler_sha256": "d" * 64,
        "build_source_manifest_sha256": source_manifest_sha256,
        "build_tracked_file_count": tracked_file_count,
        "build_source_dirty": False,
        "build_development_override": False,
        "build_cpp_source_kind": "git_blob_snapshot",
        "build_source_attestation_version": 1,
        "wheel_entry_sha256": wheel_entries,
        "native_wheel_entry": "evrptw/_core.cpython-313-x86_64-linux-gnu.so",
        "scheduler_wheel_entry": "evrptw/_native_host_scheduler",
        "runner_wheel_entry": "evrptw/experiments/stage052_native_architectures.py",
        "scheduler_build_attestation": {
            "schema_version": 1,
            "revision": revision,
            "git_tree": git_tree,
            "source_manifest_sha256": source_manifest_sha256,
            "tracked_file_count": tracked_file_count,
            "source_dirty": False,
            "development_override": False,
            "cpp_source_kind": "git_blob_snapshot",
        },
    }
    manifest: dict[str, object] = {
        "revision": revision,
        "git_tree": git_tree,
        "wheel_sha256": "b" * 64,
        "native_sha256": "c" * 64,
        "scheduler_sha256": "d" * 64,
        "wheel_receipt": receipt,
    }
    assert _review_build_attestation(manifest) == (
        git_tree,
        source_manifest_sha256,
    )

    for field, value, message in (
        ("build_source_dirty", True, "source is dirty"),
        ("build_development_override", True, "development build override"),
        ("build_cpp_source_kind", "working_tree_override", "Git-blob C\\+\\+ snapshot"),
        ("build_source_attestation_version", True, "version is invalid"),
        ("build_git_tree", "0" * 40, "identity does not reconcile"),
    ):
        invalid_receipt = dict(receipt)
        invalid_receipt[field] = value
        invalid_manifest = {**manifest, "wheel_receipt": invalid_receipt}
        with pytest.raises(RuntimeError, match=message):
            _review_build_attestation(invalid_manifest)

    for mutate in ("missing", "unexpected"):
        invalid_entries = dict(wheel_entries)
        if mutate == "missing":
            committed_entry = next(
                entry
                for entry in committed_wheel_project_entries(root, revision)
                if entry
                != "evrptw/experiments/stage052_native_architectures.py"
            )
            invalid_entries.pop(committed_entry)
        else:
            invalid_entries["evrptw/uncommitted.py"] = "a" * 64
        invalid_receipt = {**receipt, "wheel_entry_sha256": invalid_entries}
        with pytest.raises(RuntimeError, match="inventory does not match Git"):
            _review_build_attestation(
                {**manifest, "wheel_receipt": invalid_receipt}
            )

    modified_runner_entries = dict(wheel_entries)
    modified_runner_entries[
        "evrptw/experiments/stage052_native_architectures.py"
    ] = "0" * 64
    with pytest.raises(RuntimeError, match="inventory does not match Git"):
        _review_build_attestation(
            {
                **manifest,
                "wheel_receipt": {
                    **receipt,
                    "wheel_entry_sha256": modified_runner_entries,
                },
            }
        )


def test_canonical_candidate_event_has_stable_candidate_identity() -> None:
    event = {
        "event_type": "candidate_state",
        "timestamp_seconds": 1.25,
        "lane": "constraint_lane",
        "iteration": 7,
        "operator": "shaw_related",
        "candidate_route_keys": ["route-b", "route-a"],
        "candidate_full_route_keys": ["full-b", "full-a"],
        "candidate_id": "producer-local-a",
        "accepted": False,
    }

    canonical = _canonical_trace_event(event)
    repeated = _canonical_trace_event(
        {
            **event,
            "timestamp_seconds": 9.0,
            "candidate_id": "producer-local-b",
        }
    )

    assert canonical == repeated
    assert canonical["candidate_id"] == (
        "067f6e7fb31c2e26543409583f6b9877247769891162f00de974d42bd994c154"
    )
    assert "timestamp_seconds" not in canonical

    trajectory_event = {
        "lane": "constraint_lane",
        "track": "constraint_lane",
        "iteration": 7,
        "operator": "shaw_related",
        "status": "prefilter_rejected",
        "candidate_route_sequences": [["C1", "C2"]],
        "candidate_objective_key": [],
    }
    trajectory_identity = {
        "lane": "constraint_lane",
        "iteration": 7,
        "operator": "shaw_related",
        "status": "prefilter_rejected",
        "candidate_route_sequences": [["C1", "C2"]],
        "candidate_objective_key": [],
        "ordinal": 0,
    }
    trajectory_event["candidate_id"] = hashlib.sha256(
        json.dumps(
            trajectory_identity,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    with pytest.raises(ValueError, match="candidate_id"):
        _semantic_trajectory(
            {
                "semantic_trajectory": [
                    {**trajectory_event, "candidate_id": "0" * 64}
                ]
            }
        )
    assert _semantic_trajectory(
        {"semantic_trajectory": [trajectory_event]}
    ) == [trajectory_event]
    with pytest.raises(ValueError, match="candidate_route_keys"):
        _canonical_trace_event({"event_type": "candidate_state"})
    with pytest.raises(ValueError, match="lane projection"):
        _semantic_trajectory(
            {"semantic_trajectory": [{"lane": "constraint_lane"}]}
        )


def test_first_divergence_reports_coordinates_and_differing_fields() -> None:
    baseline = [
        {
            "lane": "constraint_lane",
            "iteration": 4,
            "operator": "shaw_related",
            "candidate_id": "candidate-4",
            "accepted": True,
            "status": "accepted",
        }
    ]
    candidate = [
        {
            "lane": "constraint_lane",
            "iteration": 4,
            "operator": "shaw_related",
            "candidate_id": "candidate-4",
            "accepted": False,
            "status": "rejected",
        }
    ]

    divergence = _describe_first_divergence(baseline, candidate)

    assert divergence == {
        "index": 0,
        "lane": "constraint_lane",
        "iteration": 4,
        "operator": "shaw_related",
        "candidate_id": "candidate-4",
        "differing_fields": {
            "accepted": {"baseline": True, "candidate": False},
            "status": {"baseline": "accepted", "candidate": "rejected"},
        },
        "baseline": baseline[0],
        "candidate": candidate[0],
    }


def test_first_divergence_distinguishes_missing_field_from_null() -> None:
    baseline = [
        {
            "lane": "legacy",
            "iteration": 2,
            "operator": "route_merge",
            "candidate_id": "candidate-2",
            "reason": None,
        }
    ]
    candidate = [{"lane": "legacy"}]

    divergence = _describe_first_divergence(baseline, candidate)

    assert divergence is not None
    assert divergence["lane"] == "legacy"
    assert divergence["iteration"] == 2
    assert divergence["operator"] == "route_merge"
    assert divergence["candidate_id"] == "candidate-2"
    assert divergence["differing_fields"] == {
        "candidate_id": {
            "baseline": "candidate-2",
            "candidate": {"field_missing": True},
        },
        "iteration": {
            "baseline": 2,
            "candidate": {"field_missing": True},
        },
        "operator": {
            "baseline": "route_merge",
            "candidate": {"field_missing": True},
        },
        "reason": {
            "baseline": None,
            "candidate": {"field_missing": True},
        }
    }


def test_canonical_semantic_streams_find_non_candidate_first_divergence() -> None:
    empty_streams = {
        name: []
        for name in (
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
    }
    baseline_streams = {name: list(rows) for name, rows in empty_streams.items()}
    candidate_streams = {name: list(rows) for name, rows in empty_streams.items()}
    baseline_streams["stage04"] = [
        {
            "lane": "legacy",
            "iteration": 7,
            "operator": "route_merge",
            "candidate_id": "candidate-7",
            "weight": 2.0,
            "stream_ordinal": 0,
            "semantic_event_id": 1,
        }
    ]
    candidate_streams["stage04"] = [
        {
            **baseline_streams["stage04"][0],
            "weight": 3.0,
        }
    ]

    baseline = _canonical_semantic_events(
        {
            "canonical_semantic_streams": baseline_streams,
            "canonical_semantic_events": _canonical_semantic_event_sequence(
                baseline_streams
            ),
        }
    )
    candidate = _canonical_semantic_events(
        {
            "canonical_semantic_streams": candidate_streams,
            "canonical_semantic_events": _canonical_semantic_event_sequence(
                candidate_streams
            ),
        }
    )
    divergence = _describe_first_divergence(baseline, candidate)

    assert _common_prefix(baseline, candidate) == 0
    assert divergence is not None
    assert divergence["lane"] == "legacy"
    assert divergence["iteration"] == 7
    assert divergence["operator"] == "route_merge"
    assert divergence["candidate_id"] == "candidate-7"
    assert divergence["differing_fields"] == {
        "weight": {"baseline": 2.0, "candidate": 3.0}
    }


def test_comparison_projection_ignores_only_implementation_batch_telemetry() -> None:
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
    trajectory_identity = {
        "lane": "quality_shadow",
        "iteration": 3,
        "operator": "route_segment_destroy",
        "status": "candidate_proposed",
        "candidate_route_sequences": (),
        "candidate_objective_key": (),
        "ordinal": 0,
    }
    trajectory = [
        {
            "lane": "quality_shadow",
            "iteration": 3,
            "operator": "route_segment_destroy",
            "candidate_id": hashlib.sha256(
                json.dumps(
                    trajectory_identity,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest(),
            "status": "candidate_proposed",
        }
    ]

    def payload(screening_rows: int) -> dict[str, object]:
        streams: dict[str, list[dict[str, object]]] = {
            name: [] for name in stream_names
        }
        event_id = 0
        for row in range(screening_rows):
            event_id += 1
            streams["screening"].append(
                {
                    "semantic_event_id": event_id,
                    "stream_ordinal": row,
                    "status": "pass",
                    "batch_row": row,
                }
            )
        event_id += 1
        streams["candidate_state"].append(
            {
                "semantic_event_id": event_id,
                "stream_ordinal": 0,
                "lane": "quality_shadow",
                "iteration": 3,
                "operator": "route_segment_destroy",
                "candidate_feasible": True,
                "accepted": False,
            }
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "semantic_trajectory": trajectory,
            "canonical_semantic_streams": streams,
            "canonical_semantic_events": _canonical_semantic_event_sequence(
                streams
            ),
        }

    baseline = payload(1)
    candidate = payload(2)
    assert baseline["canonical_semantic_events"] != candidate[
        "canonical_semantic_events"
    ]
    assert _comparison_semantic_events(baseline) == _comparison_semantic_events(
        candidate
    )


def test_canonical_semantic_events_require_explicit_contiguous_causal_sequence() -> None:
    streams = {
        name: []
        for name in (
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
    }
    streams["operator"] = [
        {
            "iteration": 0,
            "lane": "legacy",
            "stream_ordinal": 0,
            "semantic_event_id": 1,
        }
    ]
    streams["exact_work"] = [
        {
            "iteration": 0,
            "lane": "legacy",
            "stream_ordinal": 0,
            "semantic_event_id": 2,
        }
    ]
    events = _canonical_semantic_event_sequence(streams)

    with pytest.raises(ValueError, match="explicit canonical semantic event sequence"):
        _canonical_semantic_events({"canonical_semantic_streams": streams})

    reordered = [
        {**events[1], "semantic_sequence": 0},
        {**events[0], "semantic_sequence": 1},
    ]
    with pytest.raises(ValueError, match="runtime event IDs"):
        _canonical_semantic_events(
            {
                "canonical_semantic_streams": streams,
                "canonical_semantic_events": reordered,
            }
        )


def test_canonical_semantic_events_reject_empty_runtime_journal() -> None:
    streams = {
        name: []
        for name in (
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
    }

    with pytest.raises(ValueError, match="cannot be empty"):
        _canonical_semantic_event_sequence(streams)

    with pytest.raises(ValueError, match="cannot be empty"):
        _canonical_semantic_events(
            {
                "canonical_semantic_streams": streams,
                "canonical_semantic_events": [],
            }
        )


def test_canonical_semantic_events_reconcile_exact_counters() -> None:
    streams = {
        name: []
        for name in (
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
    }
    for event_id, stream_name in enumerate(
        (
            "candidate_state",
            "operator",
            "stage04",
            "candidate_transaction",
            "exact_work",
            "exact_result",
            "cache",
            "screening",
            "termination",
        ),
        start=1,
    ):
        streams[stream_name].append(
            {
                "event_type": (
                    "exact_batch_started"
                    if stream_name == "exact_work"
                    else "exact_route_result"
                    if stream_name == "exact_result"
                    else "screening_decision"
                    if stream_name == "screening"
                    else "termination"
                    if stream_name == "termination"
                    else stream_name
                ),
                "runtime_event_id": event_id,
                "semantic_event_id": event_id,
                "stream_ordinal": 0,
                **(
                    {"started_calls": 1}
                    if stream_name == "exact_work"
                    else {
                        "exact_started": True,
                        "exact_completed": True,
                    }
                    if stream_name == "exact_result"
                    else {"status": "iteration_limit"}
                    if stream_name == "termination"
                    else {}
                ),
            }
        )
    events = _canonical_semantic_event_sequence(streams)

    with pytest.raises(ValueError, match="exact-start counters"):
        _canonical_semantic_events(
            {
                "mode": "per_solve_runtime",
                "exact_started_calls": 2,
                "exact_completed_calls": 1,
                "termination_reason": "iteration_limit",
                "canonical_semantic_streams": streams,
                "canonical_semantic_events": events,
            }
        )


def test_v6_axis_replay_rejects_bad_candidate_id_on_wall_clock_axis(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    source = _review_fixture_records(root, tmp_path)[0]
    payload = dict(source.payload)
    payload["schema_version"] = (
        "stage05.2-native-architecture-comparison-v6"
    )
    payload["axis"] = "wall_clock_30"
    payload["semantic_trajectory"] = [
        {
            "lane": "legacy",
            "iteration": 0,
            "operator": "route_elimination",
            "status": "prefilter_rejected",
            "candidate_route_sequences": [],
            "candidate_objective_key": [],
            "candidate_id": "0" * 64,
        }
    ]

    replay = _replay_record(
        ReviewRecord(source.path, payload),
        root / "data" / "schneider",
    )

    assert replay == {
        "valid": False,
        "reason": (
            "semantic trajectory replay failed: semantic candidate_id does not "
            "match its canonical route projection"
        ),
    }

    for missing_value in ("absent", None):
        missing_payload = dict(payload)
        if missing_value == "absent":
            del missing_payload["semantic_trajectory"]
        else:
            missing_payload["semantic_trajectory"] = None
        assert _replay_record(
            ReviewRecord(source.path, missing_payload),
            root / "data" / "schneider",
        ) == {
            "valid": False,
            "reason": (
                "semantic trajectory replay failed: v4 evidence is missing"
            ),
        }


def test_cuda_condition_does_not_reuse_exact_backend_occupancy(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    records = list(_review_fixture_records(root, tmp_path))
    host = next(record for record in records if record.mode.value == "host_scheduler")
    assert isinstance(host.payload, dict)
    host.payload["backend_metrics"] = {"launch_occupancies": [64]}

    condition = _scheduler_screening_occupancy((host,))

    assert condition == {
        "available": False,
        "condition_met": False,
        "maximum": None,
        "reason": "native candidate-screening occupancy is not recorded",
    }


def test_raw_axis_inventory_binds_json_and_sidecar_bytes(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    records = _review_fixture_records(root, tmp_path)
    before = _raw_axis_inventory(records)
    sidecar = records[0].path.with_suffix(records[0].path.suffix + ".sha256")
    sidecar.write_text(sidecar.read_text(encoding="ascii") + "\n", encoding="ascii")

    after = _raw_axis_inventory(records)

    assert before["axis_count"] == len(MODES)
    assert before["tree_sha256"] != after["tree_sha256"]


def test_review_writer_emits_hash_bound_review_manifest(tmp_path: Path) -> None:
    output_json = tmp_path / "paired_review.json"
    output_markdown = tmp_path / "paired_report.md"
    review = {
        "schema_version": "test-review-v1",
        "scope": "paired",
        "review_status": "NOT_READY",
        "reviewer_provenance": {
            "repository_revision": "a" * 40,
            "source_sha256": "b" * 64,
        },
        "producer_identity": {
            "repository_revisions": ["c" * 40],
            "wheel_sha256": ["d" * 64],
            "native_sha256": ["e" * 64],
            "scheduler_sha256": ["f" * 64],
            "run_labels": ["stage05.2_native_architecture_test_attempt01"],
        },
        "mode_metrics": {mode.value: {} for mode in MODES},
        "performance": {},
        "differential_gates": {},
        "historical_attempt72": {
            "available": False,
            "identity_verified": False,
            "comparison_count": 0,
        },
        "cuda_evaluation_condition": {
            "available": False,
            "condition_met": False,
            "maximum": None,
            "reason": "native candidate-screening occupancy is not recorded",
        },
        "axis_replay_passed": False,
        "axis_count": 0,
    }

    write_review(review, output_json=output_json, output_markdown=output_markdown)

    manifest = json.loads(
        (tmp_path / "paired_review_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["reviewer_provenance"] == review["reviewer_provenance"]
    assert manifest["producer_identity"] == review["producer_identity"]
    assert manifest["files"][output_json.name] == hashlib.sha256(
        output_json.read_bytes()
    ).hexdigest()
    assert manifest["files"][output_markdown.name] == hashlib.sha256(
        output_markdown.read_bytes()
    ).hexdigest()


def test_one_wall_clock_group_runs_all_five_modes_with_one_scheduler(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    endpoint = tmp_path / "scheduler.sock"
    task = ArchitectureAxisTask(
        scope="paired",
        repeat=0,
        axis="wall_clock_30",
        instance_name="c101C5",
        seed=2014,
        benchmark_dir=root / "data/schneider",
        output_root=tmp_path,
        run_labels=run_labels_for_scope("paired", 99),
        scheduler_socket_path=str(endpoint),
        wheel_sha256="a" * 64,
        native_sha256="b" * 64,
        scheduler_sha256="f" * 64,
        revision="c" * 40,
        initial_customer_sequences=_c5_warm_start(root),
        initial_solution_provenance=_c5_source_provenance(
            root, tmp_path / "wall-clock-source.json"
        ),
    )

    with NativeHostScheduler(endpoint):
        written = _run_group(task)

    assert len(written) == len(MODES)
    payloads = [json.loads(Path(path).read_bytes()) for path in written]
    assert {payload["mode"] for payload in payloads} == {mode.value for mode in MODES}
    assert {payload["scheduler_sha256"] for payload in payloads} == {"f" * 64}
    status_by_mode = {payload["mode"]: payload["status"] for payload in payloads}
    assert status_by_mode == {
        "current_stage052": "completed",
        "python_candidate_control": "completed",
        "per_solve_runtime": "completed",
        "full_native_alns": "failed",
        "host_scheduler": "failed",
    }
    for payload in payloads:
        if payload["mode"] in {
            "full_native_alns",
            "host_scheduler",
        }:
            assert payload["error"]
    assert all(payload["persistence_seconds"] > 0.0 for payload in payloads)
    assert not tuple(tmp_path.rglob("*.persistence-probe-*"))


def test_one_fixed_work_group_retains_every_mode_axis(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    endpoint = tmp_path / "scheduler.sock"
    task = ArchitectureAxisTask(
        scope="paired",
        repeat=0,
        axis="fixed_work",
        instance_name="c101C5",
        seed=2014,
        benchmark_dir=root / "data/schneider",
        output_root=tmp_path,
        run_labels=run_labels_for_scope("paired", 98),
        scheduler_socket_path=str(endpoint),
        wheel_sha256="a" * 64,
        native_sha256="b" * 64,
        scheduler_sha256="f" * 64,
        revision="c" * 40,
        initial_customer_sequences=_c5_warm_start(root),
        initial_solution_provenance=_c5_source_provenance(
            root, tmp_path / "fixed-work-source.json"
        ),
    )

    with NativeHostScheduler(endpoint):
        written = _run_group(task)

    payloads = [json.loads(Path(path).read_bytes()) for path in written]
    assert len(payloads) == len(MODES)
    assert {payload["mode"] for payload in payloads} == {mode.value for mode in MODES}
    status_by_mode = {payload["mode"]: payload["status"] for payload in payloads}
    assert status_by_mode == {
        "current_stage052": "completed",
        "python_candidate_control": "completed",
        "per_solve_runtime": "completed",
        "full_native_alns": "failed",
        "host_scheduler": "failed",
    }
