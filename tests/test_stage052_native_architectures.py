from __future__ import annotations

import copy
import gc
import hashlib
import json
import os
import pickle
import struct
import subprocess
import sys
import time
import tracemalloc
import zipfile
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from evrptw.alns import ALNSResult
from evrptw.cache_incremental import RouteCacheKey, canonical_instance_hash
from evrptw.charging import solve_exact_charging
from evrptw.experiments.stage052_native_architecture_review import (
    REVIEW_SCHEMA_VERSION,
    ReviewRecord,
    _axis_end_to_end_seconds,
    _axis_payload_with_persistence,
    _canonical_semantic_events,
    _common_prefix,
    _comparison_projection_from_selected,
    _comparison_semantic_events,
    _describe_first_divergence,
    _external_semantic_trajectory,
    _load_axis_record,
    _mode_wave_affinity_matches_profile,
    _mode_wave_metrics,
    _NativeCanonicalProjectionHasher,
    _raw_axis_inventory,
    _records_use_current_profile_schema,
    _relative_time_improvement,
    _replay_canonical_journal,
    _replay_initial_state_receipt,
    _replay_physical_screening,
    _replay_raw_native_control_journal,
    _replay_record,
    _review_build_attestation,
    _review_mode_wave_resources,
    _review_scheduler_runtime_statistics,
    _review_worker_terminal_io,
    _RowEvidenceAccumulator,
    _scheduler_screening_occupancy,
    _semantic_trajectory,
    _trajectory_row,
    load_records,
    render_report,
    review_records,
    write_review,
)
from evrptw.experiments.stage052_native_architectures import (
    CALIBRATION_REVIEW_SCHEMA_VERSION,
    LEGACY_PROFILE_COMPARISON_SCHEMA_VERSION,
    MODE_WAVE_RESOURCE_ACCOUNTING_SOURCE,
    MODES,
    PAIRED_INSTANCES,
    PAIRED_REVIEW_SCHEMA_VERSION,
    PREVIOUS_COMPARISON_SCHEMA_VERSION,
    PRIOR_PROFILE_COMPARISON_SCHEMA_VERSION,
    SCHEMA_VERSION,
    SEEDS,
    TICK_PROFILE_COMPARISON_SCHEMA_VERSION,
    WARM_START_SCHEMA_VERSION,
    ArchitectureAxisExecutionFailed,
    ArchitectureAxisTask,
    ArchitectureAxisWorkerFailed,
    ArchitectureMode,
    _assert_axis_publication_namespace_empty,
    _assign_performance_topology,
    _canonical_semantic_event_sequence,
    _canonical_trace_event,
    _cgroup_counter_delta,
    _cgroup_io_deltas,
    _iter_semantic_candidate_trajectory,
    _load_qualified_calibration_review,
    _load_qualified_paired_review,
    _performance_identity_blocks,
    _persist_axis_terminal_transaction,
    _reconcile_mode_wave_terminal_io,
    _require_campaign_identity,
    _require_native_architecture_capabilities,
    _run_fresh_spawn_axis_batch,
    _run_group,
    _run_mode_with_terminal_io,
    _runtime_cgroup_snapshot,
    _runtime_io_accounting,
    _semantic_candidate_trajectory,
    _solve_mode,
    _validate_native_build_attestation,
    _verify_installed_project_files,
    _write_signed_json,
    _write_signed_json_with_receipt,
    build_axis_plan,
    expected_axis_count,
    load_warm_start_bundle,
    performance_family_for_instance,
    rotated_modes,
    run_experiment,
    run_labels_for_scope,
)
from evrptw.experiments.stage052_performance_calibration_review import (
    CALIBRATION_REVIEW_SCHEMA_VERSION as PRODUCER_CALIBRATION_REVIEW_SCHEMA_VERSION,
)
from evrptw.measurement import canonical_route_key
from evrptw.native_scheduler import NativeHostScheduler
from evrptw.neighborhoods import screen_route_candidate
from evrptw.objective import SolutionObjective
from evrptw.parser import parse_schneider
from evrptw.stage052_performance import ExecutionTopology
from evrptw.stage052_semantic_journal import (
    evidence_json_value,
    iter_canonical_semantic_events,
    iter_verified_native_control_events,
    iter_verified_semantic_journal,
    write_semantic_journal,
)
from evrptw.validation import validate_routes
from evrptw.warm_start import canonical_customer_sequences_sha256
from tools.native_build_attestation import (
    committed_source_attestation,
    committed_wheel_project_entries,
    committed_wheel_project_entry_sha256,
)


def _spawn_axis_with_stuck_descendant(
    connection: object,
    task: ArchitectureAxisTask,
    _mode: ArchitectureMode,
) -> None:
    os.setsid()
    process_group_id = os.getpgrp()
    cast(Any, connection).send(
        {"status": "started", "process_group_id": process_group_id}
    )
    descendant = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        close_fds=True,
    )
    (task.output_root / "stuck-descendant.pid").write_text(
        str(descendant.pid),
        encoding="ascii",
    )
    time.sleep(60)


def _spawn_axis_failure_or_hang(
    connection: object,
    task: ArchitectureAxisTask,
    _mode: ArchitectureMode,
) -> None:
    os.setsid()
    process_group_id = os.getpgrp()
    cast(Any, connection).send(
        {"status": "started", "process_group_id": process_group_id}
    )
    if task.seed == 2014:
        cast(Any, connection).send(
            {
                "status": "worker_failed",
                "error_type": "InjectedAxisFailure",
                "error": "injected first-axis failure",
            }
        )
        cast(Any, connection).close()
        return
    time.sleep(60)


def test_signed_json_writer_is_durable_atomic_and_non_overwriting(
    tmp_path: Path,
) -> None:
    path = tmp_path / "axis.json"
    observed_bytes, receipt = _write_signed_json_with_receipt(
        path,
        {"value": 1},
    )

    data = path.read_bytes()
    sidecar = path.with_suffix(".json.sha256")
    assert observed_bytes == len(data) + sidecar.stat().st_size
    assert sidecar.read_text(encoding="ascii").strip() == hashlib.sha256(data).hexdigest()
    assert receipt["schema_version"] == "stage05.2-signed-json-persistence-v2"
    assert receipt["data_bytes"] == len(data)
    assert receipt["sidecar_bytes"] == sidecar.stat().st_size
    assert receipt["write_chunk_bytes"] == 512 * 1024
    assert receipt["write_batch_count"] == 1
    assert receipt["maximum_write_batch_bytes"] == len(data)
    for field_name in (
        "encoding_seconds",
        "hash_seconds",
        "write_seconds",
        "fsync_seconds",
        "atomic_publish_seconds",
        "total_seconds",
    ):
        assert float(receipt[field_name]) >= 0.0
    assert not tuple(tmp_path.glob("*.tmp-*"))
    with pytest.raises(FileExistsError, match="already exists"):
        _write_signed_json_with_receipt(path, {"value": 2})

    large_path = tmp_path / "large-axis.json"
    large_bytes, large_receipt = _write_signed_json_with_receipt(
        large_path,
        {"value": "x" * (1024 * 1024)},
    )
    assert large_bytes == (
        large_path.stat().st_size + large_path.with_suffix(".json.sha256").stat().st_size
    )
    assert large_receipt["write_batch_count"] == 3
    assert large_receipt["maximum_write_batch_bytes"] == 512 * 1024


def test_external_semantic_trajectory_replays_operator_journal_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from evrptw.experiments import stage052_native_architecture_review as review_module

    operator_event = {
        "semantic_stream": "operator",
        "runtime_event_id": 3,
        "semantic_event_id": 2,
        "semantic_sequence": 1,
        "stream_ordinal": 0,
        "iteration": 4,
        "operator": "relocate",
        "status": "candidate_proposed",
        "candidate_route_sequences": [["C1", "C2"]],
        "candidate_objective_key": [1, 10.0, 0.0, 0],
        "_operator_native_candidate_feasible": True,
        "native_telemetry": {"runtime_native_lane_id": 7},
    }
    aggregate_event = {
        **operator_event,
        "semantic_event_id": 3,
        "status": "pair_prefilter_rejected_aggregate",
        "aggregate_count": 17,
    }
    expected = _trajectory_row(operator_event, 0)
    evidence = _RowEvidenceAccumulator()
    evidence.append(expected)
    descriptor = {
        "schema_version": "stage05.2-external-semantic-trajectory-v1",
        "source": "canonical_semantic_journal:operator",
        **evidence.receipt(),
    }
    monkeypatch.setattr(
        review_module,
        "iter_verified_semantic_journal",
        lambda _path, _descriptor: iter((operator_event, aggregate_event)),
    )

    assert _external_semantic_trajectory(
        tmp_path / "axis.json",
        {"canonical_semantic_journal": {"schema_version": "test"}},
        descriptor,
    ) == [expected]
    with pytest.raises(RuntimeError, match="digest does not reconcile"):
        _external_semantic_trajectory(
            tmp_path / "axis.json",
            {"canonical_semantic_journal": {"schema_version": "test"}},
            {**descriptor, "sha256": "0" * 64},
        )


def test_run_mode_publishes_failed_axis_then_raises_fail_fast(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from evrptw.experiments import stage052_native_architectures as architectures

    def fail_solve(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("injected axis failure")

    monkeypatch.setattr(architectures, "_solve_mode", fail_solve)
    labels = run_labels_for_scope("paired", 99)
    task = ArchitectureAxisTask(
        scope="paired",
        repeat=0,
        axis="fixed_work",
        instance_name="c101C5",
        seed=2014,
        benchmark_dir=tmp_path,
        output_root=tmp_path,
        run_labels=labels,
        scheduler_socket_path=str(tmp_path / "scheduler.sock"),
        wheel_sha256="a" * 64,
        native_sha256="b" * 64,
        scheduler_sha256="c" * 64,
        revision="d" * 40,
        initial_customer_sequences=(),
        initial_solution_provenance={},
    )

    with pytest.raises(ArchitectureAxisExecutionFailed) as captured:
        architectures._run_mode(task, ArchitectureMode.CURRENT_STAGE052)

    axis_path = Path(captured.value.axis_path)
    assert axis_path.is_file()
    assert json.loads(axis_path.read_text(encoding="utf-8"))["status"] == "failed"
    assert axis_path.with_suffix(".json.sha256").is_file()
    assert axis_path.with_suffix(".json.persistence").is_file()


def test_run_mode_releases_solved_trace_when_resource_summary_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from evrptw.experiments import stage052_native_architectures as architectures

    released = False

    class Trace:
        def release_runtime_semantic_storage(self) -> None:
            nonlocal released
            released = True

    solved_result = SimpleNamespace(measurement_trace=Trace())

    def fail_after_solve(
        *_args: object,
        solved_result_sink: object = None,
        **_kwargs: object,
    ) -> object:
        assert callable(solved_result_sink)
        solved_result_sink(solved_result)
        raise RuntimeError("injected resource-summary failure")

    monkeypatch.setattr(architectures, "_solve_mode", fail_after_solve)
    task = ArchitectureAxisTask(
        scope="paired",
        repeat=0,
        axis="fixed_work",
        instance_name="c101C5",
        seed=2014,
        benchmark_dir=tmp_path,
        output_root=tmp_path,
        run_labels=run_labels_for_scope("paired", 98),
        scheduler_socket_path=str(tmp_path / "scheduler.sock"),
        wheel_sha256="a" * 64,
        native_sha256="b" * 64,
        scheduler_sha256="c" * 64,
        revision="d" * 40,
        initial_customer_sequences=(),
        initial_solution_provenance={},
    )

    with pytest.raises(ArchitectureAxisExecutionFailed, match="resource-summary failure"):
        architectures._run_mode(task, ArchitectureMode.CURRENT_STAGE052)

    assert released is True


def test_axis_failure_is_pickle_safe_for_process_pool_transport() -> None:
    original = ArchitectureAxisExecutionFailed(
        "/evidence/failed-axis.json",
        "RuntimeError",
        "injected failure",
    )

    restored = pickle.loads(pickle.dumps(original))

    assert isinstance(restored, ArchitectureAxisExecutionFailed)
    assert restored.axis_path == original.axis_path
    assert restored.error_type == original.error_type
    assert restored.error == original.error
    assert str(restored) == str(original)


def test_fresh_spawn_axis_batch_times_out_and_terminates_stalled_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from evrptw.experiments import stage052_native_architectures as architectures

    clock = [100.0]

    class FakeConnection:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    class FakeProcess:
        def __init__(self) -> None:
            self.daemon = True
            self.pid = 98765
            self.exitcode: int | None = None
            self.terminated = False
            self.killed = False
            self.join_timeouts: list[float | None] = []

        def start(self) -> None:
            return

        def is_alive(self) -> bool:
            return self.exitcode is None

        def terminate(self) -> None:
            self.terminated = True
            self.exitcode = -15

        def kill(self) -> None:
            self.killed = True
            self.exitcode = -9

        def join(self, timeout: float | None = None) -> None:
            self.join_timeouts.append(timeout)

    class FakeContext:
        def __init__(self) -> None:
            self.processes: list[FakeProcess] = []

        def Pipe(self, *, duplex: bool) -> tuple[FakeConnection, FakeConnection]:
            assert duplex is False
            return FakeConnection(), FakeConnection()

        def Process(self, **_kwargs: object) -> FakeProcess:
            process = FakeProcess()
            self.processes.append(process)
            return process

    context = FakeContext()

    def bounded_wait(
        _connections: tuple[object, ...],
        timeout: float | None = None,
    ) -> list[object]:
        assert timeout is not None and timeout > 0.0, "spawn wait must be bounded"
        clock[0] += timeout
        return []

    monkeypatch.setattr(architectures.multiprocessing, "get_context", lambda _name: context)
    monkeypatch.setattr(architectures, "wait_for_connections", bounded_wait)
    monkeypatch.setattr(architectures.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(
        architectures,
        "_axis_parent_timeout_seconds",
        lambda _task: 0.25,
        raising=False,
    )
    task = ArchitectureAxisTask(
        scope="paired",
        repeat=0,
        axis="fixed_work",
        instance_name="c101C5",
        seed=2014,
        benchmark_dir=tmp_path,
        output_root=tmp_path,
        run_labels=run_labels_for_scope("paired", 97),
        scheduler_socket_path=str(tmp_path / "scheduler.sock"),
        wheel_sha256="a" * 64,
        native_sha256="b" * 64,
        scheduler_sha256="c" * 64,
        revision="d" * 40,
        initial_customer_sequences=(),
        initial_solution_provenance={},
    )

    (outcome,) = _run_fresh_spawn_axis_batch(
        (task,),
        ArchitectureMode.CURRENT_STAGE052,
        scheduler_process_id=None,
    )

    assert isinstance(outcome.error, ArchitectureAxisWorkerFailed)
    assert outcome.error.error_type == "WorkerTimeout"
    assert context.processes[0].terminated is True
    assert context.processes[0].join_timeouts


def test_fresh_spawn_axis_batch_reaps_stalled_descendant_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from evrptw.experiments import stage052_native_architectures as architectures

    monkeypatch.setattr(
        architectures,
        "_fresh_axis_process_entry",
        _spawn_axis_with_stuck_descendant,
    )
    monkeypatch.setattr(
        architectures,
        "_axis_parent_timeout_seconds",
        lambda _task: 2.0,
    )
    task = ArchitectureAxisTask(
        scope="paired",
        repeat=0,
        axis="fixed_work",
        instance_name="c101C5",
        seed=2014,
        benchmark_dir=tmp_path,
        output_root=tmp_path,
        run_labels=run_labels_for_scope("paired", 96),
        scheduler_socket_path=str(tmp_path / "scheduler.sock"),
        wheel_sha256="a" * 64,
        native_sha256="b" * 64,
        scheduler_sha256="c" * 64,
        revision="d" * 40,
        initial_customer_sequences=(),
        initial_solution_provenance={},
    )

    (outcome,) = _run_fresh_spawn_axis_batch(
        (task,),
        ArchitectureMode.CURRENT_STAGE052,
        scheduler_process_id=None,
    )

    assert isinstance(outcome.error, ArchitectureAxisWorkerFailed)
    assert outcome.error.error_type == "WorkerTimeout"
    descendant_pid = int((tmp_path / "stuck-descendant.pid").read_text(encoding="ascii"))
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        try:
            raw_stat = Path(f"/proc/{descendant_pid}/stat").read_text(encoding="ascii")
        except FileNotFoundError:
            break
        state = raw_stat[raw_stat.rfind(")") + 2 :].split()[0]
        if state == "Z":
            break
        time.sleep(0.01)
    else:
        pytest.fail("stalled spawned-axis descendant remained active")


def test_fresh_spawn_axis_batch_cancels_sibling_on_first_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from evrptw.experiments import stage052_native_architectures as architectures

    monkeypatch.setattr(
        architectures,
        "_fresh_axis_process_entry",
        _spawn_axis_failure_or_hang,
    )
    monkeypatch.setattr(
        architectures,
        "_axis_parent_timeout_seconds",
        lambda _task: 20.0,
    )

    def task(seed: int) -> ArchitectureAxisTask:
        return ArchitectureAxisTask(
            scope="paired",
            repeat=0,
            axis="fixed_work",
            instance_name="c101C5",
            seed=seed,
            benchmark_dir=tmp_path,
            output_root=tmp_path,
            run_labels=run_labels_for_scope("paired", 95),
            scheduler_socket_path=str(tmp_path / f"scheduler-{seed}.sock"),
            wheel_sha256="a" * 64,
            native_sha256="b" * 64,
            scheduler_sha256="c" * 64,
            revision="d" * 40,
            initial_customer_sequences=(),
            initial_solution_provenance={},
        )

    started = time.monotonic()
    outcomes = _run_fresh_spawn_axis_batch(
        (task(2014), task(2015)),
        ArchitectureMode.CURRENT_STAGE052,
        scheduler_process_id=None,
    )
    elapsed = time.monotonic() - started

    assert elapsed < 10.0
    assert outcomes[0].task.seed == 2014
    assert isinstance(outcomes[0].error, ArchitectureAxisWorkerFailed)
    assert outcomes[0].error.error_type == "InjectedAxisFailure"
    assert isinstance(outcomes[1].error, ArchitectureAxisWorkerFailed)
    assert outcomes[1].error.error_type == "SiblingCancelled"


def test_spawned_axis_batch_cleanup_uses_shared_signal_deadlines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from evrptw.experiments import stage052_native_architectures as architectures

    class ResistantProcess:
        def __init__(self, pid: int) -> None:
            self.pid = pid
            self.terminated = False
            self.killed = False
            self.alive = True

        def is_alive(self) -> bool:
            return self.alive

        def terminate(self) -> None:
            self.terminated = True

        def kill(self) -> None:
            self.killed = True
            self.alive = False

        def join(self, timeout: float | None = None) -> None:
            assert timeout == 0.0

    processes = tuple(ResistantProcess(900_000 + index) for index in range(8))
    monkeypatch.setattr(architectures, "AXIS_PROCESS_EXIT_GRACE_SECONDS", 0.05)
    monkeypatch.setattr(architectures, "_process_group_is_alive", lambda _group: False)

    started = time.monotonic()
    states = architectures._stop_spawned_axis_processes(  # noqa: SLF001
        tuple(
            (index, cast(Any, process), process.pid)
            for index, process in enumerate(processes)
        )
    )
    elapsed = time.monotonic() - started

    assert all(states.values())
    assert all(process.terminated and process.killed for process in processes)
    assert elapsed < 0.2


def test_spawned_axis_envelope_read_obeys_parent_deadline() -> None:
    from evrptw.experiments import stage052_native_architectures as architectures

    class PartialEnvelope:
        def recv(self) -> object:
            time.sleep(1.0)
            return {"status": "completed"}

    started = time.perf_counter()
    with pytest.raises(TimeoutError, match="result envelope timed out"):
        architectures._receive_axis_envelope(  # type: ignore[arg-type]  # noqa: SLF001
            PartialEnvelope(),
            deadline=started + 0.05,
        )
    assert time.perf_counter() - started < 0.2


def test_spawned_axis_batch_receive_uses_earliest_pending_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from evrptw.experiments import stage052_native_architectures as architectures

    clock = [100.0]

    class FakeConnection:
        def __init__(self, ordinal: int) -> None:
            self.ordinal = ordinal

        def close(self) -> None:
            return

    class FakeProcess:
        def __init__(self, pid: int) -> None:
            self.pid = pid
            self.exitcode: int | None = None

        def start(self) -> None:
            return

        def is_alive(self) -> bool:
            return self.exitcode is None

        def terminate(self) -> None:
            self.exitcode = -15

        def kill(self) -> None:
            self.exitcode = -9

        def join(self, timeout: float | None = None) -> None:
            assert timeout == 0.0

    class FakeContext:
        def __init__(self) -> None:
            self.parents: list[FakeConnection] = []
            self.processes: list[FakeProcess] = []

        def Pipe(self, *, duplex: bool) -> tuple[FakeConnection, FakeConnection]:
            assert duplex is False
            ordinal = len(self.parents)
            parent = FakeConnection(ordinal)
            self.parents.append(parent)
            return parent, FakeConnection(ordinal)

        def Process(self, **_kwargs: object) -> FakeProcess:
            process = FakeProcess(90_000 + len(self.processes))
            self.processes.append(process)
            return process

    context = FakeContext()
    wait_call = 0

    def wait(
        _connections: tuple[object, ...],
        timeout: float | None = None,
    ) -> list[object]:
        nonlocal wait_call
        wait_call += 1
        assert timeout is not None and timeout > 0.0
        return list(context.parents) if wait_call == 1 else [context.parents[1]]

    observed_deadlines: list[tuple[int, float]] = []
    receives: Counter[int] = Counter()

    def receive(connection: FakeConnection, *, deadline: float) -> object:
        observed_deadlines.append((connection.ordinal, deadline))
        receives[connection.ordinal] += 1
        if receives[connection.ordinal] == 1:
            return {
                "status": "started",
                "process_group_id": 90_000 + connection.ordinal,
            }
        clock[0] = 100.75
        return {
            "status": "worker_failed",
            "error_type": "InjectedFailure",
            "error": "stop",
        }

    monkeypatch.setattr(architectures.multiprocessing, "get_context", lambda _name: context)
    monkeypatch.setattr(architectures, "wait_for_connections", wait)
    monkeypatch.setattr(architectures, "_receive_axis_envelope", receive)
    monkeypatch.setattr(architectures.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(
        architectures,
        "_axis_parent_timeout_seconds",
        lambda task: 1.0 if task.seed == 2014 else 10.0,
    )

    def task(seed: int) -> ArchitectureAxisTask:
        return ArchitectureAxisTask(
            scope="paired",
            repeat=0,
            axis="fixed_work",
            instance_name="c101C5",
            seed=seed,
            benchmark_dir=tmp_path,
            output_root=tmp_path,
            run_labels=run_labels_for_scope("paired", 94),
            scheduler_socket_path=str(tmp_path / f"scheduler-{seed}.sock"),
            wheel_sha256="a" * 64,
            native_sha256="b" * 64,
            scheduler_sha256="c" * 64,
            revision="d" * 40,
            initial_customer_sequences=(),
            initial_solution_provenance={},
        )

    outcomes = _run_fresh_spawn_axis_batch(
        (task(2014), task(2015)),
        ArchitectureMode.CURRENT_STAGE052,
        scheduler_process_id=None,
    )

    second_terminal_deadline = observed_deadlines[-1]
    assert second_terminal_deadline == (1, 101.0)
    assert outcomes[0].parent_terminal_seconds == pytest.approx(0.75)


def test_run_mode_with_terminal_io_binds_task_start_and_axis_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from evrptw.experiments import stage052_native_architectures as architectures

    task = cast(ArchitectureAxisTask, SimpleNamespace())
    receipt = {
        "schema_version": "stage05.2-terminal-process-io-v1",
        "pid": 11,
        "parent_pid": 10,
        "create_time": 1.0,
        "start_time_ticks": 101,
        "task_started_monotonic": 20.0,
        "captured_monotonic": 21.0,
        "read_bytes": 3,
        "write_bytes": 5,
    }
    monkeypatch.setattr(architectures.time, "monotonic", lambda: 20.0)
    monkeypatch.setattr(
        architectures,
        "_run_mode",
        lambda observed_task, mode: (
            "/evidence/axis.json"
            if observed_task is task and mode is ArchitectureMode.CURRENT_STAGE052
            else pytest.fail("terminal-I/O wrapper changed the axis task identity")
        ),
    )

    def capture(*, task_started_monotonic: float) -> dict[str, object]:
        assert task_started_monotonic == 20.0
        return receipt

    monkeypatch.setattr(
        architectures,
        "capture_current_process_terminal_io_receipt",
        capture,
    )

    assert _run_mode_with_terminal_io(
        task,
        ArchitectureMode.CURRENT_STAGE052,
    ) == ("/evidence/axis.json", [receipt])


def test_mode_wave_reconciles_core_terminal_io_separately_from_axis_binding() -> None:
    applied: list[dict[str, object]] = []

    class Monitor:
        def apply_terminal_process_io_receipts(
            self,
            receipts: Iterable[dict[str, object]],
        ) -> None:
            applied.extend(receipts)

    core = {
        "schema_version": "stage05.2-terminal-process-io-v1",
        "pid": 11,
        "parent_pid": 10,
        "create_time": 1.0,
        "start_time_ticks": 101,
        "task_started_monotonic": 20.0,
        "captured_monotonic": 21.0,
        "read_bytes": 3,
        "write_bytes": 5,
    }
    bound = {
        "repeat": 1,
        "axis": "fixed_work",
        "instance": "c101C5",
        "seed": 2014,
        **core,
    }

    _reconcile_mode_wave_terminal_io(
        cast(Any, Monitor()),
        terminal_io_receipts=[core],
        bound_terminal_io_receipts=[bound],
        scheduler_terminal_io_receipts=[],
        expected_count=1,
    )

    assert applied == [core]
    assert set(applied[0]) == {
        "schema_version",
        "pid",
        "parent_pid",
        "create_time",
        "start_time_ticks",
        "task_started_monotonic",
        "captured_monotonic",
        "read_bytes",
        "write_bytes",
    }
    with pytest.raises(RuntimeError, match="inventory is incomplete"):
        _reconcile_mode_wave_terminal_io(
            cast(Any, Monitor()),
            terminal_io_receipts=[core],
            bound_terminal_io_receipts=[],
            scheduler_terminal_io_receipts=[],
            expected_count=1,
        )


def test_axis_end_to_end_uses_parent_terminal_and_independent_replay() -> None:
    identity = {
        "repeat": 1,
        "axis": "fixed_work",
        "instance": "c101_21",
        "seed": 2014,
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        **identity,
        "producer_pre_receipt_seconds": 2.0,
    }
    timing = {
        **identity,
        "subwave_index": 0,
        "producer_parent_terminal_seconds": 2.4,
    }
    record = ReviewRecord(
        Path("axis.json"),
        payload,
        {"axis_parent_terminal_timings": [timing]},
    )

    assert _axis_end_to_end_seconds(record, 0.3) == pytest.approx(2.7)

    timing["producer_parent_terminal_seconds"] = 1.9
    with pytest.raises(ValueError, match="omits producer publication"):
        _axis_end_to_end_seconds(record, 0.3)


def test_axis_terminal_publication_failure_rolls_back_every_created_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    axis_path = tmp_path / "axis.json"
    bundle = axis_path.with_suffix(".semantic.bundle")
    bundle.mkdir()
    (bundle / "events.jsonl.gz").write_bytes(b"journal")
    task_receipt = axis_path.with_suffix(axis_path.suffix + ".native-work-tasks.jsonl")
    task_receipt.write_bytes(b"task\n")
    task_sidecar = task_receipt.with_suffix(task_receipt.suffix + ".sha256")
    task_sidecar.write_text("0" * 64 + "\n", encoding="ascii")
    semantic = {
        "path": bundle.name,
        "bundle_bytes": sum(path.stat().st_size for path in bundle.iterdir()),
        "persistence": {"formal_attributed_seconds": 0.1},
    }

    def fail_receipt(_path: Path, _payload: object) -> int:
        raise OSError("injected terminal receipt failure")

    monkeypatch.setattr(
        "evrptw.experiments.stage052_native_architectures._write_signed_json",
        fail_receipt,
    )
    with pytest.raises(OSError, match="injected terminal receipt failure"):
        _persist_axis_terminal_transaction(
            path=axis_path,
            payload={"status": "completed"},
            semantic_journal=semantic,
            task_receipt_path=task_receipt,
            journal_persistence_seconds=0.1,
            journal_persistence_wall_seconds=0.1,
            startup_seconds=0.0,
            axis_started=0.0,
            created_targets={bundle, task_receipt, task_sidecar},
        )

    persistence = axis_path.with_suffix(axis_path.suffix + ".persistence")
    assert not bundle.exists()
    assert not task_receipt.exists()
    assert not task_sidecar.exists()
    assert not axis_path.exists()
    assert not axis_path.with_suffix(axis_path.suffix + ".sha256").exists()
    assert not persistence.exists()
    assert not persistence.with_suffix(persistence.suffix + ".sha256").exists()


def test_axis_terminal_collision_preserves_preexisting_evidence(
    tmp_path: Path,
) -> None:
    axis_path = tmp_path / "axis.json"
    axis_path.write_text("immutable-axis\n", encoding="utf-8")
    axis_sidecar = axis_path.with_suffix(axis_path.suffix + ".sha256")
    axis_sidecar.write_text("f" * 64 + "\n", encoding="ascii")
    bundle = axis_path.with_suffix(".semantic.bundle")
    bundle.mkdir()
    (bundle / "events.jsonl.gz").write_bytes(b"new-journal")
    task_receipt = axis_path.with_suffix(axis_path.suffix + ".native-work-tasks.jsonl")
    task_receipt.write_bytes(b"new-task\n")
    task_sidecar = task_receipt.with_suffix(task_receipt.suffix + ".sha256")
    task_sidecar.write_text("0" * 64 + "\n", encoding="ascii")
    semantic = {
        "path": bundle.name,
        "bundle_bytes": sum(path.stat().st_size for path in bundle.iterdir()),
        "persistence": {"formal_attributed_seconds": 0.1},
    }

    with pytest.raises(FileExistsError, match="already exists"):
        _persist_axis_terminal_transaction(
            path=axis_path,
            payload={"status": "completed"},
            semantic_journal=semantic,
            task_receipt_path=task_receipt,
            journal_persistence_seconds=0.1,
            journal_persistence_wall_seconds=0.1,
            startup_seconds=0.0,
            axis_started=0.0,
            created_targets={bundle, task_receipt, task_sidecar},
        )

    assert axis_path.read_text(encoding="utf-8") == "immutable-axis\n"
    assert axis_sidecar.read_text(encoding="ascii") == "f" * 64 + "\n"
    assert not bundle.exists()
    assert not task_receipt.exists()
    assert not task_sidecar.exists()


def test_axis_namespace_preflight_preserves_preexisting_ancillary_evidence(
    tmp_path: Path,
) -> None:
    axis_path = tmp_path / "axis.json"
    bundle = axis_path.with_suffix(".semantic.bundle")
    bundle.mkdir()
    marker = bundle / "immutable"
    marker.write_text("journal\n", encoding="utf-8")
    task_receipt = axis_path.with_suffix(axis_path.suffix + ".native-work-tasks.jsonl")
    task_receipt.write_text("task\n", encoding="utf-8")
    task_sidecar = task_receipt.with_suffix(task_receipt.suffix + ".sha256")
    task_sidecar.write_text("0" * 64 + "\n", encoding="ascii")

    with pytest.raises(FileExistsError, match="namespace is not empty"):
        _assert_axis_publication_namespace_empty(
            axis_path,
            task_receipt_path=task_receipt,
        )

    assert marker.read_text(encoding="utf-8") == "journal\n"
    assert task_receipt.read_text(encoding="utf-8") == "task\n"
    assert task_sidecar.read_text(encoding="ascii") == "0" * 64 + "\n"


def test_performance_gate_uses_fractional_time_saved_not_speedup() -> None:
    assert _relative_time_improvement(85.0, 100.0) == pytest.approx(0.15)
    assert _relative_time_improvement(86.0, 100.0) == pytest.approx(0.14)
    assert 100.0 / 86.0 - 1.0 > 0.15
    with pytest.raises(ValueError, match="positive finite"):
        _relative_time_improvement(0.0, 100.0)


def test_runtime_cgroup_snapshot_records_memory_io_and_event_deltas(
    tmp_path: Path,
) -> None:
    cgroup_root = tmp_path / "cgroup"
    scope = cgroup_root / "stage052"
    scope.mkdir(parents=True)
    membership = tmp_path / "self.cgroup"
    membership.write_text("0::/stage052\n", encoding="ascii")
    (scope / "memory.current").write_text("100\n", encoding="ascii")
    (scope / "memory.peak").write_text("200\n", encoding="ascii")
    (scope / "memory.swap.current").write_text("0\n", encoding="ascii")
    (scope / "memory.swap.peak").write_text("0\n", encoding="ascii")
    (scope / "memory.events").write_text(
        "low 1\nhigh 2\nmax 3\noom 4\noom_kill 5\n",
        encoding="ascii",
    )
    (scope / "io.stat").write_text(
        "8:0 rbytes=10 wbytes=20 rios=1 wios=2 dbytes=3 dios=4\n"
        "8:16 rbytes=30 wbytes=40 rios=3 wios=4 dbytes=5 dios=6\n",
        encoding="ascii",
    )

    before = _runtime_cgroup_snapshot(
        cgroup_root=cgroup_root,
        membership_path=membership,
    )
    assert before["status"] == "available"
    assert before["memory_current_bytes"] == 100
    assert before["memory_peak_bytes"] == 200
    assert before["io"] == {
        "read_bytes": 40,
        "write_bytes": 60,
        "read_operations": 4,
        "write_operations": 6,
        "discard_bytes": 8,
        "discard_operations": 10,
    }

    (scope / "memory.events").write_text(
        "low 1\nhigh 2\nmax 3\noom 4\noom_kill 6\n",
        encoding="ascii",
    )
    after = _runtime_cgroup_snapshot(
        cgroup_root=cgroup_root,
        membership_path=membership,
    )
    assert _cgroup_counter_delta(before, after, "oom") == 0
    assert _cgroup_counter_delta(before, after, "oom_kill") == 1
    assert _cgroup_io_deltas(before, after) == {
        "read_bytes": 0,
        "write_bytes": 0,
        "read_operations": 0,
        "write_operations": 0,
        "discard_bytes": 0,
        "discard_operations": 0,
    }


def test_runtime_cgroup_snapshot_marks_missing_membership_unavailable(
    tmp_path: Path,
) -> None:
    snapshot = _runtime_cgroup_snapshot(
        cgroup_root=tmp_path / "missing-cgroup",
        membership_path=tmp_path / "missing-membership",
    )
    assert snapshot == {
        "status": "unavailable",
        "cgroup_path": "unavailable",
    }


def test_runtime_io_accounting_uses_process_tree_when_cgroup_io_is_unavailable() -> None:
    assert _runtime_io_accounting(
        {"io": "unavailable"},
        {"io": "unavailable"},
        {
            "process_tree_read_bytes": 123,
            "process_tree_write_bytes": 456,
            "process_io_terminal_status": "available",
            "process_io_uncovered_identities": [],
        },
    ) == {
        "source": "process_tree_proc_io",
        "read_bytes": 123,
        "write_bytes": 456,
    }

    with pytest.raises(RuntimeError, match="cgroup I/O accounting is invalid"):
        _runtime_io_accounting(
            {"io": {"read_bytes": 1}},
            {"io": "unavailable"},
            {"process_tree_read_bytes": 123, "process_tree_write_bytes": 456},
        )


def test_free_scheduler_topology_keeps_thread_budget_but_exposes_all_cpus() -> None:
    task = ArchitectureAxisTask(
        scope="paired",
        repeat=0,
        axis="fixed_work",
        instance_name="c101C5",
        seed=2014,
        benchmark_dir=Path("/tmp/benchmarks"),
        output_root=Path("/tmp/output"),
        run_labels={mode.value: mode.value for mode in MODES},
        scheduler_socket_path="/tmp/stage052.sock",
        wheel_sha256="a" * 64,
        native_sha256="b" * 64,
        scheduler_sha256="c" * 64,
        revision="d" * 40,
        initial_customer_sequences=(("C1",),),
        initial_solution_provenance={},
    )
    topology = ExecutionTopology(
        workload_class="c5",
        shards=((0, 1), (2, 3)),
        worker_count=2,
        request_threads=2,
        affinity_policy="free_scheduler",
    )
    assigned = _assign_performance_topology(
        task,
        topology,
        shard_index=0,
        profile_sha256="e" * 64,
        topology_key="per_solve_runtime:c5",
    )
    assert assigned.axis_cpu_ids == (0, 1, 2, 3)
    assert assigned.threads_per_shard == 2
    assert assigned.total_compute_threads == 4


def _scheduler_runtime_statistics_fixture(tmp_path: Path) -> dict[str, object]:
    histogram = [1, *([0] * 31)]
    request_queue: dict[str, object] = {
        "pending": 0,
        "peak_pending": 1,
        "queue_full_count": 0,
        "rejected_count": 0,
        "completed": 1,
        "total_wait_seconds": 0.0,
        "maximum_wait_seconds": 0.0,
        "total_service_seconds": 0.0,
        "maximum_service_seconds": 0.0,
        "wait_histogram": histogram,
        "service_histogram": histogram,
    }
    work_queue = {
        **request_queue,
        "active": 0,
        "peak_active": 1,
    }
    quantiles = {
        "wait_seconds": {"p50": 0.0, "p95": 0.0, "p99": 0.0},
        "service_seconds": {"p50": 0.0, "p95": 0.0, "p99": 0.0},
    }
    receipt_path = tmp_path / "scheduler-task-receipts.jsonl"
    task_row = (
        '{"kind":"task","task_sequence":0,"worker_index":0,'
        '"first_index":0,"last_index":1,'
        '"submitted_nanoseconds":1,"started_nanoseconds":2,'
        '"completed_nanoseconds":3}\n'
    ).encode("ascii")
    batch_sha256 = hashlib.sha256(task_row).hexdigest()
    receipt_data = task_row + (
        '{"kind":"batch","batch_ordinal":0,"row_count":1,'
        '"first_task_sequence":0,"last_task_sequence":0,'
        f'"sha256":"{batch_sha256}"}}\n'
        '{"kind":"trailer",'
        '"schema_version":"stage05.2-native-work-task-receipts-v3",'
        '"storage_model":"bounded_async_fifo_stream",'
        '"receipt_batch_capacity":4096,"queue_bound_batches":1,'
        '"submitted_batches":1,"completed_batches":1,'
        '"task_receipt_dropped_count":0,'
        '"completed_tasks":1,"receipt_count":1}\n'
    ).encode("ascii")
    receipt_path.write_bytes(receipt_data)
    receipt_sha256 = hashlib.sha256(receipt_data).hexdigest()
    receipt_sidecar = receipt_path.with_suffix(receipt_path.suffix + ".sha256")
    receipt_sidecar.write_text(receipt_sha256 + "\n", encoding="ascii")
    return {
        "schema_version": "stage05.2-native-scheduler-runtime-v3",
        "worker_threads": 2,
        "request_threads": 1,
        "receipt_writer_threads": 1,
        "peak_active_requests": 1,
        "peak_distinct_client_pids": 1,
        "request_queue": request_queue,
        "work_queue": work_queue,
        "latency_quantiles": {
            "request_queue": quantiles,
            "work_queue": quantiles,
        },
        "task_receipts": {
            "schema_version": "stage05.2-native-work-task-receipts-v3",
            "path": receipt_path.name,
            "sha256": receipt_sha256,
            "bytes": len(receipt_data),
            "count": 1,
            "storage_model": "bounded_async_fifo_stream",
            "receipt_batch_capacity": 4_096,
            "queue_bound_batches": 1,
            "peak_queued_batches": 1,
            "submitted_batches": 1,
            "completed_batches": 1,
            "dropped_count": 0,
            "producer_wait_seconds": 0.0,
            "writer_wall_seconds": 0.0,
            "writer_cpu_seconds": 0.0,
            "serialization_seconds": 0.0,
            "write_seconds": 0.0,
            "file_fsync_seconds": 0.0,
            "atomic_publish_seconds": 0.0,
            "parent_fsync_seconds": 0.0,
            "sidecar_path": receipt_sidecar.name,
            "sidecar_sha256": hashlib.sha256(receipt_sidecar.read_bytes()).hexdigest(),
            "validated_worker_indices": [0],
        },
    }


def test_reviewer_independently_replays_scheduler_queue_statistics(tmp_path: Path) -> None:
    statistics = _scheduler_runtime_statistics_fixture(tmp_path)
    _review_scheduler_runtime_statistics(
        statistics,
        worker_threads=2,
        request_threads=1,
        evidence_root=tmp_path,
    )

    request_queue = statistics["request_queue"]
    assert isinstance(request_queue, dict)
    request_queue["queue_full_count"] = 1
    with pytest.raises(RuntimeError, match="queue resource gate"):
        _review_scheduler_runtime_statistics(
            statistics,
            worker_threads=2,
            request_threads=1,
            evidence_root=tmp_path,
        )


def test_reviewer_independently_replays_cgroup_resource_gate() -> None:
    io_before = {
        "read_bytes": 10,
        "write_bytes": 20,
        "read_operations": 1,
        "write_operations": 2,
        "discard_bytes": 0,
        "discard_operations": 0,
    }
    io_after = {
        "read_bytes": 15,
        "write_bytes": 30,
        "read_operations": 2,
        "write_operations": 4,
        "discard_bytes": 0,
        "discard_operations": 0,
    }
    wave: dict[str, object] = {
        "worker_process_lifecycle": "one_shard_per_spawned_process",
        "worker_multiprocessing_start_method": "spawn",
        "worker_max_tasks_per_child": 1,
        "cgroup_before": {
            "status": "available",
            "cgroup_path": "/stage052",
            "memory_current_bytes": 100,
            "memory_peak_bytes": 200,
            "memory_swap_current_bytes": 0,
            "memory_swap_peak_bytes": 0,
            "memory_events": {"oom": 0, "oom_kill": 0},
            "io": io_before,
        },
        "cgroup_after": {
            "status": "available",
            "cgroup_path": "/stage052",
            "memory_current_bytes": 150,
            "memory_peak_bytes": 250,
            "memory_swap_current_bytes": 0,
            "memory_swap_peak_bytes": 0,
            "memory_events": {"oom": 0, "oom_kill": 0},
            "io": io_after,
        },
        "cgroup_memory_event_deltas": {"oom": 0, "oom_kill": 0},
        "io_accounting": {
            "source": "cgroup_v2_io_stat",
            "read_bytes": 5,
            "write_bytes": 10,
            "read_operations": 1,
            "write_operations": 2,
            "discard_bytes": 0,
            "discard_operations": 0,
        },
        "memory_gate_bytes": 1_000,
        "peak_aggregate_rss_bytes": 300,
        "peak_aggregate_pss_bytes": 250,
        "axis_count": 2,
        "elapsed_seconds": 1.0,
        "axes_per_hour": 7_200.0,
        "effective_cores": 1.5,
        "cpu_utilization_fraction_of_compute_limit": 0.75,
        "compute_thread_limit": 2,
        "cpu_clock_tick_hz": 100,
        "cpu_quantization_lane_count": 2,
        "peak_concurrent_processes": 1,
        "monitor_start_monotonic": 0.0,
        "monitor_end_monotonic": 1.0,
        "effective_elapsed_seconds": 1.0,
        "cpu_limit_tolerance_seconds": 0.02,
        "cpu_normalized_within_limit": True,
        "cpu_utilization_percent_of_compute_limit": 75.0,
        "process_tree_cpu_seconds": 1.5,
        "process_tree_user_cpu_seconds": 1.2,
        "process_tree_system_cpu_seconds": 0.3,
        "process_tree_cpu_user_seconds": 1.2,
        "process_tree_cpu_system_seconds": 0.3,
        "process_tree_voluntary_context_switches": 5,
        "process_tree_involuntary_context_switches": 2,
        "process_tree_context_switches": 7,
        "process_tree_minor_faults": 4,
        "process_tree_major_faults": 0,
        "process_tree_read_bytes": 7,
        "process_tree_write_bytes": 11,
        "process_tree_cpu_migrations": 2,
        "process_tree_migration_count": 2,
        "process_tree_schedstat": {
            "runtime_ns": 100,
            "runqueue_delay_ns": 200,
            "timeslices": 3,
        },
        "actual_affinity_union": [0, 1],
        "actual_affinity_intersection": [0, 1],
        "cpu_affinity_union": [0, 1],
        "cpu_affinity_intersection": [0, 1],
        "actual_affinity_union_count": 2,
        "actual_affinity_intersection_count": 2,
        "affinity_status": "available",
        "process_metrics": [
            {
                "pid": 1,
                "create_time": 1.0,
                "sample_count": 2,
                "cpu_baseline_source": "monitor_start",
                "user_cpu_seconds": 1.2,
                "system_cpu_seconds": 0.3,
                "counters": {
                    "voluntary_context_switches": 5,
                    "involuntary_context_switches": 2,
                    "minor_faults": 4,
                    "major_faults": 0,
                    "read_bytes": 7,
                    "write_bytes": 11,
                    "schedstat_runtime_ns": 100,
                    "schedstat_runqueue_delay_ns": 200,
                    "schedstat_timeslices": 3,
                    "cpu_migrations": 2,
                },
                "cpu_migrations": 2,
                "schedstat": {
                    "runtime_ns": 100,
                    "runqueue_delay_ns": 200,
                    "timeslices": 3,
                },
                "last_affinity": [0, 1],
            }
        ],
        "resource_summary_accounting_source": (
            MODE_WAVE_RESOURCE_ACCOUNTING_SOURCE
        ),
        "thread_tree_status": "available",
        "thread_tree": {
            "status": "available",
            "observation_capacity": 4096,
            "observed_thread_identities": 1,
            "sample_missed_processes": 0,
            "unresolved_thread_observation_count": 0,
            "unresolved_thread_ids": [],
            "user_cpu_seconds": 0.25,
            "system_cpu_seconds": 0.05,
            "counters": {
                "voluntary_context_switches": 3,
                "involuntary_context_switches": 1,
                "minor_faults": 2,
                "major_faults": 0,
                "schedstat_runtime_ns": 1,
                "schedstat_runqueue_delay_ns": 2,
                "schedstat_timeslices": 3,
                "cpu_migrations": 1,
            },
            "affinity_union": [0, 1],
            "affinity_intersection": [0, 1],
        },
        "thread_metrics": [
            {
                "pid": 1,
                "process_create_time": 1.0,
                "tid": 1,
                "thread_start_time_ticks": 1,
                "sample_count": 2,
                "cpu_baseline_source": "monitor_start",
                "user_cpu_seconds": 0.25,
                "system_cpu_seconds": 0.05,
                "counters": {
                    "voluntary_context_switches": 3,
                    "involuntary_context_switches": 1,
                    "minor_faults": 2,
                    "major_faults": 0,
                    "schedstat_runtime_ns": 1,
                    "schedstat_runqueue_delay_ns": 2,
                    "schedstat_timeslices": 3,
                    "cpu_migrations": 1,
                },
                "last_affinity": [0, 1],
            }
        ],
        "monitor_start_boot_time_ticks": 1,
        "thread_tree_user_cpu_seconds": 0.25,
        "thread_tree_system_cpu_seconds": 0.05,
        "thread_affinity_union": [0, 1],
        "thread_affinity_intersection": [0, 1],
        "thread_tree_schedstat": {
            "runtime_ns": 1,
            "runqueue_delay_ns": 2,
            "timeslices": 3,
        },
        "thread_tree_context_switches": 4,
        "thread_tree_cpu_migrations": 1,
        "thread_tree_minor_faults": 2,
        "thread_tree_major_faults": 0,
    }
    _review_mode_wave_resources(
        wave,
        comparison_schema=TICK_PROFILE_COMPARISON_SCHEMA_VERSION,
    )
    current_metrics = _mode_wave_metrics(
        (
            ReviewRecord(
                Path("current-axis.json"),
                {"schema_version": TICK_PROFILE_COMPARISON_SCHEMA_VERSION},
                wave,
            ),
        )
    )
    assert cast(dict[str, object], current_metrics["context_switches"])["median"] == 7.0
    assert cast(dict[str, object], current_metrics["cpu_migrations"])["median"] == 2.0

    wave["process_tree_context_switches"] = 8
    with pytest.raises(RuntimeError, match="process-tree aggregates do not replay"):
        _review_mode_wave_resources(
            wave,
            comparison_schema=TICK_PROFILE_COMPARISON_SCHEMA_VERSION,
        )
    wave["process_tree_context_switches"] = 7

    partial_thread_wave = copy.deepcopy(wave)
    partial_thread_tree = cast(dict[str, object], partial_thread_wave["thread_tree"])
    partial_thread_tree["sample_missed_processes"] = 1
    partial_thread_tree["unresolved_thread_observation_count"] = 1
    partial_thread_tree["unresolved_thread_ids"] = [
        {"pid": 2, "process_create_time": 2.0, "tid": 3}
    ]
    _review_mode_wave_resources(
        partial_thread_wave,
        comparison_schema=TICK_PROFILE_COMPARISON_SCHEMA_VERSION,
    )

    prior_wave = copy.deepcopy(wave)
    prior_wave.pop("resource_summary_accounting_source")
    _review_mode_wave_resources(
        prior_wave,
        comparison_schema=PRIOR_PROFILE_COMPARISON_SCHEMA_VERSION,
    )
    prior_metrics = _mode_wave_metrics(
        (
            ReviewRecord(
                Path("prior-axis.json"),
                {"schema_version": PRIOR_PROFILE_COMPARISON_SCHEMA_VERSION},
                prior_wave,
            ),
        )
    )
    assert cast(dict[str, object], prior_metrics["io_read_bytes"])["median"] == 5.0
    assert prior_metrics["io_accounting_sources"] == ["cgroup_v2_io_stat"]

    legacy_wave = copy.deepcopy(prior_wave)
    legacy_io = cast(dict[str, object], legacy_wave.pop("io_accounting"))
    legacy_io.pop("source")
    legacy_wave["cgroup_io_deltas"] = legacy_io
    _review_mode_wave_resources(
        legacy_wave,
        comparison_schema=LEGACY_PROFILE_COMPARISON_SCHEMA_VERSION,
    )
    legacy_metrics = _mode_wave_metrics(
        (
            ReviewRecord(
                Path("legacy-axis.json"),
                {"schema_version": LEGACY_PROFILE_COMPARISON_SCHEMA_VERSION},
                legacy_wave,
            ),
        )
    )
    assert cast(dict[str, object], legacy_metrics["io_read_bytes"])["median"] == 5.0
    assert legacy_metrics["io_accounting_sources"] == ["cgroup_v2_io_stat"]

    thread_tree = cast(dict[str, object], wave["thread_tree"])
    counters = cast(dict[str, int], thread_tree["counters"])
    counters["cpu_migrations"] += 1
    with pytest.raises(RuntimeError, match="aggregates do not replay"):
        _review_mode_wave_resources(
            wave,
            comparison_schema=TICK_PROFILE_COMPARISON_SCHEMA_VERSION,
        )
    counters["cpu_migrations"] -= 1

    after = wave["cgroup_after"]
    assert isinstance(after, dict)
    after["memory_swap_peak_bytes"] = 1
    with pytest.raises(RuntimeError, match="swap gate"):
        _review_mode_wave_resources(
            wave,
            comparison_schema=TICK_PROFILE_COMPARISON_SCHEMA_VERSION,
        )
    after["memory_swap_peak_bytes"] = 0

    before = cast(dict[str, object], wave["cgroup_before"])
    before["io"] = "unavailable"
    after["io"] = "unavailable"
    wave["io_accounting"] = {
        "source": "process_tree_proc_io",
        "read_bytes": 7,
        "write_bytes": 11,
    }
    _review_mode_wave_resources(
        wave,
        comparison_schema=TICK_PROFILE_COMPARISON_SCHEMA_VERSION,
    )
    process_io = cast(dict[str, object], wave["io_accounting"])
    process_io["write_bytes"] = 12
    wave["process_tree_write_bytes"] = 12
    with pytest.raises(RuntimeError, match="aggregates do not replay"):
        _review_mode_wave_resources(
            wave,
            comparison_schema=TICK_PROFILE_COMPARISON_SCHEMA_VERSION,
        )
    process_io["write_bytes"] = 11
    wave["process_tree_write_bytes"] = 11
    before["io"] = io_before
    after["io"] = io_after
    with pytest.raises(RuntimeError, match="did not prefer cgroup"):
        _review_mode_wave_resources(
            wave,
            comparison_schema=TICK_PROFILE_COMPARISON_SCHEMA_VERSION,
        )


def test_current_mode_wave_replays_worker_terminal_io_receipts() -> None:
    core = {
        "schema_version": "stage05.2-terminal-process-io-v1",
        "pid": 41,
        "parent_pid": 40,
        "create_time": 11.0,
        "start_time_ticks": 101,
        "task_started_monotonic": 21.0,
        "captured_monotonic": 25.0,
        "read_bytes": 101,
        "write_bytes": 202,
    }
    axis = {"repeat": 1, "axis": "fixed_work", "instance": "c101C5", "seed": 2014}
    wave: dict[str, object] = {
        "monitor_start_wall_time": 10.0,
        "monitor_start_monotonic": 20.0,
        "monitor_start_boot_time_ticks": 100,
        "monitor_end_monotonic": 30.0,
        "sample_count": 2,
        "root_process_id": 40,
        "process_io_terminal_status": "available",
        "process_io_uncovered_identities": [],
        "identities": [axis],
        "scheduler_process_ids": [],
        "terminal_process_io_receipts": [core],
        "worker_terminal_io_receipts": [{**axis, **core}],
        "scheduler_terminal_io_receipts": [],
        "process_metrics": [
            {
                "pid": 41,
                "create_time": 11.0,
                "parent_pid": 40,
                "start_time_ticks": 101,
                "terminal_io_evidence": "cooperative_receipt",
                "last_sample_index": 0,
                "cpu_baseline_source": "process_create_time",
                "counters": {"read_bytes": 101, "write_bytes": 202},
            }
        ],
    }

    _review_worker_terminal_io(wave)

    cast(list[dict[str, object]], wave["worker_terminal_io_receipts"])[0][
        "write_bytes"
    ] = 203
    with pytest.raises(RuntimeError, match="does not replay"):
        _review_worker_terminal_io(wave)


def test_current_host_scheduler_wave_partitions_terminal_io_receipts() -> None:
    worker_core = {
        "schema_version": "stage05.2-terminal-process-io-v1",
        "pid": 41,
        "parent_pid": 40,
        "create_time": 11.0,
        "start_time_ticks": 101,
        "task_started_monotonic": 21.0,
        "captured_monotonic": 25.0,
        "read_bytes": 101,
        "write_bytes": 202,
    }
    scheduler_core = {
        "schema_version": "stage05.2-terminal-process-io-v1",
        "pid": 42,
        "parent_pid": 40,
        "create_time": 12.0,
        "start_time_ticks": 102,
        "task_started_monotonic": 20.5,
        "captured_monotonic": 26.0,
        "read_bytes": 303,
        "write_bytes": 404,
    }
    axis = {"repeat": 1, "axis": "fixed_work", "instance": "c101C5", "seed": 2014}
    wave: dict[str, object] = {
        "monitor_start_wall_time": 10.0,
        "monitor_start_monotonic": 20.0,
        "monitor_start_boot_time_ticks": 100,
        "monitor_end_monotonic": 30.0,
        "sample_count": 2,
        "root_process_id": 40,
        "process_io_terminal_status": "available",
        "process_io_uncovered_identities": [],
        "identities": [axis],
        "scheduler_process_ids": [42],
        "terminal_process_io_receipts": [dict(worker_core), dict(scheduler_core)],
        "worker_terminal_io_receipts": [{**axis, **worker_core}],
        "scheduler_terminal_io_receipts": [dict(scheduler_core)],
        "process_metrics": [
            {
                "pid": receipt["pid"],
                "create_time": receipt["create_time"],
                "parent_pid": 40,
                "start_time_ticks": receipt["start_time_ticks"],
                "terminal_io_evidence": "cooperative_receipt",
                "last_sample_index": 0,
                "cpu_baseline_source": "process_create_time",
                "counters": {
                    "read_bytes": receipt["read_bytes"],
                    "write_bytes": receipt["write_bytes"],
                },
            }
            for receipt in (worker_core, scheduler_core)
        ],
    }

    _review_worker_terminal_io(wave)

    cast(list[dict[str, object]], wave["scheduler_terminal_io_receipts"])[0][
        "pid"
    ] = 41
    with pytest.raises(RuntimeError, match="scheduler terminal process I/O"):
        _review_worker_terminal_io(wave)


def _streaming_evidence_digest(rows: Iterable[object]) -> dict[str, object]:
    digest = hashlib.sha256(b"stage05.2-test-streaming-evidence-v1")
    count = 0
    for row in rows:
        encoded = json.dumps(
            evidence_json_value(row),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        digest.update(struct.pack("<Q", len(encoded)))
        digest.update(encoded)
        count += 1
    return {"count": count, "sha256": digest.hexdigest()}


def _paired_semantic_snapshot(
    result: ALNSResult,
    axis_path: Path,
    *,
    compare_raw_neighborhood: bool = True,
    journal_round_trip: bool = False,
) -> dict[str, object]:
    selected_names = (
        "candidate_state",
        "stage04",
        "exact_result",
        "deadline",
        "termination",
        "native_failure",
    )
    selected: dict[str, list[dict[str, object]]] = {name: [] for name in selected_names}
    if journal_round_trip:
        journal = write_semantic_journal(axis_path, result)
        canonical_events = iter_verified_semantic_journal(axis_path, journal)
    else:
        canonical_events = iter_canonical_semantic_events(result)
    for event in canonical_events:
        stream_name = event.get("semantic_stream")
        if isinstance(stream_name, str) and stream_name in selected:
            selected[stream_name].append(event)
    semantic_trajectory = list(_iter_semantic_candidate_trajectory(result))
    comparison_projection = _comparison_projection_from_selected(
        {"semantic_trajectory": semantic_trajectory},
        selected,
    )
    snapshot: dict[str, object] = {
        "feasible": result.feasible,
        "customer_sequences": result.customer_sequences,
        "objective": None if result.objective is None else result.objective.key,
        "iterations": result.iterations,
        "effective_iterations": result.effective_iterations,
        "termination_reason": result.termination_reason,
        "exact_started_calls": result.exact_started_calls,
        "exact_completed_calls": result.exact_completed_calls,
        "candidate_work_hash": result.candidate_work_hash,
        "route_result_hash": result.route_result_hash,
        "stage04_statistics": result.stage04_statistics,
        "stage04_weight_history": _streaming_evidence_digest(result.stage04_weight_history),
        "stage04_event_log": _streaming_evidence_digest(result.stage04_event_log),
        "semantic_trajectory": _streaming_evidence_digest(semantic_trajectory),
        "canonical_semantic_projection": comparison_projection,
    }
    if compare_raw_neighborhood:
        snapshot["neighborhood_statistics"] = result.neighborhood_statistics
        snapshot["neighborhood_events"] = _streaming_evidence_digest(result.neighborhood_events)
    return snapshot


def _release_result_runtime_semantic_storage(result: ALNSResult) -> None:
    trace = result.measurement_trace
    if trace is not None:
        trace.release_runtime_semantic_storage()


def _assert_paired_semantic_snapshots(
    python_snapshot: dict[str, object],
    native_snapshot: dict[str, object],
) -> None:
    if native_snapshot == python_snapshot:
        return
    changed_fields: dict[str, object] = {}
    for key in python_snapshot.keys() | native_snapshot.keys():
        python_value = python_snapshot.get(key)
        native_value = native_snapshot.get(key)
        if python_value == native_value:
            continue
        if key == "canonical_semantic_projection":
            assert isinstance(python_value, list)
            assert isinstance(native_value, list)
            first_difference = next(
                (
                    (index, python_row, native_row)
                    for index, (python_row, native_row) in enumerate(
                        zip(python_value, native_value, strict=False)
                    )
                    if python_row != native_row
                ),
                None,
            )
            changed_fields[key] = {
                "python_count": len(python_value),
                "native_count": len(native_value),
                "first_difference": first_difference,
            }
        else:
            changed_fields[key] = {
                "python": python_value,
                "native": native_value,
            }
    pytest.fail(
        json.dumps(
            {"changed_fields": changed_fields},
            sort_keys=True,
            default=str,
        )
    )


@pytest.mark.external_data
def test_full_native_first_twelve_rounds_match_python_candidate_control(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from evrptw.experiments import stage052_native_architectures as architectures

    root = Path.cwd()
    bundle = root / "results/stage05.2_native_architecture_warm_start_paired_attempt04.json"
    if not bundle.is_file():
        pytest.skip("paired attempt04 warm-start bundle is not linked")
    warm_starts = load_warm_start_bundle(
        bundle,
        benchmark_dir=root / "data/schneider",
    )
    sequences, provenance = warm_starts[("c101_21", 2014)]
    task = ArchitectureAxisTask(
        scope="paired",
        repeat=0,
        axis="fixed_work",
        instance_name="c101_21",
        seed=2014,
        benchmark_dir=root / "data/schneider",
        output_root=tmp_path,
        run_labels=run_labels_for_scope("paired", 99),
        scheduler_socket_path=str(tmp_path / "scheduler.sock"),
        wheel_sha256="a" * 64,
        native_sha256="b" * 64,
        scheduler_sha256="c" * 64,
        revision="d" * 40,
        initial_customer_sequences=sequences,
        initial_solution_provenance=provenance,
    )
    real_solve_alns = architectures.solve_alns

    def capped_solve_alns(*args: object, **kwargs: object) -> ALNSResult:
        kwargs["max_iterations"] = 12
        return real_solve_alns(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(architectures, "solve_alns", capped_solve_alns)
    python_result, *_ = architectures._solve_mode(
        ArchitectureMode.PYTHON_CANDIDATE_CONTROL,
        task,
    )
    try:
        native_result, *_ = architectures._solve_mode(
            ArchitectureMode.FULL_NATIVE_ALNS,
            task,
        )
        try:
            assert native_result.exact_started_calls == python_result.exact_started_calls, (
                python_result.exact_started_calls,
                native_result.exact_started_calls,
                tuple(
                    (
                        event.get("iteration"),
                        event.get("operator"),
                        len(event.get("sequences", [])),
                        event.get("sequences", [])[:1],
                    )
                    for event in python_result.candidate_work_events
                ),
                tuple(
                    (
                        event.get("iteration"),
                        event.get("operator"),
                        len(event.get("sequences", [])),
                        event.get("sequences", [])[:1],
                    )
                    for event in native_result.candidate_work_events
                ),
            )
            assert _semantic_candidate_trajectory(native_result) == (
                _semantic_candidate_trajectory(python_result)
            )
            assert native_result.candidate_work_events == (python_result.candidate_work_events), (
                python_result.candidate_work_events,
                native_result.candidate_work_events,
            )
            assert native_result.route_result_events == python_result.route_result_events
            assert native_result.neighborhood_events == python_result.neighborhood_events
            assert _paired_semantic_snapshot(
                native_result,
                tmp_path / "twelve-round-native.json",
                journal_round_trip=True,
            ) == _paired_semantic_snapshot(
                python_result,
                tmp_path / "twelve-round-python.json",
                journal_round_trip=True,
            )
        finally:
            _release_result_runtime_semantic_storage(native_result)
    finally:
        _release_result_runtime_semantic_storage(python_result)


@pytest.mark.external_data
@pytest.mark.parametrize("instance_name", PAIRED_INSTANCES)
@pytest.mark.parametrize("seed", SEEDS)
def test_paired_runner_full_native_replays_real_worker_count_four_axis(
    tmp_path: Path,
    instance_name: str,
    seed: int,
) -> None:
    root = Path.cwd()
    bundle = root / "results/stage05.2_native_architecture_warm_start_paired_attempt04.json"
    if not bundle.is_file():
        pytest.skip("paired attempt04 warm-start bundle is not linked")
    warm_starts = load_warm_start_bundle(
        bundle,
        benchmark_dir=root / "data/schneider",
    )
    sequences, provenance = warm_starts[(instance_name, seed)]
    task = ArchitectureAxisTask(
        scope="paired",
        repeat=0,
        axis="fixed_work",
        instance_name=instance_name,
        seed=seed,
        benchmark_dir=root / "data/schneider",
        output_root=tmp_path,
        run_labels=run_labels_for_scope("paired", 99),
        scheduler_socket_path=str(tmp_path / "scheduler.sock"),
        wheel_sha256="a" * 64,
        native_sha256="b" * 64,
        scheduler_sha256="c" * 64,
        revision="d" * 40,
        initial_customer_sequences=sequences,
        initial_solution_provenance=provenance,
    )

    python_result, _python_seconds, _python_topology = _solve_mode(
        ArchitectureMode.PYTHON_CANDIDATE_CONTROL,
        task,
    )
    try:
        python_snapshot = _paired_semantic_snapshot(
            python_result,
            tmp_path / "python-candidate-control.json",
            compare_raw_neighborhood=False,
        )
    finally:
        _release_result_runtime_semantic_storage(python_result)
    python_result = None
    gc.collect()
    native_result, _native_seconds, _native_topology = _solve_mode(
        ArchitectureMode.FULL_NATIVE_ALNS,
        task,
    )
    try:
        native_snapshot = _paired_semantic_snapshot(
            native_result,
            tmp_path / "full-native-alns.json",
            compare_raw_neighborhood=False,
        )
    finally:
        _release_result_runtime_semantic_storage(native_result)
    native_result = None
    gc.collect()

    _assert_paired_semantic_snapshots(python_snapshot, native_snapshot)


@pytest.mark.external_data
@pytest.mark.parametrize("instance_name", PAIRED_INSTANCES)
@pytest.mark.parametrize("seed", SEEDS)
def test_paired_runner_per_solve_replays_worker_count_four_axis(
    tmp_path: Path,
    instance_name: str,
    seed: int,
) -> None:
    """Production per-solve mode must match the same attempt04 semantic baseline."""

    root = Path.cwd()
    bundle = root / "results/stage05.2_native_architecture_warm_start_paired_attempt04.json"
    if not bundle.is_file():
        pytest.skip("paired attempt04 warm-start bundle is not linked")
    warm_starts = load_warm_start_bundle(
        bundle,
        benchmark_dir=root / "data/schneider",
    )
    sequences, provenance = warm_starts[(instance_name, seed)]
    task = ArchitectureAxisTask(
        scope="paired",
        repeat=0,
        axis="fixed_work",
        instance_name=instance_name,
        seed=seed,
        benchmark_dir=root / "data/schneider",
        output_root=tmp_path,
        run_labels=run_labels_for_scope("paired", 99),
        scheduler_socket_path=str(tmp_path / "scheduler.sock"),
        wheel_sha256="a" * 64,
        native_sha256="b" * 64,
        scheduler_sha256="c" * 64,
        revision="d" * 40,
        initial_customer_sequences=sequences,
        initial_solution_provenance=provenance,
    )

    python_result, _python_seconds, _python_topology = _solve_mode(
        ArchitectureMode.PYTHON_CANDIDATE_CONTROL,
        task,
    )
    try:
        python_snapshot = _paired_semantic_snapshot(
            python_result,
            tmp_path / "python-candidate-control.json",
            compare_raw_neighborhood=False,
        )
    finally:
        _release_result_runtime_semantic_storage(python_result)
    python_result = None
    gc.collect()
    native_result, _native_seconds, _native_topology = _solve_mode(
        ArchitectureMode.PER_SOLVE_RUNTIME,
        task,
    )
    try:
        pair_pruning_events = [
            event
            for event in native_result.neighborhood_events
            if event["status"] == "pair_prefilter_rejected_aggregate"
        ]
        assert all(int(event["aggregate_count"]) > 0 for event in pair_pruning_events)
        assert all(len(str(event["candidate_pool_hash"])) == 64 for event in pair_pruning_events)
        assert native_result.native_execution_statistics["fallback_count"] == 0
        native_snapshot = _paired_semantic_snapshot(
            native_result,
            tmp_path / "per-solve-runtime.json",
            compare_raw_neighborhood=False,
        )
    finally:
        _release_result_runtime_semantic_storage(native_result)
    native_result = None
    gc.collect()

    _assert_paired_semantic_snapshots(python_snapshot, native_snapshot)


@pytest.fixture(scope="module")
def native_architecture_host_scheduler_socket(
    tmp_path_factory: pytest.TempPathFactory,
) -> str:
    endpoint = tmp_path_factory.mktemp("native-architecture-host") / "scheduler.sock"
    with NativeHostScheduler(endpoint):
        yield str(endpoint)


@pytest.mark.external_data
@pytest.mark.parametrize("instance_name", PAIRED_INSTANCES)
@pytest.mark.parametrize("seed", SEEDS)
def test_paired_runner_host_scheduler_replays_shared_pool_axis(
    tmp_path: Path,
    instance_name: str,
    seed: int,
    native_architecture_host_scheduler_socket: str,
) -> None:
    """The shared host pool must preserve the attempt04 canonical semantics."""

    root = Path.cwd()
    bundle = root / "results/stage05.2_native_architecture_warm_start_paired_attempt04.json"
    if not bundle.is_file():
        pytest.skip("paired attempt04 warm-start bundle is not linked")
    warm_starts = load_warm_start_bundle(
        bundle,
        benchmark_dir=root / "data/schneider",
    )
    sequences, provenance = warm_starts[(instance_name, seed)]
    task = ArchitectureAxisTask(
        scope="paired",
        repeat=0,
        axis="fixed_work",
        instance_name=instance_name,
        seed=seed,
        benchmark_dir=root / "data/schneider",
        output_root=tmp_path,
        run_labels=run_labels_for_scope("paired", 99),
        scheduler_socket_path=native_architecture_host_scheduler_socket,
        wheel_sha256="a" * 64,
        native_sha256="b" * 64,
        scheduler_sha256="c" * 64,
        revision="d" * 40,
        initial_customer_sequences=sequences,
        initial_solution_provenance=provenance,
    )

    python_result, _python_seconds, _python_topology = _solve_mode(
        ArchitectureMode.PYTHON_CANDIDATE_CONTROL,
        task,
    )
    try:
        python_snapshot = _paired_semantic_snapshot(
            python_result,
            tmp_path / "python-candidate-control.json",
            compare_raw_neighborhood=False,
        )
    finally:
        _release_result_runtime_semantic_storage(python_result)
    python_result = None
    gc.collect()
    native_result, _native_seconds, _native_topology = _solve_mode(
        ArchitectureMode.HOST_SCHEDULER,
        task,
    )
    try:
        assert native_result.native_execution_statistics["fallback_count"] == 0
        assert native_result.native_execution_statistics["shared_native_work_pool"] is True
        native_snapshot = _paired_semantic_snapshot(
            native_result,
            tmp_path / "host-scheduler.json",
            compare_raw_neighborhood=False,
        )
    finally:
        _release_result_runtime_semantic_storage(native_result)
    native_result = None
    gc.collect()

    _assert_paired_semantic_snapshots(python_snapshot, native_snapshot)


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
        __build_performance_profile__="portable-o3",
        __build_compiler_id__="GNU",
        __build_compiler_version__="13.3.0",
        __build_interprocedural_optimization__=False,
        __build_host_native__=False,
    )
    assert (
        _validate_native_build_attestation(
            clean,
            expected_revision="a" * 40,
            expected_tree="b" * 40,
            expected_source_manifest_sha256="c" * 64,
            expected_tracked_file_count=878,
        )
        is None
    )

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
    assert receipt == {"evrptw/runtime.py": hashlib.sha256(b"reviewed-runtime").hexdigest()}

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


def _initial_four_lane_projection(*, request_sha256: str) -> tuple[dict[str, object], str, str]:
    route_offsets = [0, 1]
    route_indices = [1]
    path_offsets = [0, 3]
    path_indices = [0, 1, 0]
    statuses = [0]
    reasons = [0]
    metrics = [[2.0, 2.0, 0.0, 0.0]]
    labels = [[1, 1, 0]]
    batch_counters = [1, 1, 1, 0, 1, 0, 0, 0, 1, 128]
    completion_order = [0]
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
        completion_order,
        objective_integer,
    ):
        lane_evidence.extend(struct.pack("<Q", len(values)))
        lane_evidence.extend(struct.pack(f"<{len(values)}q", *values))
    lane_evidence.extend(struct.pack("<Q", len(objective_float)))
    lane_evidence.extend(struct.pack("<2d", *objective_float))
    lane_sha256 = hashlib.sha256(lane_evidence).hexdigest()
    initial_state_evidence = bytearray(b"stage05.2-native-initial-search-state-v2")
    initial_state_evidence.extend(request_sha256.encode("ascii"))
    for values in (path_offsets, path_indices, statuses, reasons):
        initial_state_evidence.extend(struct.pack("<Q", len(values)))
        initial_state_evidence.extend(struct.pack(f"<{len(values)}q", *values))
    initial_state_evidence.extend(struct.pack("<Q", len(flattened_metrics)))
    initial_state_evidence.extend(struct.pack(f"<{len(flattened_metrics)}d", *flattened_metrics))
    for values in (
        [value for row in labels for value in row],
        batch_counters,
        completion_order,
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
            "completion_order": completion_order,
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
        },
    }

    assert (
        _replay_initial_state_receipt(
            payload,
            ArchitectureMode.HOST_SCHEDULER,
            expected_node_kind=(0, 1),
            expected_exact_batch_size=128,
        )
        is None
    )
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
    assert (
        _replay_initial_state_receipt(
            payload,
            ArchitectureMode.HOST_SCHEDULER,
            expected_node_kind=(0, 1),
            expected_exact_batch_size=128,
        )
        == "initial-state ownership receipt does not reconcile"
    )


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
                "transaction_sha256": hashlib.sha256(receipt_evidence).hexdigest(),
            },
        },
    }

    assert (
        _replay_initial_state_receipt(
            payload,
            ArchitectureMode.HOST_SCHEDULER,
            expected_node_kind=(0, 1),
            expected_exact_batch_size=128,
        )
        == "initial-state projection hash mismatch"
    )


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
        "completion_order",
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
        },
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
                "initial_four_lane_state_sha256": (initial_four_lane_state_sha256),
                "initial_four_lane_projection": projection,
                "transaction_sha256": hashlib.sha256(evidence).hexdigest(),
            },
        },
    }

    assert (
        _replay_initial_state_receipt(
            payload,
            ArchitectureMode.HOST_SCHEDULER,
            expected_node_kind=(0, 1, 1),
            expected_exact_batch_size=128,
        )
        == "initial four-lane projection values do not reconcile"
    )


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
        lambda command, **_kwargs: Receipt("c" * 40 + "\n" if command[1] == "rev-parse" else ""),
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
                "native_telemetry": {
                    "runtime_native_event_id": event_id,
                    "runtime_native_stream_code": event_id - 1,
                    "runtime_native_event_code": event_id,
                    "runtime_native_lane_id": 0,
                    "runtime_native_operator_id": 0,
                    "runtime_native_iteration": event_id - 1,
                    "runtime_native_transaction_id": event_id,
                    "runtime_native_subject_id": 0,
                    "runtime_native_status_code": 0,
                    "runtime_native_flags": 0,
                },
                **({"status": "iteration_limit"} if stream_name == "termination" else {}),
            }
        )
    return streams


def _full_native_inline_payload(
    streams: dict[str, list[dict[str, object]]],
    events: list[dict[str, object]],
) -> dict[str, object]:
    projection = _NativeCanonicalProjectionHasher()
    for event in events:
        raw_native = event.get("native_telemetry")
        if isinstance(raw_native, dict):
            projection.append(raw_native)
    return {
        "mode": "full_native_alns",
        "canonical_semantic_streams": streams,
        "canonical_semantic_events": events,
        "native_execution_statistics": {
            "canonical_event_count": projection.count,
            "canonical_event_journal_sha256": "0" * 64,
            "canonical_event_projection_sha256": projection.hexdigest(),
        },
    }


def test_unified_causal_ids_are_monotonic_and_cover_all_required_domains() -> None:
    streams = _complete_causal_streams()
    events = _canonical_semantic_event_sequence(streams)

    assert [event["semantic_event_id"] for event in events] == list(range(1, len(events) + 1))
    assert [event["runtime_event_id"] for event in events] == list(range(1, len(events) + 1))
    assert {str(event["semantic_stream"]) for event in events} >= (_BOUNDARY_CAUSAL_STREAMS)

    reviewed = _canonical_semantic_events(_full_native_inline_payload(streams, events))
    assert {str(event["semantic_stream"]) for event in reviewed} >= (_BOUNDARY_CAUSAL_STREAMS)


def test_unified_causal_ids_reject_duplicate_native_source_identity() -> None:
    streams = _complete_causal_streams()
    events = _canonical_semantic_event_sequence(streams)
    tampered_streams = {name: [dict(row) for row in rows] for name, rows in streams.items()}
    tampered_events = [dict(event) for event in events]
    first_native = tampered_events[0]["native_telemetry"]
    second_native = tampered_events[1]["native_telemetry"]
    assert isinstance(first_native, dict)
    assert isinstance(second_native, dict)
    second_native = dict(second_native)
    second_native["runtime_native_event_id"] = first_native["runtime_native_event_id"]
    tampered_events[1]["native_telemetry"] = second_native
    second_stream = str(tampered_events[1]["semantic_stream"])
    tampered_streams[second_stream][0] = {
        key: value
        for key, value in tampered_events[1].items()
        if key not in {"semantic_stream", "semantic_sequence"}
    }

    with pytest.raises(ValueError, match="native semantic event IDs are not contiguous"):
        _canonical_semantic_events(_full_native_inline_payload(tampered_streams, tampered_events))


@pytest.mark.parametrize("corruption", ("tail", "digest"))
def test_full_native_canonical_receipt_rejects_incomplete_or_tampered_projection(
    corruption: str,
) -> None:
    streams = _complete_causal_streams()
    events = _canonical_semantic_event_sequence(streams)
    payload = copy.deepcopy(_full_native_inline_payload(streams, events))
    raw_events = payload["canonical_semantic_events"]
    raw_streams = payload["canonical_semantic_streams"]
    assert isinstance(raw_events, list)
    assert isinstance(raw_streams, dict)
    if corruption == "tail":
        terminal = raw_events.pop()
        assert isinstance(terminal, dict)
        terminal_stream = terminal["semantic_stream"]
        stream_rows = raw_streams[terminal_stream]
        assert isinstance(stream_rows, list)
        stream_rows.pop()
        message = "coverage is incomplete"
    else:
        first = raw_events[0]
        assert isinstance(first, dict)
        native_telemetry = first["native_telemetry"]
        assert isinstance(native_telemetry, dict)
        native_telemetry["runtime_native_flags"] = 1
        first_stream = raw_streams[first["semantic_stream"]]
        assert isinstance(first_stream, list)
        first_stream[0]["native_telemetry"]["runtime_native_flags"] = 1
        message = "digest does not replay"
    with pytest.raises(ValueError, match=message):
        _canonical_semantic_events(payload)


@pytest.mark.parametrize(
    "tamper",
    ("context", "native_transaction", "source_native_transaction"),
)
def test_cache_commit_requires_its_exact_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    instance = parse_schneider(Path(__file__).resolve().parents[1] / "data/schneider/c101C5.txt")
    sequence = ("C30",)
    exact = solve_exact_charging(instance, sequence)
    assert exact.feasible
    route_key = canonical_route_key(sequence)
    charging_version = "test-charging-v1"
    objective_version = "test-objective-v1"
    cache_digest = RouteCacheKey(
        instance_hash=canonical_instance_hash(instance),
        customer_sequence=sequence,
        charging_configuration_version=charging_version,
        objective_schema_version=objective_version,
    ).digest

    def native(event_id: int, stream_code: int, transaction_id: int) -> dict[str, int]:
        return {
            "runtime_native_event_id": event_id,
            "runtime_native_stream_code": stream_code,
            "runtime_native_event_code": event_id,
            "runtime_native_lane_id": 0,
            "runtime_native_operator_id": 0,
            "runtime_native_iteration": 1,
            "runtime_native_transaction_id": transaction_id,
            "runtime_native_subject_id": 0,
            "runtime_native_status_code": 0,
            "runtime_native_flags": 0,
        }

    context = {"lane": "legacy", "iteration": 1, "operator": "route_merge"}
    events: list[dict[str, object]] = [
        {
            "semantic_stream": "exact_work",
            "event_type": "exact_batch_started",
            **context,
            "customer_sequences": [list(sequence)],
            "requested_calls": 1,
            "started_calls": 1,
            "native_telemetry": native(1, 4, 42),
        },
        {
            "semantic_stream": "exact_result",
            "event_type": "exact_route_result",
            **context,
            "route_key": route_key,
            "exact_started": True,
            "exact_completed": True,
            "status": "completed",
            "feasible": True,
            "failure_reason": exact.failure_reason,
            "native_telemetry": native(2, 5, 42),
        },
        {
            "semantic_stream": "cache",
            "event_type": "cache_lifecycle",
            **context,
            "route_key": route_key,
            "status": "store",
            "cache_key_digest": cache_digest,
            "entry_bytes": 1,
            "current_entries": 1,
            "current_bytes": 1,
            "native_telemetry": native(3, 6, 42),
        },
    ]
    if tamper == "context":
        events[-1]["iteration"] = 2
    elif tamper == "native_transaction":
        raw_native = events[-1]["native_telemetry"]
        assert isinstance(raw_native, dict)
        raw_native["runtime_native_transaction_id"] = 43
    else:
        events[-1].pop("native_telemetry")
        events[-1]["native_transaction_id"] = 43
    monkeypatch.setattr(
        "evrptw.experiments.stage052_native_architecture_review.iter_verified_semantic_journal",
        lambda *_args, **_kwargs: iter(events),
    )
    payload = {
        "canonical_semantic_journal": {"schema_version": "test"},
        "mode": "full_native_alns",
        "route_result_hash": "",
        "cache_incremental_statistics": {
            "config": {
                "enabled": True,
                "eviction_policy": "lru",
                "max_entries": 8,
                "max_memory_bytes": 1_000_000,
                "charging_configuration_version": charging_version,
                "objective_schema_version": objective_version,
                "instance_hash": canonical_instance_hash(instance),
            }
        },
    }

    assert (
        _replay_canonical_journal(
            payload,
            axis_path=tmp_path / "axis.json",
            instance=instance,
        )
        == "cache store transaction does not match exact result"
    )


def test_reference_distance_replay_caches_each_unique_route(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from evrptw.experiments import stage052_native_architecture_review as review

    instance = parse_schneider(Path(__file__).resolve().parents[1] / "data/schneider/c101C5.txt")
    sequence = ("C30",)
    expected = solve_exact_charging(instance, sequence)
    assert expected.feasible
    charging_count = sum(instance.by_name[node].kind.value == "f" for node in expected.route)
    statistics = [0] * 11
    events: list[dict[str, object]] = []
    for batch_ordinal, transaction_id in enumerate((41, 42)):
        events.append(
            {
                "event_type": "reference_distance_resolution",
                "native_transaction_id": transaction_id,
                "batch_ordinal": batch_ordinal,
                "customer_sequence": list(sequence),
                "status": "feasible",
                "vehicle_count": 1,
                "charging_count": charging_count,
                "reference_distance": expected.distance,
                "reference_charging_time": expected.charging_time,
                "transaction_status_code": 5,
            }
        )
        if batch_ordinal == 0:
            events.append(
                {
                    "event_type": "candidate_control_budget",
                    "native_transaction_id": transaction_id,
                    "batch_ordinal": batch_ordinal,
                    "status": "budget_skipped",
                    "context": "quality_shadow:relocate:candidate_pool",
                    "requested": 2,
                    "granted": 0,
                    "remaining": 1,
                    "iteration": 0,
                }
            )
        events.append(
            {
                "event_type": "candidate_cache_transaction",
                "native_transaction_id": transaction_id,
                "batch_ordinal": batch_ordinal,
                "status": "committed",
                "lookups": 0,
                "hits": 0,
                "exact_stores": 0,
                "cache_statistics": statistics,
                "implementation_internal": True,
                "reference_distance_resolution": True,
            }
        )
    calls = 0
    real_solve = review.solve_exact_charging

    def counted_solve(
        observed_instance: object,
        observed_sequence: tuple[str, ...],
    ) -> object:
        nonlocal calls
        calls += 1
        return real_solve(observed_instance, observed_sequence)  # type: ignore[arg-type]

    monkeypatch.setattr(review, "solve_exact_charging", counted_solve)
    monkeypatch.setattr(
        review,
        "iter_verified_native_control_events",
        lambda *_args, **_kwargs: iter(events),
    )
    payload = {
        "canonical_semantic_journal": {"native_control_events": {"native_source_sha256": "a" * 64}},
        "native_execution_statistics": {"control_journal_sha256": "a" * 64},
        "candidate_control_statistics": {"budget_skips": 1},
        "cache_incremental_statistics": {
            name: 0
            for name in (
                "cache_lookups",
                "cache_hits",
                "cache_misses",
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

    assert _replay_raw_native_control_journal(
        payload,
        axis_path=tmp_path / "axis.json",
        instance=instance,
    ) == "reference-distance transaction contains a budget skip"
    assert calls == 1


def test_reference_distance_internal_prefix_skip_has_no_public_budget_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from evrptw.experiments import stage052_native_architecture_review as review

    instance = parse_schneider(Path(__file__).resolve().parents[1] / "data/schneider/c101C5.txt")
    reference = solve_exact_charging(instance, ("C30",))
    assert reference.feasible
    reference_charging_count = sum(
        instance.by_name[node].kind.value == "f" for node in reference.route
    )
    events: list[dict[str, object]] = [
        {
            "event_type": "reference_distance_resolution",
            "native_transaction_id": 40,
            "batch_ordinal": 0,
            "customer_sequence": ["C30"],
            "status": "feasible",
            "vehicle_count": 1,
            "charging_count": reference_charging_count,
            "reference_distance": reference.distance,
            "reference_charging_time": reference.charging_time,
            "transaction_status_code": 5,
        },
        {
            "event_type": "cache_store_receipt",
            "native_transaction_id": 40,
            "batch_ordinal": 0,
            "candidate_id": 0,
            "status": "oversize_not_cached",
            "eviction_count": 0,
            "entry_bytes": 1,
        },
        {
            "event_type": "candidate_cache_transaction",
            "native_transaction_id": 40,
            "batch_ordinal": 0,
            "status": "committed",
            "lookups": 0,
            "hits": 0,
            "exact_stores": 0,
            "cache_statistics": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            "implementation_internal": True,
            "round_budget_suppressed": False,
            "reference_distance_resolution": True,
            "iteration": 0,
            "budget_state": [1, 0, 0, 1, 0, 1, 1, 0, 0],
        },
        {
            "event_type": "reference_distance_resolution",
            "native_transaction_id": 41,
            "batch_ordinal": 1,
            "customer_sequence": ["C30"],
            "status": "budget_skipped",
            "vehicle_count": -1,
            "charging_count": -1,
            "reference_distance": None,
            "reference_charging_time": None,
            "transaction_status_code": 3,
        },
        {
            "event_type": "candidate_cache_transaction",
            "native_transaction_id": 41,
            "batch_ordinal": 1,
            "status": "committed",
            "lookups": 0,
            "hits": 0,
            "exact_stores": 0,
            "cache_statistics": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            "implementation_internal": True,
            "round_budget_suppressed": False,
            "reference_distance_resolution": True,
            "iteration": 0,
            "budget_state": [1, 0, 0, 1, 0, 1, 1, 0, 0],
        },
    ]
    monkeypatch.setattr(
        review,
        "iter_verified_native_control_events",
        lambda *_args, **_kwargs: iter(events),
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "axis": "fixed_work",
        "fixed_work_budget": {"exact_calls": 10},
        "canonical_semantic_journal": {
            "native_control_events": {"native_source_sha256": "a" * 64}
        },
        "native_execution_statistics": {"control_journal_sha256": "a" * 64},
        "candidate_control_statistics": {
            "budget_skips": 0,
            "max_exact_calls_per_round": 1,
        },
        "cache_incremental_statistics": {
            "cache_lookups": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "cache_stores": 0,
            "cache_evictions": 0,
            "cache_oversize_not_cached": 1,
            "entries_current": 0,
            "entries_peak": 0,
            "bytes_current": 0,
            "bytes_peak": 0,
            "unique_route_evaluations": 1,
        },
    }

    assert (
        _replay_raw_native_control_journal(
            payload,
            axis_path=tmp_path / "axis.json",
            instance=instance,
        )
        is None
    )

    transaction = events[4]
    transaction["budget_state"] = [1, 0, 0, 0, 1, 1, 1, 0, 0]
    assert _replay_raw_native_control_journal(
        payload,
        axis_path=tmp_path / "axis.json",
        instance=instance,
    ) == "native control round budget state does not replay"

    transaction["budget_state"] = [1, 0, 0, 0, 0, 1, 1, 0, 0]
    assert _replay_raw_native_control_journal(
        payload,
        axis_path=tmp_path / "axis.json",
        instance=instance,
    ) == "native control budget state does not match configured limits"

    first_transaction = events[2]
    events.insert(
        0,
        {
            "event_type": "candidate_plan_decision",
            "native_transaction_id": 40,
            "batch_ordinal": 0,
            "candidate_id": 0,
            "rank": 0,
            "status": "selected",
            "transaction_status_code": 5,
            "implementation_internal": True,
        },
    )
    assert _replay_raw_native_control_journal(
        payload,
        axis_path=tmp_path / "axis.json",
        instance=instance,
    ) == "native reference-distance transaction shape is invalid"
    events.pop(0)

    fixed_budget = payload["fixed_work_budget"]
    assert isinstance(fixed_budget, dict)
    fixed_budget["exact_calls"] = 1
    first_transaction["round_budget_suppressed"] = True
    first_transaction["iteration"] = None
    first_transaction["budget_state"] = [0, -1, -1, 0, 1, 1, 1, 0, 1]
    transaction["budget_state"] = [1, 0, 0, 0, 1, 1, 1, 0, 1]
    assert (
        _replay_raw_native_control_journal(
            payload,
            axis_path=tmp_path / "axis.json",
            instance=instance,
        )
        is None
    )

    first_transaction["budget_state"] = [1, 0, 0, 1, 0, 1, 1, 0, 1]
    assert _replay_raw_native_control_journal(
        payload,
        axis_path=tmp_path / "axis.json",
        instance=instance,
    ) == "suppressed native control round state changed"

    first_transaction["round_budget_suppressed"] = False
    assert _replay_raw_native_control_journal(
        payload,
        axis_path=tmp_path / "axis.json",
        instance=instance,
    ) == "native control accounted round identity is invalid"

    fixed_budget["exact_calls"] = 10
    first_transaction["iteration"] = 0
    payload["candidate_control_statistics"]["max_exact_calls_per_round"] = 2  # type: ignore[index]
    first_transaction["budget_state"] = [1, 0, 0, 2, 0, 1, 1, 0, 0]
    transaction["budget_state"] = [1, 0, 0, 2, 0, 1, 1, 0, 0]
    assert _replay_raw_native_control_journal(
        payload,
        axis_path=tmp_path / "axis.json",
        instance=instance,
    ) == "native control round budget state does not replay"

    payload["candidate_control_statistics"]["max_exact_calls_per_round"] = 1  # type: ignore[index]
    first_transaction["budget_state"] = [1, 0, 0, 1, 0, 1, 1, 0, 0]
    transaction["budget_state"] = [1, 0, 0, 1, 0, 11, 11, 0, 1]
    assert _replay_raw_native_control_journal(
        payload,
        axis_path=tmp_path / "axis.json",
        instance=instance,
    ) == "native control budget state does not match configured limits"

    transaction["budget_state"] = [0, -1, -1, 0, 0, 1, 1, 0, 0]
    assert _replay_raw_native_control_journal(
        payload,
        axis_path=tmp_path / "axis.json",
        instance=instance,
    ) == "native control budget state does not match configured limits"


def test_non_reference_internal_exact_rows_replay_round_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from evrptw.experiments import stage052_native_architecture_review as review

    instance = parse_schneider(Path(__file__).resolve().parents[1] / "data/schneider/c101C5.txt")
    events: list[dict[str, object]] = [
        {
            "event_type": "cache_event",
            "native_transaction_id": 42,
            "batch_ordinal": 0,
            "candidate_id": 0,
            "operation": "lookup",
            "result": "miss",
            "customer_sequence": ["C30"],
        },
        {
            "event_type": "cache_store_receipt",
            "native_transaction_id": 42,
            "batch_ordinal": 0,
            "candidate_id": 0,
            "status": "oversize_not_cached",
            "eviction_count": 0,
            "entry_bytes": 1,
            "customer_sequence": ["C30"],
        },
        {
            "event_type": "candidate_cache_transaction",
            "native_transaction_id": 42,
            "batch_ordinal": 0,
            "status": "committed",
            "lookups": 1,
            "hits": 0,
            "exact_stores": 0,
            "cache_statistics": [1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1],
            "implementation_internal": True,
            "round_budget_suppressed": False,
            "reference_distance_resolution": False,
            "iteration": 0,
            "budget_state": [1, 0, 0, 1, 0, 1, 1, 0, 0],
        },
    ]
    monkeypatch.setattr(
        review,
        "iter_verified_native_control_events",
        lambda *_args, **_kwargs: iter(events),
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "axis": "fixed_work",
        "fixed_work_budget": {"exact_calls": 10},
        "canonical_semantic_journal": {
            "native_control_events": {"native_source_sha256": "a" * 64}
        },
        "native_execution_statistics": {"control_journal_sha256": "a" * 64},
        "candidate_control_statistics": {
            "budget_skips": 0,
            "max_exact_calls_per_round": 1,
        },
        "cache_incremental_statistics": {
            "cache_lookups": 1,
            "cache_hits": 0,
            "cache_misses": 1,
            "cache_stores": 0,
            "cache_evictions": 0,
            "cache_oversize_not_cached": 1,
            "entries_current": 0,
            "entries_peak": 0,
            "bytes_current": 0,
            "bytes_peak": 0,
            "unique_route_evaluations": 1,
        },
    }

    assert (
        _replay_raw_native_control_journal(
            payload,
            axis_path=tmp_path / "axis.json",
            instance=instance,
        )
        is None
    )

    transaction = events[-1]
    transaction["budget_state"] = [1, 0, 0, 0, 1, 1, 1, 0, 0]
    assert _replay_raw_native_control_journal(
        payload,
        axis_path=tmp_path / "axis.json",
        instance=instance,
    ) == "native control round budget state does not replay"

    transaction["budget_state"] = [1, 0, 0, 1, 0, 1, 1, 0, 0]
    events.insert(
        0,
        {
            "event_type": "candidate_plan_decision",
            "native_transaction_id": 42,
            "batch_ordinal": 0,
            "candidate_id": 0,
            "rank": 1,
            "status": "selected",
            "transaction_status_code": 5,
            "implementation_internal": True,
        },
    )
    assert _replay_raw_native_control_journal(
        payload,
        axis_path=tmp_path / "axis.json",
        instance=instance,
    ) == "implementation-internal transaction contains plan decisions"


def test_current_raw_native_budget_skip_replays_plan_misses_and_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from evrptw.experiments import stage052_native_architecture_review as review

    instance = parse_schneider(Path(__file__).resolve().parents[1] / "data/schneider/c101C5.txt")
    statistics = [2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2]
    events: list[dict[str, object]] = [
        {
            "event_type": "candidate_plan_decision",
            "native_transaction_id": 41,
            "batch_ordinal": 0,
            "candidate_id": 0,
            "rank": 1,
            "status": "selected",
            "transaction_status_code": 3,
            "implementation_internal": False,
        },
        *[
            {
                "event_type": "cache_event",
                "native_transaction_id": 41,
                "batch_ordinal": 0,
                "candidate_id": 0,
                "operation": "lookup",
                "result": "miss",
                "customer_sequence": [customer],
            }
            for customer in ("C1", "C2")
        ],
        {
            "event_type": "candidate_control_budget",
            "native_transaction_id": 41,
            "batch_ordinal": 0,
            "candidate_id": 0,
            "transaction_status_code": 3,
            "implementation_internal": False,
            "status": "budget_skipped",
            "context": "legacy:relocate:candidate_pool",
            "requested": 2,
            "granted": 0,
            "remaining": 1,
            "round_remaining": 1,
            "exact_remaining": 10,
            "available": 1,
            "iteration": 0,
        },
        {
            "event_type": "candidate_cache_transaction",
            "native_transaction_id": 41,
            "batch_ordinal": 0,
            "status": "committed",
            "lookups": 2,
            "hits": 0,
            "exact_stores": 0,
            "cache_statistics": statistics,
            "implementation_internal": False,
            "round_budget_suppressed": False,
            "reference_distance_resolution": False,
            "iteration": 0,
            "budget_state": [1, 0, 0, 0, 1, 0, 0, 0, 0],
        },
    ]
    monkeypatch.setattr(
        review,
        "iter_verified_native_control_events",
        lambda *_args, **_kwargs: iter(events),
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "axis": "fixed_work",
        "fixed_work_budget": {"exact_calls": 10},
        "canonical_semantic_journal": {
            "native_control_events": {"native_source_sha256": "a" * 64}
        },
        "native_execution_statistics": {"control_journal_sha256": "a" * 64},
        "candidate_control_statistics": {
            "budget_skips": 1,
            "max_exact_calls_per_round": 1,
        },
        "cache_incremental_statistics": {
            name: statistics[index]
            for index, name in enumerate(
                (
                    "cache_lookups",
                    "cache_hits",
                    "cache_misses",
                    "cache_stores",
                    "cache_evictions",
                    "cache_oversize_not_cached",
                    "entries_current",
                    "entries_peak",
                    "bytes_current",
                    "bytes_peak",
                    "unique_route_evaluations",
                )
            )
        },
    }

    assert (
        _replay_raw_native_control_journal(
            payload,
            axis_path=tmp_path / "axis.json",
            instance=instance,
        )
        is None
    )

    events[3]["requested"] = 1
    assert _replay_raw_native_control_journal(
        payload,
        axis_path=tmp_path / "axis.json",
        instance=instance,
    ) == "native control budget skip does not replay"


@pytest.mark.parametrize(
    ("field", "value"),
    (
        (None, 0),
        ("round_remaining", 2),
        ("exact_remaining", 9),
        ("available", 2),
    ),
)
def test_current_canonical_budget_skip_replays_actual_round_and_exact_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str | None,
    value: int,
) -> None:
    from evrptw.experiments import stage052_native_architecture_review as review

    instance = parse_schneider(
        Path(__file__).resolve().parents[1] / "data/schneider/c101C5.txt"
    )
    charging_version = "test-charging-v1"
    objective_version = "test-objective-v1"

    def native(event_id: int, stream_code: int) -> dict[str, int]:
        return {
            "runtime_native_event_id": event_id,
            "runtime_native_stream_code": stream_code,
            "runtime_native_event_code": event_id,
            "runtime_native_lane_id": 0,
            "runtime_native_operator_id": 0,
            "runtime_native_iteration": 0,
            "runtime_native_transaction_id": 41,
            "runtime_native_subject_id": 0,
            "runtime_native_status_code": 0,
            "runtime_native_flags": 0,
        }

    context = {"lane": "legacy", "iteration": 0, "operator": "relocate"}
    events: list[dict[str, object]] = [
        {
            "semantic_stream": "candidate_transaction",
            "event_type": "candidate_control_round",
            "status": "started",
            "lane": "all",
            "iteration": 0,
            "budget": 1,
        },
        {
            "semantic_stream": "candidate_transaction",
            "event_type": "candidate_plan_decision",
            **context,
            "candidate_id": 0,
            "rank": 1,
            "status": "selected",
            "transaction_status_code": 3,
            "native_transaction_id": 41,
            "native_telemetry": native(1, 3),
        },
    ]
    for event_id, customer in enumerate(("C1", "C2"), start=2):
        sequence = (customer,)
        route_key = canonical_route_key(sequence)
        events.append(
            {
                "semantic_stream": "cache",
                "event_type": "cache_lookup_result",
                **context,
                "candidate_id": 0,
                "native_transaction_id": 41,
                "route_key": route_key,
                "cache_key_digest": RouteCacheKey(
                    instance_hash=canonical_instance_hash(instance),
                    customer_sequence=sequence,
                    charging_configuration_version=charging_version,
                    objective_schema_version=objective_version,
                ).digest,
                "status": "miss",
                "cache_scope": "committed",
                "current_entries": 0,
                "current_bytes": 0,
                "native_telemetry": native(event_id, 6),
            }
        )
    budget_event: dict[str, object] = {
        "semantic_stream": "candidate_transaction",
        "event_type": "candidate_control_budget",
        **context,
        "context": "legacy:relocate:candidate_pool",
        "candidate_id": 0,
        "native_transaction_id": 41,
        "transaction_status_code": 3,
        "status": "budget_skipped",
        "requested": 2,
        "granted": 0,
        "remaining": 1,
        "round_remaining": 1,
        "exact_remaining": 10,
        "available": 1,
        "native_telemetry": native(4, 3),
    }
    if field is not None:
        budget_event[field] = value
        if field == "round_remaining":
            budget_event["remaining"] = value
    events.extend(
        [
            budget_event,
            {
                "semantic_stream": "candidate_transaction",
                "event_type": "candidate_control_round",
                "status": "completed",
                "lane": "all",
                "iteration": 0,
                "budget": 1,
                "used": 0,
                "remainder": 1,
            },
        ]
    )
    monkeypatch.setattr(
        review,
        "iter_verified_semantic_journal",
        lambda *_args, **_kwargs: iter(events),
    )
    monkeypatch.setattr(
        review,
        "_native_canonical_receipt_error",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        review,
        "_replay_physical_telemetry",
        lambda *_args, **_kwargs: None,
    )
    cache_metrics = {
        "cache_lookups": 2,
        "cache_hits": 0,
        "cache_misses": 2,
        "cache_stores": 0,
        "cache_evictions": 0,
        "cache_oversize_not_cached": 0,
        "entries_current": 0,
        "entries_peak": 0,
        "bytes_current": 0,
        "bytes_peak": 0,
        "unique_route_evaluations": 2,
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "canonical_semantic_journal": {"schema_version": "test"},
        "mode": "full_native_alns",
        "axis": "fixed_work",
        "fixed_work_budget": {"exact_calls": 10},
        "candidate_work_hash": "",
        "route_result_hash": "",
        "exact_started_calls": 0,
        "exact_completed_calls": 0,
        "exact_interrupted_calls": 0,
        "candidate_control_statistics": {
            "enabled": True,
            "max_exact_calls_per_round": 1,
            "candidate_decisions": 1,
            "selected_candidates": 1,
            "skipped_candidates": 0,
            "budget_events": 1,
            "budget_skips": 1,
            "completed_rounds": 1,
            "maximum_exact_calls_per_round": 0,
            "total_round_remainder": 1,
        },
        "cache_incremental_statistics": {
            **cache_metrics,
            "config": {
                "enabled": True,
                "eviction_policy": "lru",
                "max_entries": 8,
                "max_memory_bytes": 1_000_000,
                "charging_configuration_version": charging_version,
                "objective_schema_version": objective_version,
                "instance_hash": canonical_instance_hash(instance),
            },
            "route_cache": {
                **cache_metrics,
                "eviction_policy": "lru",
                "max_entries": 8,
                "max_memory_bytes": 1_000_000,
                "instance_hash": canonical_instance_hash(instance),
            },
        },
    }

    assert _replay_canonical_journal(
        payload,
        axis_path=tmp_path / "axis.json",
        instance=instance,
    ) == (None if field is None else "native canonical budget skip does not replay")


def test_current_canonical_budget_skip_rejects_eviction_reassigned_to_another_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from evrptw.experiments import stage052_native_architecture_review as review

    instance = parse_schneider(
        Path(__file__).resolve().parents[1] / "data/schneider/c101C5.txt"
    )
    charging_version = "test-charging-v1"
    objective_version = "test-objective-v1"
    first_sequence = ("C30",)
    second_sequence = ("C12",)
    first_exact = solve_exact_charging(instance, first_sequence)
    second_exact = solve_exact_charging(instance, second_sequence)
    assert first_exact.feasible and second_exact.feasible
    first_key = canonical_route_key(first_sequence)
    second_key = canonical_route_key(second_sequence)
    first_bytes = review.estimate_cache_entry_bytes(first_exact)
    second_bytes = review.estimate_cache_entry_bytes(second_exact)

    def cache_digest(sequence: tuple[str, ...]) -> str:
        return RouteCacheKey(
            instance_hash=canonical_instance_hash(instance),
            customer_sequence=sequence,
            charging_configuration_version=charging_version,
            objective_schema_version=objective_version,
        ).digest

    native_event_id = 0

    def native(stream_code: int, transaction_id: int, subject_id: int = 0) -> dict[str, int]:
        nonlocal native_event_id
        native_event_id += 1
        return {
            "runtime_native_event_id": native_event_id,
            "runtime_native_stream_code": stream_code,
            "runtime_native_event_code": native_event_id,
            "runtime_native_lane_id": 0,
            "runtime_native_operator_id": 0,
            "runtime_native_iteration": 0,
            "runtime_native_transaction_id": transaction_id,
            "runtime_native_subject_id": subject_id,
            "runtime_native_status_code": 0,
            "runtime_native_flags": 0,
        }

    context = {"lane": "legacy", "iteration": 0, "operator": "relocate"}
    events: list[dict[str, object]] = [
        {
            "semantic_stream": "exact_work",
            "event_type": "exact_batch_started",
            **context,
            "customer_sequences": [list(first_sequence)],
            "requested_calls": 1,
            "started_calls": 1,
            "native_telemetry": native(4, 40),
        },
        {
            "semantic_stream": "exact_result",
            "event_type": "exact_route_result",
            **context,
            "route_key": first_key,
            "exact_started": True,
            "exact_completed": True,
            "status": "completed",
            "feasible": True,
            "failure_reason": first_exact.failure_reason,
            "native_telemetry": native(5, 40),
        },
        {
            "semantic_stream": "cache",
            "event_type": "cache_lifecycle",
            **context,
            "candidate_id": 0,
            "native_transaction_id": 40,
            "route_key": first_key,
            "status": "store",
            "cache_key_digest": cache_digest(first_sequence),
            "entry_bytes": first_bytes,
            "current_entries": 1,
            "current_bytes": first_bytes,
            "native_telemetry": native(6, 40),
        },
        {
            "semantic_stream": "candidate_transaction",
            "event_type": "candidate_control_round",
            "status": "started",
            "lane": "all",
            "iteration": 0,
            "budget": 1,
        },
        {
            "semantic_stream": "candidate_transaction",
            "event_type": "candidate_plan_decision",
            **context,
            "candidate_id": 0,
            "rank": 1,
            "status": "selected",
            "transaction_status_code": 3,
            "native_transaction_id": 41,
            "native_telemetry": native(3, 41),
        },
    ]
    for customer in ("C1", "C2"):
        sequence = (customer,)
        events.append(
            {
                "semantic_stream": "cache",
                "event_type": "cache_lookup_result",
                **context,
                "candidate_id": 0,
                "native_transaction_id": 41,
                "route_key": canonical_route_key(sequence),
                "cache_key_digest": cache_digest(sequence),
                "status": "miss",
                "cache_scope": "committed",
                "current_entries": 1,
                "current_bytes": first_bytes,
                "native_telemetry": native(6, 41),
            }
        )
    events.extend(
        [
            {
                "semantic_stream": "candidate_transaction",
                "event_type": "candidate_control_budget",
                **context,
                "context": "legacy:relocate:candidate_pool",
                "candidate_id": 0,
                "native_transaction_id": 41,
                "transaction_status_code": 3,
                "status": "budget_skipped",
                "requested": 2,
                "granted": 0,
                "remaining": 1,
                "round_remaining": 1,
                "exact_remaining": 9,
                "available": 1,
                "native_telemetry": native(3, 41),
            },
            {
                "semantic_stream": "candidate_transaction",
                "event_type": "candidate_plan_decision",
                **context,
                "candidate_id": 1,
                "rank": 2,
                "status": "selected",
                "transaction_status_code": 5,
                "native_transaction_id": 41,
                "native_telemetry": native(3, 41, 1),
            },
            {
                "semantic_stream": "cache",
                "event_type": "cache_lifecycle",
                **context,
                "candidate_id": 0,
                "native_transaction_id": 41,
                "route_key": first_key,
                "status": "evict",
                "cache_key_digest": cache_digest(first_sequence),
                "current_entries": 1,
                "current_bytes": second_bytes,
                "native_telemetry": native(6, 41),
            },
            {
                "semantic_stream": "exact_work",
                "event_type": "exact_batch_started",
                **context,
                "customer_sequences": [list(second_sequence)],
                "requested_calls": 1,
                "started_calls": 1,
                "native_telemetry": native(4, 41, 1),
            },
            {
                "semantic_stream": "exact_result",
                "event_type": "exact_route_result",
                **context,
                "route_key": second_key,
                "exact_started": True,
                "exact_completed": True,
                "status": "completed",
                "feasible": True,
                "failure_reason": second_exact.failure_reason,
                "native_telemetry": native(5, 41, 1),
            },
            {
                "semantic_stream": "cache",
                "event_type": "cache_lifecycle",
                **context,
                "candidate_id": 1,
                "native_transaction_id": 41,
                "route_key": second_key,
                "status": "store",
                "cache_key_digest": cache_digest(second_sequence),
                "entry_bytes": second_bytes,
                "current_entries": 1,
                "current_bytes": second_bytes,
                "native_telemetry": native(6, 41, 1),
            },
            {
                "semantic_stream": "candidate_transaction",
                "event_type": "candidate_control_round",
                "status": "completed",
                "lane": "all",
                "iteration": 0,
                "budget": 1,
                "used": 1,
                "remainder": 0,
            },
        ]
    )
    monkeypatch.setattr(
        review,
        "iter_verified_semantic_journal",
        lambda *_args, **_kwargs: iter(events),
    )
    monkeypatch.setattr(review, "_native_canonical_receipt_error", lambda *_args: None)
    monkeypatch.setattr(review, "_replay_physical_telemetry", lambda *_args: None)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "canonical_semantic_journal": {"schema_version": "test"},
        "mode": "full_native_alns",
        "axis": "fixed_work",
        "fixed_work_budget": {"exact_calls": 10},
        "candidate_work_hash": "",
        "route_result_hash": "",
        "exact_started_calls": 2,
        "exact_completed_calls": 2,
        "exact_interrupted_calls": 0,
        "candidate_control_statistics": {
            "enabled": True,
            "max_exact_calls_per_round": 1,
            "candidate_decisions": 2,
            "selected_candidates": 2,
            "skipped_candidates": 0,
            "budget_events": 1,
            "budget_skips": 1,
            "completed_rounds": 1,
            "maximum_exact_calls_per_round": 1,
            "total_round_remainder": 0,
        },
        "cache_incremental_statistics": {
            "cache_lookups": 2,
            "cache_hits": 0,
            "cache_misses": 2,
            "cache_stores": 2,
            "cache_evictions": 1,
            "cache_oversize_not_cached": 0,
            "entries_current": 1,
            "entries_peak": 1,
            "bytes_current": second_bytes,
            "bytes_peak": max(first_bytes, second_bytes),
            "unique_route_evaluations": 4,
            "config": {
                "enabled": True,
                "eviction_policy": "lru",
                "max_entries": 1,
                "max_memory_bytes": 1_000_000,
                "charging_configuration_version": charging_version,
                "objective_schema_version": objective_version,
                "instance_hash": canonical_instance_hash(instance),
            },
        },
    }

    assert _replay_canonical_journal(
        payload,
        axis_path=tmp_path / "axis.json",
        instance=instance,
    ) == "cache eviction transaction does not match pending store"


@pytest.mark.parametrize("missing_stream", sorted(_REQUIRED_CAUSAL_STREAMS))
def test_unified_causal_ids_reject_missing_required_domain(
    missing_stream: str,
) -> None:
    streams = _complete_causal_streams(exclude=missing_stream)
    events = _canonical_semantic_event_sequence(streams)

    with pytest.raises(ValueError):
        _canonical_semantic_events(_full_native_inline_payload(streams, events))


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
        _canonical_semantic_events(_full_native_inline_payload(streams, corrupted))


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
        "source_customer_sequences_sha256": canonical_customer_sequences_sha256(sequences),
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


def test_performance_identity_blocks_keep_c_r_rc_mode_e2e_separate(tmp_path: Path) -> None:
    blocks = _performance_identity_blocks(_plan("paired", tmp_path))

    assert blocks
    for block in blocks:
        assert len({task.repeat for task in block}) == 1
        assert len({task.axis for task in block}) == 1
        assert len({performance_family_for_instance(task.instance_name) for task in block}) == 1


def test_pilot_plan_has_180_wall_clock_axes_and_independent_labels(
    tmp_path: Path,
) -> None:
    plan = _plan("pilot", tmp_path)
    labels = run_labels_for_scope("pilot", 1)

    assert len(plan) * len(MODES) == expected_axis_count("pilot") == 180
    assert {task.axis for task in plan} == {"wall_clock_30"}
    assert len(set(labels.values())) == len(MODES)
    assert all("_pilot_attempt01" in label for label in labels.values())


def test_pilot_gate_requires_signed_qualified_paired_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path(__file__).resolve().parents[1]
    revision = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    wheel = "b" * 64
    native = "c" * 64
    scheduler = "d" * 64
    profile = "e" * 64
    path = tmp_path / "paired-review.json"
    reviewer_source = Path(write_review.__code__.co_filename).resolve()
    raw_inventory = {
        "axis_count": 360,
        "json_bytes": 360,
        "sidecar_bytes": 360,
        "tree_sha256": "0" * 64,
        "algorithm": "test inventory",
        "by_mode": {},
    }
    payload = {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "scope": "paired",
        "attempt": 8,
        "axis_count": 360,
        "axis_replay_passed": True,
        "semantic_gates_passed": True,
        "current_schema_qualified": True,
        "qualification_passed": True,
        "review_status": "COMPARISON_COMPLETE_QUALIFIED",
        "replay_failures": [],
        "producer_identity": {
            "repository_revisions": [revision],
            "wheel_sha256": [wheel],
            "native_sha256": [native],
            "scheduler_sha256": [scheduler],
            "performance_profile_sha256": [profile],
            "run_labels": sorted(run_labels_for_scope("paired", 8).values()),
        },
        "reviewer_provenance": {
            "repository_revision": revision,
            "source_path": "src/evrptw/experiments/stage052_native_architecture_review.py",
            "source_sha256": hashlib.sha256(reviewer_source.read_bytes()).hexdigest(),
        },
        "raw_axis_inventory": raw_inventory,
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
            "median": None,
            "p95": None,
            "maximum": None,
            "reason": "test",
        },
    }
    report = tmp_path / "paired-review.md"
    write_review(payload, output_json=path, output_markdown=report)
    execution_path = tmp_path / "paired-review_execution.json"
    monkeypatch.setattr(
        "evrptw.experiments.stage052_native_architectures._recompute_paired_review_inventory",
        lambda _root, _attempt: (raw_inventory, frozenset({SCHEMA_VERSION})),
    )
    assert PAIRED_REVIEW_SCHEMA_VERSION == REVIEW_SCHEMA_VERSION
    gate = _load_qualified_paired_review(
        path,
        review_execution_path=execution_path,
        paired_results_root=tmp_path,
        paired_attempt=8,
        revision=revision,
        wheel_sha256=wheel,
        native_sha256=native,
        scheduler_sha256=scheduler,
        performance_profile_sha256=profile,
    )
    assert gate["axis_count"] == 360
    assert gate["qualification_passed"] is True

    tampered = dict(payload)
    tampered["qualification_passed"] = False
    rejected = tmp_path / "paired-review-rejected.json"
    _write_signed_json(rejected, tampered)
    with pytest.raises(RuntimeError, match="qualified 360/360"):
        _load_qualified_paired_review(
            rejected,
            review_execution_path=execution_path,
            paired_results_root=tmp_path,
            paired_attempt=8,
            revision=revision,
            wheel_sha256=wheel,
            native_sha256=native,
            scheduler_sha256=scheduler,
            performance_profile_sha256=profile,
        )

    for index, invalid_payload in enumerate(
        (
            {**payload, "schema_version": "stage05.2-native-architecture-review-v10"},
            {key: value for key, value in payload.items() if key != "current_schema_qualified"},
            {**payload, "current_schema_qualified": False},
        )
    ):
        invalid_path = tmp_path / f"paired-review-schema-rejected-{index}.json"
        _write_signed_json(invalid_path, invalid_payload)
        with pytest.raises(RuntimeError, match="qualified 360/360"):
            _load_qualified_paired_review(
                invalid_path,
                review_execution_path=execution_path,
                paired_results_root=tmp_path,
                paired_attempt=8,
                revision=revision,
                wheel_sha256=wheel,
                native_sha256=native,
                scheduler_sha256=scheduler,
                performance_profile_sha256=profile,
            )


def test_campaign_gate_requires_signed_qualified_calibration_review(
    tmp_path: Path,
) -> None:
    revision = "a" * 40
    tree = "b" * 40
    source = "c" * 64
    wheel = "d" * 64
    native = "e" * 64
    scheduler = "f" * 64
    profile_canonical = "1" * 64
    calibration_run_label = "stage05.2_native_architecture_performance_calibration_attempt01"
    calibration_dir = tmp_path / calibration_run_label
    calibration_dir.mkdir()
    calibration_receipt = calibration_dir / "calibration_receipt.json"
    _write_signed_json(calibration_receipt, {"status": "calibrated"})
    calibration_receipt_sha = hashlib.sha256(calibration_receipt.read_bytes()).hexdigest()
    profile_path = calibration_dir / "frozen_performance_profile.json"
    _write_signed_json(profile_path, {"canonical_sha256": profile_canonical})
    profile_file_sha = hashlib.sha256(profile_path.read_bytes()).hexdigest()
    review_path = calibration_dir / "review" / "calibration_review_receipt.json"
    assert PRODUCER_CALIBRATION_REVIEW_SCHEMA_VERSION == CALIBRATION_REVIEW_SCHEMA_VERSION
    payload = {
        "schema_version": PRODUCER_CALIBRATION_REVIEW_SCHEMA_VERSION,
        "status": "qualified",
        "qualification": "QUALIFIED_FOR_ATTEMPT08",
        "calibration_run_label": calibration_run_label,
        "storage_alias": "stage052-performance-calibration-run",
        "calibration_receipt_relative_path": "calibration_receipt.json",
        "calibration_receipt_sha256": calibration_receipt_sha,
        "calibration_manifest_sha256": "2" * 64,
        "profile_relative_path": "frozen_performance_profile.json",
        "profile_sha256": profile_file_sha,
        "profile_canonical_sha256": profile_canonical,
        "rederived_profile_canonical_sha256": profile_canonical,
        "repository_revision": revision,
        "git_tree": tree,
        "source_manifest_sha256": source,
        "selected_build_profile": "portable-lto",
        "wheel_sha256": wheel,
        "native_sha256": native,
        "scheduler_sha256": scheduler,
        "observation_count": 30,
        "raw_axis_replay_count": 360,
        "resource_summary_replay_count": 60,
        "memory_rejection_replay_count": 2,
        "fixed_work_semantics_identical": True,
        "validator_objective_replay_passed": True,
        "queue_full_count": 0,
        "rejected_count": 0,
        "swap_used": False,
        "selector_recomputation_passed": True,
        "current_schema_qualified": True,
        "raw_axis_schema_versions": [SCHEMA_VERSION],
        "telemetry_raw_axis_schema_versions": [SCHEMA_VERSION],
        "resource_evidence_schema_versions": [
            "stage05.2-calibration-resource-evidence-v5"
        ],
        "reviewer_source_sha256": "3" * 64,
        "review_seconds": 1.0,
        "formal_started": False,
        "cuda_started": False,
        "attempt08_started": False,
    }
    _write_signed_json(review_path, payload)
    review_execution_path = calibration_dir / "review" / "review_execution.json"
    _write_signed_json(
        review_execution_path,
        {
            "schema_version": "experiment-review-execution-v1",
            "run_label": calibration_run_label,
            "status": "completed",
            "finalized": True,
            "exit_code": 0,
            "reviewer_module_name": ("evrptw.experiments.stage052_performance_calibration_review"),
            "reviewer_installed_distribution_digest": "3" * 64,
            "raw_manifest_sha256_before": "2" * 64,
            "raw_manifest_sha256_after": "2" * 64,
            "raw_manifest_unchanged": True,
            "review_manifest_sha256": hashlib.sha256(review_path.read_bytes()).hexdigest(),
            "command": [
                "/usr/bin/python",
                "-m",
                "evrptw.experiments.stage052_performance_calibration_review",
            ],
        },
    )

    gate = _load_qualified_calibration_review(
        review_path,
        review_execution_path=review_execution_path,
        calibration_run_label=calibration_run_label,
        revision=revision,
        git_tree=tree,
        source_manifest_sha256=source,
        wheel_sha256=wheel,
        native_sha256=native,
        scheduler_sha256=scheduler,
        performance_profile_path=profile_path,
        performance_profile_file_sha256=profile_file_sha,
        performance_profile_sha256=profile_canonical,
        selected_build_profile="portable-lto",
    )
    assert gate["qualification"] == "QUALIFIED_FOR_ATTEMPT08"
    assert gate["raw_axis_replay_count"] == 360

    legacy = dict(payload)
    legacy["schema_version"] = (
        "stage05.2-native-architecture-performance-calibration-review-v1"
    )
    legacy_path = calibration_dir / "review" / "legacy-v1.json"
    _write_signed_json(legacy_path, legacy)
    with pytest.raises(RuntimeError, match="qualified independent replay"):
        _load_qualified_calibration_review(
            legacy_path,
            review_execution_path=review_execution_path,
            calibration_run_label=calibration_run_label,
            revision=revision,
            git_tree=tree,
            source_manifest_sha256=source,
            wheel_sha256=wheel,
            native_sha256=native,
            scheduler_sha256=scheduler,
            performance_profile_path=profile_path,
            performance_profile_file_sha256=profile_file_sha,
            performance_profile_sha256=profile_canonical,
            selected_build_profile="portable-lto",
        )

    rejected = dict(payload)
    rejected["selector_recomputation_passed"] = False
    rejected_path = calibration_dir / "review" / "rejected.json"
    _write_signed_json(rejected_path, rejected)
    with pytest.raises(RuntimeError, match="qualified independent replay"):
        _load_qualified_calibration_review(
            rejected_path,
            review_execution_path=review_execution_path,
            calibration_run_label=calibration_run_label,
            revision=revision,
            git_tree=tree,
            source_manifest_sha256=source,
            wheel_sha256=wheel,
            native_sha256=native,
            scheduler_sha256=scheduler,
            performance_profile_path=profile_path,
            performance_profile_file_sha256=profile_file_sha,
            performance_profile_sha256=profile_canonical,
            selected_build_profile="portable-lto",
        )


def _review_fixture_records(root: Path, evidence_root: Path) -> tuple[ReviewRecord, ...]:
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
    assert review["current_schema_qualified"] is False
    assert review["qualification_passed"] is False
    assert "五模式事实表" in render_report(review)


def test_v11_profile_gate_uses_process_affinity_and_historical_gate_keeps_thread_affinity() -> None:
    wave = {
        "actual_affinity_union": [0, 1, 2, 3],
        "thread_affinity_union": [0, 1],
    }

    assert _mode_wave_affinity_matches_profile(
        wave,
        comparison_schema=SCHEMA_VERSION,
        allowed_cpu_ids=(0, 1, 2, 3),
    )
    assert not _mode_wave_affinity_matches_profile(
        wave,
        comparison_schema=PRIOR_PROFILE_COMPARISON_SCHEMA_VERSION,
        allowed_cpu_ids=(0, 1, 2, 3),
    )


def test_only_current_comparison_schema_can_be_newly_qualified(tmp_path: Path) -> None:
    current = ReviewRecord(tmp_path / "current.json", {"schema_version": SCHEMA_VERSION})
    historical = ReviewRecord(
        tmp_path / "historical.json",
        {"schema_version": PRIOR_PROFILE_COMPARISON_SCHEMA_VERSION},
    )

    assert _records_use_current_profile_schema((current,))
    assert not _records_use_current_profile_schema((current, historical))
    assert not _records_use_current_profile_schema((historical,))


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
            "schema_version": PREVIOUS_COMPARISON_SCHEMA_VERSION,
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
        "schema_version": PREVIOUS_COMPARISON_SCHEMA_VERSION,
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
                if entry != "evrptw/experiments/stage052_native_architectures.py"
            )
            invalid_entries.pop(committed_entry)
        else:
            invalid_entries["evrptw/uncommitted.py"] = "a" * 64
        invalid_receipt = {**receipt, "wheel_entry_sha256": invalid_entries}
        with pytest.raises(RuntimeError, match="inventory does not match Git"):
            _review_build_attestation({**manifest, "wheel_receipt": invalid_receipt})

    modified_runner_entries = dict(wheel_entries)
    modified_runner_entries["evrptw/experiments/stage052_native_architectures.py"] = "0" * 64
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
            {"semantic_trajectory": [{**trajectory_event, "candidate_id": "0" * 64}]}
        )
    assert _semantic_trajectory({"semantic_trajectory": [trajectory_event]}) == [trajectory_event]
    with pytest.raises(ValueError, match="candidate_route_keys"):
        _canonical_trace_event({"event_type": "candidate_state"})
    with pytest.raises(ValueError, match="lane projection"):
        _semantic_trajectory({"semantic_trajectory": [{"lane": "constraint_lane"}]})


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
        },
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
            "canonical_semantic_events": _canonical_semantic_event_sequence(baseline_streams),
        }
    )
    candidate = _canonical_semantic_events(
        {
            "canonical_semantic_streams": candidate_streams,
            "canonical_semantic_events": _canonical_semantic_event_sequence(candidate_streams),
        }
    )
    divergence = _describe_first_divergence(baseline, candidate)

    assert _common_prefix(baseline, candidate) == 0
    assert divergence is not None
    assert divergence["lane"] == "legacy"
    assert divergence["iteration"] == 7
    assert divergence["operator"] == "route_merge"
    assert divergence["candidate_id"] == "candidate-7"
    assert divergence["differing_fields"] == {"weight": {"baseline": 2.0, "candidate": 3.0}}


def test_comparison_projection_separates_physical_and_logical_screening() -> None:
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

    def payload(
        *,
        batch_row: int,
        physical_repeats: int = 1,
        status: str = "pass",
        reason: str = "",
        logical_status: str = "candidate_proposed",
        logical_reason: str = "",
    ) -> dict[str, object]:
        streams: dict[str, list[dict[str, object]]] = {name: [] for name in stream_names}
        event_id = 1
        for repeat in range(physical_repeats):
            streams["screening"].append(
                {
                    "semantic_event_id": event_id,
                    "stream_ordinal": repeat,
                    "event_type": "screening_decision",
                    "lane": "quality_shadow",
                    "iteration": 3,
                    "operator": "route_segment_destroy",
                    "route_key": "route:2:C1",
                    "status": status,
                    "reason": reason,
                    "first_failed_check": "" if status == "pass" else "route_structure",
                    "negative_cache_hit": False,
                    "exact_call_blocked": status != "pass",
                    "checks": [
                        {
                            "check": "route_structure",
                            "status": status,
                            "value": status == "pass",
                        }
                    ],
                    "batch_row": batch_row + repeat,
                    "queue_wait_seconds": float(batch_row + repeat),
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
        logical_trajectory = {
            **trajectory[0],
            "status": logical_status,
            "reason": logical_reason,
        }
        logical_identity = {
            **trajectory_identity,
            "status": logical_status,
        }
        logical_trajectory["candidate_id"] = hashlib.sha256(
            json.dumps(
                logical_identity,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        return {
            "schema_version": SCHEMA_VERSION,
            "semantic_trajectory": [logical_trajectory],
            "canonical_semantic_streams": streams,
            "canonical_semantic_events": _canonical_semantic_event_sequence(streams),
        }

    baseline = payload(batch_row=0)
    candidate = payload(batch_row=99)
    assert baseline["canonical_semantic_events"] != candidate["canonical_semantic_events"]
    assert _comparison_semantic_events(baseline) == _comparison_semantic_events(candidate)
    repeated_physical_decision = payload(batch_row=99, physical_repeats=3)
    assert _comparison_semantic_events(baseline) == _comparison_semantic_events(
        repeated_physical_decision
    )
    physical_decision = payload(
        batch_row=99,
        status="rejected",
        reason="route_structure_prefilter",
    )
    assert _comparison_semantic_events(baseline) == _comparison_semantic_events(physical_decision)
    changed_decision = payload(
        batch_row=99,
        logical_status="candidate_control_skipped",
        logical_reason="route_structure_prefilter",
    )
    assert _comparison_semantic_events(baseline) != _comparison_semantic_events(changed_decision)

    malformed_checks = payload(batch_row=100)
    malformed_streams = malformed_checks["canonical_semantic_streams"]
    assert isinstance(malformed_streams, dict)
    malformed_screening = malformed_streams["screening"]
    assert isinstance(malformed_screening, list)
    malformed_screening[0]["checks"] = ["not-an-object"]
    malformed_checks["canonical_semantic_events"] = _canonical_semantic_event_sequence(
        malformed_streams
    )
    with pytest.raises(ValueError, match="screening checks are invalid"):
        _comparison_semantic_events(malformed_checks)


@pytest.mark.external_data
def test_physical_screening_replay_rejects_self_consistent_decision_corruption() -> None:
    instance = parse_schneider(Path("data/schneider/c101C5.txt"))
    sequence = (instance.customers[0].name,)
    replayed = screen_route_candidate(instance, sequence, full=True)
    event: dict[str, object] = {
        "event_type": "screening_decision",
        "route_key": "route:" + "|".join(f"{len(customer)}:{customer}" for customer in sequence),
        "lane": "legacy",
        "iteration": 0,
        "operator": "route_merge",
        "status": "pass" if replayed.accepted else "rejected",
        "first_failed_check": replayed.first_failed_check,
        "reason": replayed.reason,
        "checks": [asdict(check) for check in replayed.checks],
        "demand": replayed.demand,
        "min_time_window_slack": replayed.min_time_window_slack,
        "distance_lower_bound": replayed.distance_lower_bound,
        "single_segment_reachable": replayed.single_segment_reachable,
        "structural_energy_lower_bound": replayed.structural_energy_lower_bound,
        "negative_cache_hit": False,
        "exact_call_blocked": not replayed.accepted,
    }
    payload = {
        "canonical_semantic_streams": {"screening": [event]},
        "screening_statistics": {
            "screening_calls": 1,
            "screening_passes": int(replayed.accepted),
            "screening_rejections": int(not replayed.accepted),
            "screening_cache_hits": 0,
            "screening_exact_call_blocked": int(not replayed.accepted),
        },
    }
    assert _replay_physical_screening(payload, instance) is None

    raw_checks = event["checks"]
    assert isinstance(raw_checks, list)
    first_check = raw_checks[0]
    assert isinstance(first_check, dict)
    original_value = first_check["value"]
    first_check["value"] = not original_value
    assert _replay_physical_screening(payload, instance) == (
        "physical screening row 0 check replay mismatch"
    )
    first_check["value"] = original_value
    original_reason = first_check["reason"]
    first_check["reason"] = "tampered"
    assert _replay_physical_screening(payload, instance) == (
        "physical screening row 0 check replay mismatch"
    )
    first_check["reason"] = original_reason

    event["distance_increment_lower_bound"] = float("nan")
    assert _replay_physical_screening(payload, instance) == (
        "physical screening row 0 increment bound is invalid"
    )
    event["distance_increment_lower_bound"] = None

    original_status = event["status"]
    event["status"] = "rejected" if replayed.accepted else "pass"
    assert _replay_physical_screening(payload, instance) == (
        "physical screening row 0 decision replay mismatch"
    )
    event["status"] = original_status

    statistics = payload["screening_statistics"]
    assert isinstance(statistics, dict)
    statistics["screening_calls"] = 2
    assert _replay_physical_screening(payload, instance) == (
        "physical screening statistics do not conserve calls"
    )
    payload["canonical_semantic_streams"] = {"screening": []}
    assert _replay_physical_screening(payload, instance) == ("physical screening stream is empty")


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
    payload["schema_version"] = "stage05.2-native-architecture-comparison-v6"
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
            "reason": ("semantic trajectory replay failed: v4 evidence is missing"),
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
        "sample_count": 0,
        "median": None,
        "p95": None,
        "maximum": None,
        "reason": "native candidate-screening occupancy is not recorded",
    }


def test_cuda_condition_uses_median_candidate_screening_occupancy(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    records = list(_review_fixture_records(root, tmp_path))
    host = next(record for record in records if record.mode.value == "host_scheduler")
    assert isinstance(host.payload, dict)
    host.payload["native_execution_statistics"] = {
        "candidate_screening_occupancies": [1, 1, 64],
    }

    condition = _scheduler_screening_occupancy((host,))

    assert condition == {
        "available": True,
        "condition_met": False,
        "sample_count": 3,
        "median": 1.0,
        "p95": 64,
        "maximum": 64,
        "reason": "native candidate-screening median occupancy replayed",
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
        "attempt": 1,
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
        "raw_axis_inventory": {
            "axis_count": 0,
            "json_bytes": 0,
            "sidecar_bytes": 0,
            "tree_sha256": "0" * 64,
            "algorithm": "test inventory",
            "by_mode": {},
        },
    }

    write_review(review, output_json=output_json, output_markdown=output_markdown)

    manifest = json.loads((tmp_path / "paired_review_manifest.json").read_text(encoding="utf-8"))
    assert manifest["reviewer_provenance"] == review["reviewer_provenance"]
    assert manifest["producer_identity"] == review["producer_identity"]
    assert (
        manifest["files"][output_json.name] == hashlib.sha256(output_json.read_bytes()).hexdigest()
    )
    assert (
        manifest["files"][output_markdown.name]
        == hashlib.sha256(output_markdown.read_bytes()).hexdigest()
    )
    manifest_path = tmp_path / "paired_review_manifest.json"
    assert manifest_path.with_suffix(".json.sha256").is_file()
    execution_path = tmp_path / "paired_review_execution.json"
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    assert execution["raw_manifest_unchanged"] is True
    assert (
        execution["review_manifest_sha256"]
        == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    )
    assert execution_path.with_suffix(".json.sha256").is_file()


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

    with NativeHostScheduler(endpoint) as scheduler:
        written = _run_group(replace(task, scheduler_process_id=scheduler.process_id))

    assert len(written) == len(MODES)
    payloads = [
        _axis_payload_with_persistence(Path(path), json.loads(Path(path).read_bytes()))
        for path in written
    ]
    assert {payload["mode"] for payload in payloads} == {mode.value for mode in MODES}
    assert {payload["scheduler_sha256"] for payload in payloads} == {"f" * 64}
    status_by_mode = {payload["mode"]: payload["status"] for payload in payloads}
    assert status_by_mode == {
        "current_stage052": "completed",
        "python_candidate_control": "completed",
        "per_solve_runtime": "completed",
        "full_native_alns": "completed",
        "host_scheduler": "completed",
    }
    for raw_path, payload in zip(written, payloads, strict=True):
        axis_path = Path(raw_path)
        if payload["status"] != "completed":
            assert "canonical_semantic_journal" not in payload
            continue
        assert "canonical_semantic_events" not in payload
        assert "canonical_semantic_streams" not in payload
        descriptor = payload["canonical_semantic_journal"]
        journal_bundle = axis_path.parent / descriptor["path"]
        journal_path = journal_bundle / descriptor["event_path"]
        journal_sidecar = journal_bundle / descriptor["sidecar_path"]
        axis_sidecar = axis_path.with_suffix(axis_path.suffix + ".sha256")
        persistence_receipt = axis_path.with_suffix(axis_path.suffix + ".persistence")
        persistence_receipt_sidecar = persistence_receipt.with_suffix(
            persistence_receipt.suffix + ".sha256"
        )
        task_receipt = axis_path.with_suffix(axis_path.suffix + ".native-work-tasks.jsonl")
        task_receipt_files = (
            (
                task_receipt,
                task_receipt.with_suffix(task_receipt.suffix + ".sha256"),
            )
            if task_receipt.is_file()
            else ()
        )
        assert journal_bundle.is_dir()
        assert journal_path.is_file() and journal_sidecar.is_file()
        assert descriptor["physical_telemetry"]["schema_version"].startswith(
            "stage05.2-physical-telemetry-"
        )
        if payload["mode"] in {"full_native_alns", "host_scheduler"}:
            control_rows = tuple(iter_verified_native_control_events(axis_path, descriptor))
            assert control_rows
            assert control_rows[-1]["event_type"] == "candidate_cache_transaction"
            assert (
                descriptor["native_control_events"]["native_source_sha256"]
                == payload["native_execution_statistics"]["control_journal_sha256"]
            )
        else:
            assert descriptor["native_control_events"] is None
        assert payload["artifact_bytes"] == sum(
            selected.stat().st_size
            for selected in (
                axis_path,
                axis_sidecar,
                persistence_receipt,
                persistence_receipt_sidecar,
                *journal_bundle.iterdir(),
                *task_receipt_files,
            )
        )
        assert payload["artifact_bytes"] < 8 * 1024 * 1024
        replayed = _replay_record(
            ReviewRecord(axis_path, payload),
            root / "data" / "schneider",
        )
        assert replayed["valid"] is True, replayed
    assert all(not payload.get("error") for payload in payloads)
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

    with NativeHostScheduler(endpoint) as scheduler:
        written = _run_group(replace(task, scheduler_process_id=scheduler.process_id))

    payloads = [
        _axis_payload_with_persistence(Path(path), json.loads(Path(path).read_bytes()))
        for path in written
    ]
    assert len(payloads) == len(MODES)
    assert {payload["mode"] for payload in payloads} == {mode.value for mode in MODES}
    status_by_mode = {payload["mode"]: payload["status"] for payload in payloads}
    assert status_by_mode == {
        "current_stage052": "completed",
        "python_candidate_control": "completed",
        "per_solve_runtime": "completed",
        "full_native_alns": "completed",
        "host_scheduler": "completed",
    }
    for raw_path, payload in zip(written, payloads, strict=True):
        axis_path = Path(raw_path)
        if payload["status"] != "completed":
            assert "canonical_semantic_journal" not in payload
            continue
        assert "canonical_semantic_events" not in payload
        assert "canonical_semantic_streams" not in payload
        descriptor = payload["canonical_semantic_journal"]
        journal_bundle = axis_path.parent / descriptor["path"]
        journal_path = journal_bundle / descriptor["event_path"]
        journal_sidecar = journal_bundle / descriptor["sidecar_path"]
        axis_sidecar = axis_path.with_suffix(axis_path.suffix + ".sha256")
        persistence_receipt = axis_path.with_suffix(axis_path.suffix + ".persistence")
        persistence_receipt_sidecar = persistence_receipt.with_suffix(
            persistence_receipt.suffix + ".sha256"
        )
        task_receipt = axis_path.with_suffix(axis_path.suffix + ".native-work-tasks.jsonl")
        task_receipt_files = (
            (
                task_receipt,
                task_receipt.with_suffix(task_receipt.suffix + ".sha256"),
            )
            if task_receipt.is_file()
            else ()
        )
        assert journal_bundle.is_dir()
        assert journal_path.is_file() and journal_sidecar.is_file()
        assert descriptor["physical_telemetry"]["schema_version"].startswith(
            "stage05.2-physical-telemetry-"
        )
        assert payload["artifact_bytes"] == sum(
            selected.stat().st_size
            for selected in (
                axis_path,
                axis_sidecar,
                persistence_receipt,
                persistence_receipt_sidecar,
                *journal_bundle.iterdir(),
                *task_receipt_files,
            )
        )
        assert payload["artifact_bytes"] < 8 * 1024 * 1024
