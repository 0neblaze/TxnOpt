from __future__ import annotations

import math

import numpy as np
import pytest

from txnopt import RunConfig, _native
from txnopt.runtime import PythonTxnRuntime, RuntimeContractError
from txnopt_cases.evrptw import (
    Instance,
    Node,
    NodeType,
    Vehicle,
    solve_exact_charging,
)
from txnopt_cases.evrptw.kernel import EVRPTWSearchKernel
from txnopt_cases.evrptw.native_oracle import NativeEVRPTWOracle
from txnopt_cases.evrptw.oracle import EVRPTWOracle, EVRPTWPlan


def _context(worker_count: int) -> _native.EVRPTWContext:
    return _native.EVRPTWContext(
        np.asarray((0, 1, 1), dtype=np.int64),
        np.asarray((0.0, 1.0, 1.0), dtype=np.float64),
        np.asarray((0.0, 0.0, 0.0), dtype=np.float64),
        np.asarray((100.0, 100.0, 100.0), dtype=np.float64),
        np.asarray((0.0, 0.0, 0.0), dtype=np.float64),
        np.asarray(
            (
                (0.0, 1.0, 2.0),
                (1.0, 0.0, 1.0),
                (2.0, 1.0, 0.0),
            ),
            dtype=np.float64,
        ),
        np.ones((3, 3), dtype=np.uint8),
        np.asarray((100.0, 10.0, 1.0, 0.1, 1.0), dtype=np.float64),
        worker_count,
    )


def _python_instance() -> Instance:
    return Instance(
        "native-round-tiny",
        (
            Node("D", NodeType.DEPOT, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0),
            Node("C1", NodeType.CUSTOMER, 1.0, 0.0, 1.0, 0.0, 100.0, 0.0),
            Node("C2", NodeType.CUSTOMER, 2.0, 0.0, 1.0, 0.0, 100.0, 0.0),
        ),
        Vehicle(100.0, 10.0, 1.0, 0.1, 1.0),
        distance_backend="python",
    )


def _round(context: _native.EVRPTWContext):
    return context.exact_round_v1(
        np.asarray((0, 1, 2), dtype=np.int64),
        np.asarray((1, 2), dtype=np.int64),
        math.inf,
        64,
    )


def test_native_round_is_one_typed_call_and_matches_python_exact_charging() -> None:
    context = _context(1)
    output = _round(context)
    python_results = tuple(
        solve_exact_charging(_python_instance(), route)
        for route in (("C1",), ("C2",))
    )

    path_offsets, path_indices, statuses, reasons, metrics = output[:5]
    receipt = output[9]
    assert _native.PROTOCOL_VERSION == "txnopt-native-round-v1"
    assert path_offsets.tolist() == [0, 3, 6]
    assert path_indices.tolist() == [0, 1, 0, 0, 2, 0]
    assert statuses.tolist() == [0, 0]
    assert reasons.tolist() == [0, 0]
    assert metrics[:, 0].tolist() == [item.distance for item in python_results]
    assert metrics[:, 3].tolist() == [item.charging_time for item in python_results]
    assert receipt["protocol"] == "txnopt-native-round-v1"
    assert receipt["phase"] == "VALIDATED"
    assert receipt["context_pack_count"] == 1
    assert receipt["screened_work"] == 2
    assert receipt["round_call_count"] == 1
    assert receipt["fallback_count"] == 0
    assert _round(context)[9]["round_call_count"] == 2


def test_native_serial_and_parallel_rounds_have_equal_semantic_arrays() -> None:
    serial = _round(_context(1))
    parallel = _round(_context(2))

    for index in range(8):
        np.testing.assert_array_equal(serial[index], parallel[index])
    assert serial[9]["worker_count"] == 1
    assert parallel[9]["worker_count"] == 2
    assert parallel[8].shape == (2, 7)


def test_native_round_rejects_dtype_coercion_and_invalid_customer_indices() -> None:
    context = _context(1)
    with pytest.raises(ValueError, match="dtype"):
        context.exact_round_v1(
            np.asarray((0, 1), dtype=np.int32),
            np.asarray((1,), dtype=np.int64),
            math.inf,
        )
    with pytest.raises(ValueError, match="customer"):
        context.exact_round_v1(
            np.asarray((0, 1), dtype=np.int64),
            np.asarray((0,), dtype=np.int64),
            math.inf,
        )


def test_native_oracle_crosses_one_round_seam_and_matches_python_semantics() -> None:
    instance = _python_instance()
    plan = EVRPTWPlan((("C1",), ("C2",)))
    initial = EVRPTWOracle(instance).solve_initial(plan)
    runtime = PythonTxnRuntime()
    native_serial = NativeEVRPTWOracle(instance, worker_count=1)
    serial = runtime.run(
        initial,
        kernel=EVRPTWSearchKernel(),
        oracle=native_serial,
        config=RunConfig(
            seed=2014,
            workers=1,
            execution_mode="serial",
            fixed_work=64,
            max_rounds=1,
        ),
    )
    native_parallel = NativeEVRPTWOracle(instance, worker_count=2)
    parallel = runtime.run(
        initial,
        kernel=EVRPTWSearchKernel(),
        oracle=native_parallel,
        config=RunConfig(
            seed=2014,
            workers=2,
            execution_mode="barrier",
            fixed_work=64,
            max_rounds=1,
        ),
    )
    python = runtime.run(
        initial,
        kernel=EVRPTWSearchKernel(),
        oracle=EVRPTWOracle(instance),
        config=RunConfig(
            seed=2014,
            workers=1,
            execution_mode="serial",
            fixed_work=64,
            max_rounds=1,
        ),
    )

    assert serial.objective.vehicle_count == parallel.objective.vehicle_count == 1
    assert serial.objective == parallel.objective == python.objective
    assert serial.semantic_digest == parallel.semantic_digest == python.semantic_digest
    assert native_serial.last_receipt is not None
    assert native_serial.last_receipt["round_call_count"] == 1
    assert native_parallel.last_receipt is not None
    assert native_parallel.last_receipt["round_call_count"] == 1


def test_native_oracle_worker_topology_must_match_run_config() -> None:
    instance = _python_instance()
    oracle = NativeEVRPTWOracle(instance, worker_count=2)
    initial = EVRPTWOracle(instance).solve_initial(EVRPTWPlan((("C1",), ("C2",))))

    with pytest.raises(RuntimeContractError, match="worker count"):
        PythonTxnRuntime().run(
            initial,
            kernel=EVRPTWSearchKernel(),
            oracle=oracle,
            config=RunConfig(
                seed=2014,
                workers=4,
                execution_mode="barrier",
                fixed_work=64,
                max_rounds=1,
            ),
        )
