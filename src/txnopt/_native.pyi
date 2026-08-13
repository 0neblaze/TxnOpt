from typing import Final, TypedDict

import numpy as np
import numpy.typing as npt

PROTOCOL_VERSION: Final[str]

class NativeBuildAttestation(TypedDict):
    schema_version: str
    source_revision: str
    source_tree: str
    source_manifest_sha256: str
    tracked_file_count: int
    source_dirty: bool
    development_override: bool
    cpp_source_kind: str
    performance_profile: str
    compiler_id: str
    compiler_version: str

BUILD_ATTESTATION: Final[NativeBuildAttestation]

class NativeRoundReceipt(TypedDict):
    protocol: str
    phase: str
    worker_count: int
    scheduled_worker_count: int
    execution_policy: str
    parallel_route_threshold: int
    context_pack_count: int
    round_call_count: int
    started_work: int
    screened_work: int
    completed_work: int
    interrupted_work: int
    budget_limit: int
    budget_reserved_work: int
    budget_remaining_work: int
    prepared_cache_write_count: int
    prepared_cache_key_checksum: int
    phase_trace: tuple[str, ...]
    semantic_event_count: int
    fallback_count: int
    source_revision: str
    source_tree: str

class EVRPTWContext:
    def __init__(
        self,
        node_kind: npt.NDArray[np.int64],
        demand: npt.NDArray[np.float64],
        ready_time: npt.NDArray[np.float64],
        due_date: npt.NDArray[np.float64],
        service_time: npt.NDArray[np.float64],
        distance: npt.NDArray[np.float64],
        reachable: npt.NDArray[np.uint8],
        vehicle: npt.NDArray[np.float64],
        worker_count: int,
    ) -> None: ...
    @property
    def worker_count(self) -> int: ...
    def exact_round_v1(
        self,
        route_offsets: npt.NDArray[np.int64],
        route_indices: npt.NDArray[np.int64],
        deadline_seconds: float,
        batch_size: int = 64,
        work_budget: int = -1,
    ) -> tuple[
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.float64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        NativeRoundReceipt,
    ]: ...
