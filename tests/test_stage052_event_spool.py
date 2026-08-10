from __future__ import annotations

import math
import threading
from contextlib import suppress
from hashlib import sha256
from typing import cast

import pytest

from evrptw.stage052_event_spool import (
    Stage052EventSpool,
    decode_stage052_attempt_payload,
    iter_stage052_attempt_audit,
)
from evrptw.stage052_writer_pipeline import validate_pipeline_receipt


def test_event_spool_replays_indexes_truncates_and_cleans_up() -> None:
    spool = Stage052EventSpool(prefix="stage052-test-")
    path = spool.path
    try:
        spool.append(("operator", {"event_type": "first", "value": 1}))
        spool.extend(
            (
                ("screening", {"event_type": "second", "value": 2}),
                ("termination", {"event_type": "termination", "value": 3}),
            )
        )

        assert len(spool) == 3
        assert spool[0] == ("operator", {"event_type": "first", "value": 1})
        assert spool[-1][0] == "termination"
        assert [stream for stream, _ in spool] == [
            "operator",
            "screening",
            "termination",
        ]
        assert len(spool[1:]) == 2

        spool.truncate(1)

        assert list(spool) == [("operator", {"event_type": "first", "value": 1})]
    finally:
        spool.close()
    assert not path.exists()


def test_event_spool_round_trips_nonfinite_scalars_through_strict_json() -> None:
    with Stage052EventSpool(prefix="stage052-test-nonfinite-") as spool:
        spool.append(
            (
                "operator",
                {
                    "positive": float("inf"),
                    "negative": float("-inf"),
                    "missing": float("nan"),
                },
            )
        )

        _stream, event = spool[0]

        assert event["positive"] == float("inf")
        assert event["negative"] == float("-inf")
        assert isinstance(event["missing"], float)
        assert math.isnan(event["missing"])


def test_event_spool_compresses_and_rejects_corrupt_frames() -> None:
    with Stage052EventSpool(prefix="stage052-test-compressed-") as spool:
        spool.append(("screening", {"repeated": "x" * 100_000}))
        assert spool[0][1]["repeated"] == "x" * 100_000
        assert spool.path.stat().st_size < 2_000
        raw = bytearray(spool.path.read_bytes())
        raw[-1] ^= 1
        spool.path.write_bytes(raw)

        with pytest.raises(RuntimeError, match="frame is corrupt"):
            list(spool)


@pytest.mark.parametrize("checkpoint", [256, 300])
def test_event_spool_truncates_across_block_boundaries_and_appends(
    checkpoint: int,
) -> None:
    with Stage052EventSpool(prefix="stage052-test-blocks-") as spool:
        spool.extend(("screening", {"ordinal": ordinal}) for ordinal in range(600))

        spool.truncate(checkpoint)
        spool.extend(
            ("operator", {"ordinal": ordinal}) for ordinal in range(checkpoint, checkpoint + 20)
        )

        rows = list(spool)
        assert len(rows) == checkpoint + 20
        assert [cast(int, row[1]["ordinal"]) for row in rows] == list(range(checkpoint + 20))
        assert all(stream == "screening" for stream, _event in rows[:checkpoint])
        assert all(stream == "operator" for stream, _event in rows[checkpoint:])


def test_event_spool_pipeline_is_bounded_non_daemon_and_replayable() -> None:
    with Stage052EventSpool(prefix="stage052-test-pipeline-") as spool:
        spool.extend(("screening", {"ordinal": ordinal}) for ordinal in range(600))

        receipt = spool.seal()
        ledger = validate_pipeline_receipt(receipt, require_finalized=True)

        assert receipt["writer_thread_count"] == 1
        assert receipt["writer_daemon"] is False
        assert receipt["queue_bound_batches"] == 1
        assert receipt["peak_queued_batches"] == 1
        assert receipt["submitted_batches"] == 3
        assert receipt["completed_batches"] == 3
        assert [row["row_count"] for row in ledger] == [256, 256, 88]
        assert len(list(spool)) == 600


def test_event_spool_rollback_rewinds_terminal_batch_ledger() -> None:
    with Stage052EventSpool(prefix="stage052-test-rewind-") as spool:
        spool.extend(("screening", {"ordinal": ordinal}) for ordinal in range(600))
        spool.truncate(300)
        spool.extend(("operator", {"ordinal": ordinal}) for ordinal in range(300, 700))

        receipt = spool.seal()
        ledger = validate_pipeline_receipt(receipt, require_finalized=True)

        assert receipt["attempted_batches"] == 5
        assert receipt["discarded_batches"] == 2
        assert receipt["discarded_rows"] == 344
        assert receipt["rewind_operations"] == 1
        assert [row["batch_ordinal"] for row in ledger] == [0, 1, 2]
        assert [row["row_count"] for row in ledger] == [256, 256, 188]
        assert [event[1]["ordinal"] for event in spool] == list(range(700))


def test_event_spool_detached_audit_replays_retained_and_discarded_attempts() -> None:
    spool = Stage052EventSpool(prefix="stage052-test-audit-")
    audit_path = None
    try:
        spool.extend(("screening", {"ordinal": ordinal}) for ordinal in range(600))
        spool.truncate(300)
        spool.extend(("operator", {"ordinal": ordinal}) for ordinal in range(300, 700))

        receipt, audit_path = spool.detach_pipeline_evidence()
        attempted = cast(list[dict[str, object]], receipt["attempted_batch_ledger"])
        frames = list(iter_stage052_attempt_audit(audit_path))

        assert len(frames) == len(attempted) == 5
        assert [row["status"] for row in attempted] == [
            "retained",
            "discarded",
            "discarded",
            "retained",
            "retained",
        ]
        for frame, row in zip(frames, attempted, strict=True):
            decoded = decode_stage052_attempt_payload(
                frame.decoded_payload,
                row_count=frame.row_count,
            )
            assert len(decoded) == row["row_count"]
            assert frame.attempt_ordinal == row["attempt_ordinal"]
            assert frame.batch_ordinal == row["batch_ordinal"]
            assert frame.payload_bytes == row["payload_bytes"]
            assert sha256(frame.decoded_payload).hexdigest() == row["sha256"]
    finally:
        spool.close()
    assert audit_path is not None and audit_path.is_file()
    audit_path.unlink()


def test_event_spool_writer_failure_aborts_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_write(
        _self: Stage052EventSpool,
        _ordinal: int,
        _row_count: int,
        _payload: object,
    ) -> object:
        raise OSError("injected event spool write failure")

    monkeypatch.setattr(Stage052EventSpool, "_write_batch", fail_write)
    spool = Stage052EventSpool(prefix="stage052-test-failure-")
    path = spool.path
    try:
        spool.extend(("screening", {"ordinal": ordinal}) for ordinal in range(256))
        with pytest.raises(OSError, match="injected event spool write failure"):
            spool.pipeline_statistics()
    finally:
        with suppress(OSError):
            spool.close()

    assert not path.exists()
    assert not any(
        thread.name == "s52-event-write" and thread.is_alive() for thread in threading.enumerate()
    )
