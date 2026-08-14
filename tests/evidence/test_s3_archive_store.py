from __future__ import annotations

import base64
import hashlib
import io
from datetime import UTC, datetime, timedelta
from typing import Any, BinaryIO, cast

import pytest

from txnopt_evidence.archive import (
    ArchiveCommit,
    ArchiveConflictError,
    ArchiveEntry,
    ArchiveIntegrityError,
    ArchiveObjectRef,
    ArchiveStore,
)
from txnopt_evidence.archive_s3 import S3ArchiveStore


class _ServiceError(RuntimeError):
    def __init__(self, code: str, status: int) -> None:
        super().__init__(code)
        self.response = {
            "Error": {"Code": code},
            "ResponseMetadata": {"HTTPStatusCode": status},
        }


class _Body(io.BytesIO):
    pass


class _FakeS3Client:
    def __init__(self) -> None:
        self.versioning = "Enabled"
        self.object_lock_enabled = "Enabled"
        self.retention_mode = "COMPLIANCE"
        self.retention_days = 365
        self.objects: dict[str, dict[str, object]] = {}
        self.multipart: dict[str, dict[str, object]] = {}
        self.completed_multipart_count = 0
        self.aborted_multipart_count = 0

    def head_bucket(self, **_kwargs: object) -> dict[str, Any]:
        return {}

    def get_bucket_versioning(self, **_kwargs: object) -> dict[str, Any]:
        return {"Status": self.versioning}

    def get_object_lock_configuration(self, **_kwargs: object) -> dict[str, Any]:
        return {
            "ObjectLockConfiguration": {
                "ObjectLockEnabled": self.object_lock_enabled,
                "Rule": {
                    "DefaultRetention": {
                        "Mode": self.retention_mode,
                        "Days": self.retention_days,
                    }
                },
            }
        }

    def head_object(self, **kwargs: object) -> dict[str, Any]:
        key = cast(str, kwargs["Key"])
        try:
            item = self.objects[key]
        except KeyError as error:
            raise _ServiceError("NoSuchKey", 404) from error
        return {
            "ContentLength": len(cast(bytes, item["Body"])),
            "Metadata": dict(cast(dict[str, str], item["Metadata"])),
        }

    def get_object(self, **kwargs: object) -> dict[str, Any]:
        key = cast(str, kwargs["Key"])
        try:
            item = self.objects[key]
        except KeyError as error:
            raise _ServiceError("NoSuchKey", 404) from error
        return {"Body": _Body(cast(bytes, item["Body"]))}

    def get_object_retention(self, **kwargs: object) -> dict[str, Any]:
        key = cast(str, kwargs["Key"])
        try:
            item = self.objects[key]
        except KeyError as error:
            raise _ServiceError("NoSuchKey", 404) from error
        return {
            "Retention": {
                "Mode": item["ObjectLockMode"],
                "RetainUntilDate": item["ObjectLockRetainUntilDate"],
            }
        }

    def put_object(self, **kwargs: object) -> dict[str, Any]:
        key = cast(str, kwargs["Key"])
        if kwargs.get("IfNoneMatch") == "*" and key in self.objects:
            raise _ServiceError("PreconditionFailed", 412)
        source = cast(BinaryIO, kwargs["Body"])
        data = source.read()
        assert kwargs["ContentLength"] == len(data)
        checksum = kwargs.get("ChecksumSHA256")
        if checksum is not None:
            assert checksum == base64.b64encode(hashlib.sha256(data).digest()).decode()
        self.objects[key] = {
            "Body": data,
            "Metadata": dict(cast(dict[str, str], kwargs["Metadata"])),
            "ObjectLockMode": kwargs["ObjectLockMode"],
            "ObjectLockRetainUntilDate": kwargs["ObjectLockRetainUntilDate"],
        }
        return {"ETag": hashlib.md5(data, usedforsecurity=False).hexdigest()}

    def create_multipart_upload(self, **kwargs: object) -> dict[str, Any]:
        upload_id = f"upload-{len(self.multipart) + 1}"
        self.multipart[upload_id] = {
            "Key": kwargs["Key"],
            "Metadata": dict(cast(dict[str, str], kwargs["Metadata"])),
            "ObjectLockMode": kwargs["ObjectLockMode"],
            "ObjectLockRetainUntilDate": kwargs["ObjectLockRetainUntilDate"],
            "Parts": {},
        }
        return {"UploadId": upload_id}

    def upload_part(self, **kwargs: object) -> dict[str, Any]:
        upload_id = cast(str, kwargs["UploadId"])
        part_number = cast(int, kwargs["PartNumber"])
        data = cast(bytes, kwargs["Body"])
        cast(dict[int, bytes], self.multipart[upload_id]["Parts"])[part_number] = data
        return {"ETag": hashlib.md5(data, usedforsecurity=False).hexdigest()}

    def complete_multipart_upload(self, **kwargs: object) -> dict[str, Any]:
        upload_id = cast(str, kwargs["UploadId"])
        state = self.multipart[upload_id]
        key = cast(str, state["Key"])
        if kwargs.get("IfNoneMatch") == "*" and key in self.objects:
            raise _ServiceError("PreconditionFailed", 412)
        parts = cast(dict[int, bytes], state["Parts"])
        data = b"".join(parts[number] for number in sorted(parts))
        self.objects[key] = {
            "Body": data,
            "Metadata": state["Metadata"],
            "ObjectLockMode": state["ObjectLockMode"],
            "ObjectLockRetainUntilDate": state["ObjectLockRetainUntilDate"],
        }
        del self.multipart[upload_id]
        self.completed_multipart_count += 1
        return {"ETag": "multipart"}

    def abort_multipart_upload(self, **kwargs: object) -> dict[str, Any]:
        upload_id = cast(str, kwargs["UploadId"])
        self.multipart.pop(upload_id, None)
        self.aborted_multipart_count += 1
        return {}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _store(
    client: _FakeS3Client | None = None,
    **kwargs: object,
) -> tuple[_FakeS3Client, S3ArchiveStore]:
    selected = client or _FakeS3Client()
    return selected, S3ArchiveStore(
        selected,
        bucket="txnopt-evidence",
        prefix="archive-v1",
        **kwargs,
    )


def test_s3_archive_requires_versioning_and_compliance_retention() -> None:
    client = _FakeS3Client()
    client.retention_mode = "GOVERNANCE"

    with pytest.raises(ArchiveIntegrityError, match="COMPLIANCE"):
        _store(client)

    client.retention_mode = "COMPLIANCE"
    client.retention_days = 30
    with pytest.raises(ArchiveIntegrityError, match="shorter"):
        _store(client)


def test_s3_archive_put_open_head_and_idempotent_blob() -> None:
    client, store = _store()
    payload = b"immutable cloud evidence\n"
    ref = ArchiveObjectRef(sha256=_sha256(payload), size=len(payload))

    assert isinstance(store, ArchiveStore)
    assert store.put_blob(
        io.BytesIO(payload),
        expected_sha256=ref.sha256,
        expected_size=ref.size,
    ) == ref
    assert store.put_blob(
        io.BytesIO(payload),
        expected_sha256=ref.sha256,
        expected_size=ref.size,
    ) == ref
    assert len(client.objects) == 1
    assert store.head(ref).verified is True
    with store.open(ref) as handle:
        assert handle.read() == payload


def test_s3_archive_rejects_wrong_source_before_remote_publication() -> None:
    client, store = _store()

    with pytest.raises(ArchiveIntegrityError, match="digest or size"):
        store.put_blob(
            io.BytesIO(b"wrong"),
            expected_sha256="0" * 64,
            expected_size=5,
        )

    assert client.objects == {}


def test_s3_archive_detects_remote_tampering_on_every_read() -> None:
    client, store = _store()
    payload = b"verified first"
    ref = store.put_blob(
        io.BytesIO(payload),
        expected_sha256=_sha256(payload),
        expected_size=len(payload),
    )
    only_key = next(iter(client.objects))
    client.objects[only_key]["Body"] = b"tampered bytes"

    with pytest.raises(ArchiveIntegrityError):
        store.open(ref)
    with pytest.raises(ArchiveIntegrityError):
        store.head(ref)


def test_s3_archive_publishes_commit_last_and_verifies_every_blob() -> None:
    client, store = _store()
    payload = b"payload"
    blob_ref = store.put_blob(
        io.BytesIO(payload),
        expected_sha256=_sha256(payload),
        expected_size=len(payload),
    )
    commit = ArchiveCommit(
        commit_id="cloud-attempt-01",
        entries=(ArchiveEntry(relative_path="raw/payload.bin", ref=blob_ref),),
    )

    commit_ref = store.publish_commit(commit)
    receipt = store.verify_commit(commit_ref)

    assert receipt.verified is True
    assert receipt.commit == commit
    assert receipt.object_count == 1
    assert receipt.total_size == len(payload)
    assert any(key.endswith("commits/cloud-attempt-01.json") for key in client.objects)

    with pytest.raises(ArchiveConflictError):
        store.publish_commit(commit)
    assert store.publish_commit(commit, expected_absent=False) == commit_ref


def test_s3_archive_uses_conditional_multipart_completion() -> None:
    part_size = 5 * 1024 * 1024
    client, store = _store(
        multipart_threshold=part_size,
        multipart_part_size=part_size,
    )
    payload = b"a" * part_size + b"tail"

    ref = store.put_blob(
        io.BytesIO(payload),
        expected_sha256=_sha256(payload),
        expected_size=len(payload),
    )

    assert client.completed_multipart_count == 1
    assert client.multipart == {}
    with store.open(ref) as handle:
        assert handle.read() == payload


def test_s3_archive_rejects_changed_immutable_metadata() -> None:
    client, store = _store()
    payload = b"metadata-bound"
    ref = store.put_blob(
        io.BytesIO(payload),
        expected_sha256=_sha256(payload),
        expected_size=len(payload),
    )
    only_key = next(iter(client.objects))
    metadata = cast(dict[str, str], client.objects[only_key]["Metadata"])
    metadata["txnopt-size"] = "999"

    with pytest.raises(ArchiveIntegrityError, match="metadata differs"):
        store.open(ref)


def test_s3_archive_rejects_missing_or_expired_object_retention() -> None:
    client, store = _store()
    payload = b"retention-bound"
    ref = store.put_blob(
        io.BytesIO(payload),
        expected_sha256=_sha256(payload),
        expected_size=len(payload),
    )
    only_key = next(iter(client.objects))
    client.objects[only_key]["ObjectLockRetainUntilDate"] = datetime.now(UTC) - timedelta(
        seconds=1
    )

    with pytest.raises(ArchiveIntegrityError, match="expired"):
        store.open(ref)


def test_s3_archive_rejects_cold_storage_before_restore_verification() -> None:
    with pytest.raises(ValueError, match="immediately readable"):
        _store(storage_class="DEEP_ARCHIVE")
