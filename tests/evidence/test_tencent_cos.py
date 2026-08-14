from __future__ import annotations

import hashlib
import io
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, BinaryIO, cast

import pytest

from txnopt_evidence.archive import ArchiveIntegrityError
from txnopt_evidence.tencent_cos import (
    TencentCosArchive,
    TencentCosContractError,
    TencentCosError,
    _QcloudCosSdkBridge,
)


class _FakeTencentCosClient:
    def __init__(self) -> None:
        self.versioning = "Enabled"
        self.lock_enabled = "Enabled"
        self.retention_mode = "COMPLIANCE"
        self.retention_days = 365
        self.next_version = 1
        self.return_version_id = True
        self.objects: dict[tuple[str, str], dict[str, object]] = {}
        self.latest: dict[str, str] = {}
        self.multipart: dict[str, dict[str, object]] = {}
        self.completed_multipart = 0
        self.aborted_multipart = 0

    def head_bucket(self, **_kwargs: object) -> dict[str, Any]:
        return {}

    def get_bucket_versioning(self, **_kwargs: object) -> dict[str, Any]:
        return {"Status": self.versioning}

    def get_bucket_object_lock(self, **_kwargs: object) -> dict[str, Any]:
        return {
            "ObjectLockEnabled": self.lock_enabled,
            "Rule": {
                "DefaultRetention": {
                    "Mode": self.retention_mode,
                    "Days": self.retention_days,
                }
            },
        }

    def put_object(self, **kwargs: object) -> dict[str, Any]:
        key = cast(str, kwargs["Key"])
        if kwargs.get("IfNoneMatch") == "*" and key in self.latest:
            raise ArchiveIntegrityError("conditional Tencent COS put conflicted")
        source = cast(BinaryIO, kwargs["Body"])
        data = source.read()
        assert len(data) == kwargs["ContentLength"]
        return self._publish(
            key,
            data,
            cast(dict[str, str], kwargs["Metadata"]),
        )

    def create_multipart_upload(self, **kwargs: object) -> dict[str, Any]:
        upload_id = f"upload-{len(self.multipart) + 1}"
        self.multipart[upload_id] = {
            "Key": kwargs["Key"],
            "Metadata": kwargs["Metadata"],
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
        state = self.multipart.pop(upload_id)
        parts = cast(dict[int, bytes], state["Parts"])
        data = b"".join(parts[number] for number in sorted(parts))
        self.completed_multipart += 1
        return self._publish(
            cast(str, state["Key"]),
            data,
            cast(dict[str, str], state["Metadata"]),
        )

    def abort_multipart_upload(self, **kwargs: object) -> None:
        upload_id = cast(str, kwargs["UploadId"])
        self.multipart.pop(upload_id, None)
        self.aborted_multipart += 1

    def head_object(self, **kwargs: object) -> dict[str, Any]:
        item, version_id = self._get(kwargs)
        metadata = cast(dict[str, str], item["Metadata"])
        return {
            "Content-Length": str(len(cast(bytes, item["Body"]))),
            "x-cos-version-id": version_id,
            "x-cos-meta-txnopt-sha256": metadata["txnopt-sha256"],
            "x-cos-meta-txnopt-size": metadata["txnopt-size"],
            "x-cos-storage-class": "STANDARD",
            "x-cos-server-side-encryption": "AES256",
        }

    def get_object(self, **kwargs: object) -> dict[str, Any]:
        item, version_id = self._get(kwargs)
        return {
            "Body": io.BytesIO(cast(bytes, item["Body"])),
            "x-cos-version-id": version_id,
        }

    def get_object_retention(self, **kwargs: object) -> dict[str, Any]:
        item, _version_id = self._get(kwargs)
        return {
            "Retention": {
                "Mode": item["RetentionMode"],
                "RetainUntilDate": item["RetainUntilDate"],
            }
        }

    def object_exists(self, **kwargs: object) -> bool:
        return cast(str, kwargs["Key"]) in self.latest

    def _publish(
        self,
        key: str,
        data: bytes,
        metadata: dict[str, str],
    ) -> dict[str, Any]:
        version_id = f"version-{self.next_version}"
        self.next_version += 1
        self.objects[(key, version_id)] = {
            "Body": data,
            "Metadata": dict(metadata),
            "RetentionMode": "COMPLIANCE",
            "RetainUntilDate": (
                datetime.now(UTC) + timedelta(days=self.retention_days)
            ).isoformat().replace("+00:00", "Z"),
        }
        self.latest[key] = version_id
        return {"x-cos-version-id": version_id} if self.return_version_id else {}

    def _get(self, kwargs: dict[str, object]) -> tuple[dict[str, object], str]:
        key = cast(str, kwargs["Key"])
        version_id = cast(str | None, kwargs.get("VersionId")) or self.latest[key]
        return self.objects[(key, version_id)], version_id


class _ExplodingSdkError(RuntimeError):
    """Synthetic SDK failure carrying every credential form."""


class _ExplodingSdkClient:
    def __init__(self, message: str) -> None:
        self._message = message

    def head_bucket(self, **_kwargs: object) -> dict[str, Any]:
        raise _ExplodingSdkError(self._message)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _archive(
    client: _FakeTencentCosClient | None = None,
    **kwargs: object,
) -> tuple[_FakeTencentCosClient, TencentCosArchive]:
    selected = client or _FakeTencentCosClient()
    return selected, TencentCosArchive(
        selected,
        bucket="txnopt-evidence-1250000000",
        region="ap-guangzhou",
        prefix="formal",
        **kwargs,
    )


def test_tencent_cos_requires_versioning_and_compliance_object_lock() -> None:
    client = _FakeTencentCosClient()
    client.versioning = "Suspended"
    with pytest.raises(TencentCosContractError, match="versioning"):
        _archive(client)

    client.versioning = "Enabled"
    client.retention_mode = "GOVERNANCE"
    with pytest.raises(TencentCosContractError, match="COMPLIANCE"):
        _archive(client)

    client.retention_mode = "COMPLIANCE"
    client.retention_days = 30
    with pytest.raises(TencentCosContractError, match="shorter"):
        _archive(client)


def test_tencent_cos_sdk_errors_redact_environment_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret_id = "AKID-cos-test-secret-id"
    secret_key = "cos-test-secret-key-value"
    session_token = "cos-test-session-token-value"
    monkeypatch.setenv("TENCENTCLOUD_SECRET_ID", secret_id)
    monkeypatch.setenv("TENCENTCLOUD_SECRET_KEY", secret_key)
    monkeypatch.setenv("TENCENTCLOUD_SESSION_TOKEN", session_token)
    sdk_message = (
        "request rejected for "
        f"{secret_id}, {secret_key}, and {session_token}"
    )
    client = _ExplodingSdkClient(sdk_message)
    bridge = _QcloudCosSdkBridge(client)
    monkeypatch.delenv("TENCENTCLOUD_SECRET_ID")
    monkeypatch.delenv("TENCENTCLOUD_SECRET_KEY")
    monkeypatch.delenv("TENCENTCLOUD_SESSION_TOKEN")

    with pytest.raises(TencentCosError) as raised:
        TencentCosArchive(
            bridge,
            bucket="txnopt-evidence-1250000000",
            region="ap-guangzhou",
        )

    message = str(raised.value)
    assert "_ExplodingSdkError" in message
    assert secret_id not in message
    assert secret_key not in message
    assert session_token not in message


def test_tencent_cos_sdk_errors_drop_untrusted_cam_role_message() -> None:
    cam_role_secret = "cam-role-cos-temporary-secret-material"
    bridge = _QcloudCosSdkBridge(_ExplodingSdkClient(cam_role_secret))

    with pytest.raises(TencentCosError) as raised:
        TencentCosArchive(
            bridge,
            bucket="txnopt-evidence-1250000000",
            region="ap-guangzhou",
        )

    message = str(raised.value)
    assert "_ExplodingSdkError" in message
    assert cam_role_secret not in message


def test_tencent_cos_binds_every_read_to_exact_version_id() -> None:
    client, archive = _archive()
    first = b"first immutable version"
    first_ref = archive.put_blob(
        io.BytesIO(first), expected_sha256=_sha256(first), expected_size=len(first)
    )
    second_ref = archive.put_blob(
        io.BytesIO(first), expected_sha256=_sha256(first), expected_size=len(first)
    )

    assert first_ref.key == second_ref.key
    assert first_ref.version_id != second_ref.version_id
    assert archive.verify_object(first_ref).verified is True
    assert archive.verify_object(second_ref).verified is True
    assert all(version_id for _key, version_id in client.objects)


def test_tencent_cos_rejects_upload_without_returned_version_id() -> None:
    client = _FakeTencentCosClient()
    client.return_version_id = False
    _, archive = _archive(client)
    payload = b"must be version-bound"

    with pytest.raises(ArchiveIntegrityError, match="VersionId"):
        archive.put_blob(
            io.BytesIO(payload),
            expected_sha256=_sha256(payload),
            expected_size=len(payload),
        )


def test_tencent_cos_rejects_wrong_source_before_upload() -> None:
    client, archive = _archive()

    with pytest.raises(ArchiveIntegrityError, match="declared digest"):
        archive.put_blob(
            io.BytesIO(b"wrong"),
            expected_sha256="0" * 64,
            expected_size=5,
        )

    assert client.objects == {}


def test_tencent_cos_rejects_tampered_exact_version() -> None:
    client, archive = _archive()
    payload = b"verified bytes"
    ref = archive.put_blob(
        io.BytesIO(payload), expected_sha256=_sha256(payload), expected_size=len(payload)
    )
    client.objects[(ref.key, ref.version_id)]["Body"] = b"tampered"

    with pytest.raises(ArchiveIntegrityError):
        archive.verify_object(ref)


def test_tencent_cos_rejects_expired_retention() -> None:
    client, archive = _archive()
    payload = b"retained bytes"
    ref = archive.put_blob(
        io.BytesIO(payload), expected_sha256=_sha256(payload), expected_size=len(payload)
    )
    client.objects[(ref.key, ref.version_id)]["RetainUntilDate"] = (
        datetime.now(UTC) - timedelta(seconds=1)
    ).isoformat()

    with pytest.raises(ArchiveIntegrityError, match="expired"):
        archive.verify_object(ref)


def test_tencent_cos_multipart_commit_verify_and_restore(tmp_path: Path) -> None:
    if not sys.platform.startswith("linux"):
        pytest.skip("source inventory and atomic restore are Linux-only")
    client, archive = _archive(
        multipart_threshold=2 * 1024 * 1024,
        multipart_part_size=1024 * 1024,
    )
    source = tmp_path / "source"
    (source / "nested").mkdir(parents=True)
    payload = b"a" * (2 * 1024 * 1024) + b"tail"
    (source / "nested" / "large.bin").write_bytes(payload)
    (source / "small.json").write_text('{"status":"SEALED"}\n', encoding="utf-8")

    ref = archive.mirror_tree(source, commit_id="tencent-attempt-01")
    verification = archive.verify_commit(ref)
    destination = tmp_path / "restored"
    restored = archive.restore_commit(ref, destination=destination)

    assert verification.verified is True
    assert verification.object_count == 2
    assert restored.verified is True
    assert client.completed_multipart == 1
    assert _tree_bytes(destination) == _tree_bytes(source)
    assert "/commits/tencent-attempt-01/" in ref.key
    assert ref.version_id


def test_tencent_cos_restore_rejects_existing_destination(tmp_path: Path) -> None:
    if not sys.platform.startswith("linux"):
        pytest.skip("source inventory and atomic restore are Linux-only")
    _, archive = _archive()
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.txt").write_text("a\n", encoding="utf-8")
    ref = archive.mirror_tree(source, commit_id="restore-conflict")
    destination = tmp_path / "existing"
    destination.mkdir()

    with pytest.raises(ArchiveIntegrityError, match="already exists"):
        archive.restore_commit(ref, destination=destination)


def test_tencent_cos_rejects_duplicate_commit_marker(tmp_path: Path) -> None:
    if not sys.platform.startswith("linux"):
        pytest.skip("source inventory is Linux-only")
    _, archive = _archive()
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.txt").write_text("first\n", encoding="utf-8")

    first = archive.mirror_tree(source, commit_id="immutable-attempt")
    assert first.key.endswith("/commits/immutable-attempt/commit.json")

    (source / "a.txt").write_text("second\n", encoding="utf-8")
    with pytest.raises(ArchiveIntegrityError, match="cannot be overwritten"):
        archive.mirror_tree(source, commit_id="immutable-attempt")


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
