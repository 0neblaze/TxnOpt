"""Tencent COS evidence archive with exact version-bound object identities.

This is a concrete Tencent Cloud module, not a generic storage port.  COS
versioning permits several byte sequences under one key, so every object and
commit reference includes the exact ``VersionId`` returned by COS.  Bucket
COMPLIANCE Object Lock protects each referenced version from mutation.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import importlib
import importlib.metadata
import io
import json
import os
import re
import shutil
import stat
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Protocol, TypeVar, cast

from txnopt_evidence._safe_sdk_error import safe_sdk_error_text
from txnopt_evidence.archive import (
    ArchiveError,
    ArchiveIntegrityError,
    inventory_tree,
    open_inventory_entry,
)
from txnopt_evidence.codec import canonical_json_bytes

_COMMIT_SCHEMA = "txnopt-tencent-cos-commit-v1"
_COMMIT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_FIVE_GIB = 5 * 1024 * 1024 * 1024
_ONE_MIB = 1024 * 1024
_SdkResult = TypeVar("_SdkResult")


class TencentCosError(ArchiveError):
    """Base error for Tencent COS archive operations."""


class TencentCosContractError(TencentCosError):
    """Raised when a COS bucket does not satisfy the evidence contract."""


@dataclass(frozen=True, slots=True)
class TencentCosObjectRef:
    key: str
    version_id: str
    sha256: str
    size: int
    retain_until: str

    def __post_init__(self) -> None:
        _validate_key(self.key)
        _validate_version_id(self.version_id)
        _validate_sha256(self.sha256)
        if self.size < 0:
            raise ValueError("Tencent COS object size must be non-negative")
        if _parse_timestamp(self.retain_until) <= datetime.now(UTC):
            raise ValueError("Tencent COS retain-until must be in the future")

    def to_payload(self) -> dict[str, object]:
        return {
            "key": self.key,
            "version_id": self.version_id,
            "sha256": self.sha256,
            "size": self.size,
            "retain_until": self.retain_until,
        }


@dataclass(frozen=True, slots=True)
class TencentCosEntry:
    relative_path: str
    ref: TencentCosObjectRef

    def __post_init__(self) -> None:
        _validate_relative_path(self.relative_path)


@dataclass(frozen=True, slots=True)
class TencentCosCommit:
    commit_id: str
    source_tree_sha256: str
    entries: tuple[TencentCosEntry, ...]
    schema_version: str = _COMMIT_SCHEMA

    def __post_init__(self) -> None:
        _validate_commit_id(self.commit_id)
        _validate_sha256(self.source_tree_sha256)
        if self.schema_version != _COMMIT_SCHEMA:
            raise ValueError("Tencent COS commit schema differs")
        paths = tuple(entry.relative_path for entry in self.entries)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ValueError("Tencent COS commit entries must be uniquely sorted")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "commit_id": self.commit_id,
            "source_tree_sha256": self.source_tree_sha256,
            "objects": [
                {
                    "relative_path": entry.relative_path,
                    **entry.ref.to_payload(),
                }
                for entry in self.entries
            ],
        }


@dataclass(frozen=True, slots=True)
class TencentCosCommitRef:
    commit_id: str
    key: str
    version_id: str
    sha256: str
    size: int
    retain_until: str

    def __post_init__(self) -> None:
        _validate_commit_id(self.commit_id)
        _validate_key(self.key)
        _validate_version_id(self.version_id)
        _validate_sha256(self.sha256)
        if self.size < 0:
            raise ValueError("Tencent COS commit size must be non-negative")
        if _parse_timestamp(self.retain_until) <= datetime.now(UTC):
            raise ValueError("Tencent COS commit retain-until must be in the future")

    @property
    def object_ref(self) -> TencentCosObjectRef:
        return TencentCosObjectRef(
            key=self.key,
            version_id=self.version_id,
            sha256=self.sha256,
            size=self.size,
            retain_until=self.retain_until,
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "commit_id": self.commit_id,
            "key": self.key,
            "version_id": self.version_id,
            "sha256": self.sha256,
            "size": self.size,
            "retain_until": self.retain_until,
        }


@dataclass(frozen=True, slots=True)
class TencentCosBucketReceipt:
    bucket: str
    region: str
    versioning_enabled: bool
    object_lock_enabled: bool
    retention_mode: str
    default_retention_days: int
    minimum_retention_days: int
    exact_version_reads_required: bool = True


@dataclass(frozen=True, slots=True)
class TencentCosObjectVerificationReceipt:
    ref: TencentCosObjectRef
    verified: bool


@dataclass(frozen=True, slots=True)
class TencentCosVerificationReceipt:
    commit: TencentCosCommit
    ref: TencentCosCommitRef
    object_count: int
    total_size: int
    verified: bool


@dataclass(frozen=True, slots=True)
class TencentCosRestoreReceipt:
    commit_ref: TencentCosCommitRef
    object_count: int
    total_size: int
    verified: bool


class _TencentCosClient(Protocol):
    def head_bucket(self, **kwargs: object) -> Mapping[str, Any]: ...

    def get_bucket_versioning(self, **kwargs: object) -> Mapping[str, Any]: ...

    def get_bucket_object_lock(self, **kwargs: object) -> Mapping[str, Any]: ...

    def put_object(self, **kwargs: object) -> Mapping[str, Any]: ...

    def create_multipart_upload(self, **kwargs: object) -> Mapping[str, Any]: ...

    def upload_part(self, **kwargs: object) -> Mapping[str, Any]: ...

    def complete_multipart_upload(self, **kwargs: object) -> Mapping[str, Any]: ...

    def abort_multipart_upload(self, **kwargs: object) -> object: ...

    def head_object(self, **kwargs: object) -> Mapping[str, Any]: ...

    def get_object(self, **kwargs: object) -> Mapping[str, Any]: ...

    def get_object_retention(self, **kwargs: object) -> Mapping[str, Any]: ...

    def object_exists(self, **kwargs: object) -> bool: ...


class TencentCosArchive:
    """Immutable evidence archive bound directly to one Tencent COS bucket."""

    def __init__(
        self,
        client: _TencentCosClient,
        *,
        bucket: str,
        region: str,
        prefix: str = "txnopt",
        minimum_retention_days: int = 365,
        multipart_threshold: int = 64 * 1024 * 1024,
        multipart_part_size: int = 16 * 1024 * 1024,
    ) -> None:
        if not bucket or not region:
            raise ValueError("Tencent COS bucket and region are required")
        if minimum_retention_days <= 0:
            raise ValueError("Tencent COS retention must be positive")
        if not _ONE_MIB <= multipart_part_size <= _FIVE_GIB:
            raise ValueError("Tencent COS multipart part size is outside COS limits")
        if not multipart_part_size <= multipart_threshold <= _FIVE_GIB:
            raise ValueError("Tencent COS multipart threshold is invalid")
        self._client = client
        self._bucket = bucket
        self._region = region
        self._prefix = _validate_prefix(prefix)
        self._minimum_retention_days = minimum_retention_days
        self._multipart_threshold = multipart_threshold
        self._multipart_part_size = multipart_part_size
        self._bucket_receipt = self.verify_bucket_contract()

    @classmethod
    def from_environment(
        cls,
        *,
        bucket: str,
        region: str,
        prefix: str = "txnopt",
        minimum_retention_days: int = 365,
    ) -> TencentCosArchive:
        """Create the live SDK adapter without accepting secrets as arguments."""

        secret_id = os.environ.get("TENCENTCLOUD_SECRET_ID")
        secret_key = os.environ.get("TENCENTCLOUD_SECRET_KEY")
        token = os.environ.get("TENCENTCLOUD_SESSION_TOKEN")
        use_cvm_role = os.environ.get("TENCENTCLOUD_USE_CVM_ROLE") == "1"
        if bool(secret_id) != bool(secret_key):
            raise TencentCosContractError(
                "Tencent COS credential environment is incomplete"
            )
        if use_cvm_role and secret_id:
            raise TencentCosContractError(
                "Tencent COS credential source is ambiguous; choose environment or CVM CAM role"
            )
        if not secret_id and not use_cvm_role:
            raise TencentCosContractError(
                "Tencent COS credentials are absent; provide environment credentials "
                "or explicitly enable the CVM CAM role"
            )
        try:
            if importlib.metadata.version("cos-python-sdk-v5") != "1.9.44":
                raise TencentCosContractError(
                    "Tencent COS package version differs from the audited 1.9.44 bridge"
                )
            sdk = importlib.import_module("qcloud_cos")
            sdk_version = importlib.import_module("qcloud_cos.version")
        except (ImportError, importlib.metadata.PackageNotFoundError) as error:
            raise TencentCosContractError(
                "Tencent COS SDK is absent; install txnopt[tencent]"
            ) from error
        if cast(Any, sdk_version).__version__ != "5.1.9.44":
            raise TencentCosContractError(
                "Tencent COS SDK version differs from the audited 1.9.44 bridge"
            )
        dynamic_sdk = cast(Any, sdk)
        config_factory = dynamic_sdk.CosConfig
        client_factory = dynamic_sdk.CosS3Client
        config_arguments: dict[str, object] = {"Region": region, "Scheme": "https"}
        if use_cvm_role:
            try:
                if importlib.metadata.version("tencentcloud-sdk-python-common") != "3.1.156":
                    raise TencentCosContractError(
                        "Tencent common SDK version differs from the audited 3.1.156 bridge"
                    )
                credential_module = cast(
                    Any,
                    importlib.import_module("tencentcloud.common.credential"),
                )
            except (ImportError, importlib.metadata.PackageNotFoundError) as error:
                raise TencentCosContractError(
                    "Tencent common SDK is required for a CVM CAM role"
                ) from error
            credential_instance: object | None = None
            credential_error: TencentCosError | None = None
            try:
                credential_instance = credential_module.CVMRoleCredential()
            except Exception as error:
                credential_error = _redacted_sdk_error(
                    "CVM CAM role credential",
                    error,
                )
            if credential_error is not None:
                raise credential_error
            config_arguments["CredentialInstance"] = credential_instance
        else:
            config_arguments["SecretId"] = secret_id
            config_arguments["SecretKey"] = secret_key
            if token:
                config_arguments["Token"] = token
        client: Any = None
        client_error: TencentCosError | None = None
        try:
            config = config_factory(**config_arguments)
            client = client_factory(config)
        except Exception as error:
            client_error = _redacted_sdk_error("client initialization", error)
        if client_error is not None:
            raise client_error
        return cls(
            _QcloudCosSdkBridge(client),
            bucket=bucket,
            region=region,
            prefix=prefix,
            minimum_retention_days=minimum_retention_days,
        )

    @property
    def bucket_receipt(self) -> TencentCosBucketReceipt:
        return self._bucket_receipt

    def verify_bucket_contract(self) -> TencentCosBucketReceipt:
        self._client.head_bucket(Bucket=self._bucket)
        versioning = self._client.get_bucket_versioning(Bucket=self._bucket)
        if versioning.get("Status") != "Enabled":
            raise TencentCosContractError("Tencent COS bucket versioning is not enabled")
        lock = self._client.get_bucket_object_lock(Bucket=self._bucket)
        if lock.get("ObjectLockEnabled") != "Enabled":
            raise TencentCosContractError("Tencent COS Object Lock is not enabled")
        rule = _required_mapping(lock.get("Rule"), "Tencent COS Object Lock rule")
        retention = _required_mapping(
            rule.get("DefaultRetention"),
            "Tencent COS default retention",
        )
        mode = retention.get("Mode")
        if mode != "COMPLIANCE":
            raise TencentCosContractError(
                "Tencent COS default retention must use COMPLIANCE mode"
            )
        days = _retention_days(retention)
        if days < self._minimum_retention_days:
            raise TencentCosContractError(
                "Tencent COS default retention is shorter than the required minimum"
            )
        return TencentCosBucketReceipt(
            bucket=self._bucket,
            region=self._region,
            versioning_enabled=True,
            object_lock_enabled=True,
            retention_mode="COMPLIANCE",
            default_retention_days=days,
            minimum_retention_days=self._minimum_retention_days,
        )

    def put_blob(
        self,
        source: BinaryIO,
        *,
        expected_sha256: str,
        expected_size: int,
    ) -> TencentCosObjectRef:
        _validate_sha256(expected_sha256)
        if expected_size < 0:
            raise ValueError("Tencent COS object size must be non-negative")
        key = self._key("blobs", "sha256", expected_sha256[:2], expected_sha256)
        return self._upload_verified_source(
            source,
            key=key,
            expected_sha256=expected_sha256,
            expected_size=expected_size,
        )

    def verify_object(
        self,
        ref: TencentCosObjectRef,
    ) -> TencentCosObjectVerificationReceipt:
        self._download_verified(ref, collect=False)
        return TencentCosObjectVerificationReceipt(ref=ref, verified=True)

    def mirror_tree(self, source: Path, *, commit_id: str) -> TencentCosCommitRef:
        inventory = inventory_tree(source)
        entries: list[TencentCosEntry] = []
        for entry in inventory.entries:
            with open_inventory_entry(source, entry) as handle:
                ref = self.put_blob(
                    handle,
                    expected_sha256=entry.sha256,
                    expected_size=entry.size,
                )
            entries.append(TencentCosEntry(relative_path=entry.relative_path, ref=ref))
        final_inventory = inventory_tree(source)
        if final_inventory != inventory:
            raise ArchiveIntegrityError("archive source tree changed during COS mirroring")
        return self.publish_commit(
            TencentCosCommit(
                commit_id=commit_id,
                source_tree_sha256=inventory.tree_sha256,
                entries=tuple(entries),
            )
        )

    def publish_commit(self, commit: TencentCosCommit) -> TencentCosCommitRef:
        data = encode_tencent_cos_commit(commit)
        digest = hashlib.sha256(data).hexdigest()
        key = self._key("commits", commit.commit_id, "commit.json")
        ref = self._upload_verified_source(
            io.BytesIO(data),
            key=key,
            expected_sha256=digest,
            expected_size=len(data),
            require_absent=True,
        )
        return TencentCosCommitRef(
            commit_id=commit.commit_id,
            key=ref.key,
            version_id=ref.version_id,
            sha256=ref.sha256,
            size=ref.size,
            retain_until=ref.retain_until,
        )

    def verify_commit(self, ref: TencentCosCommitRef) -> TencentCosVerificationReceipt:
        data = self._download_verified(ref.object_ref, collect=True)
        if data is None:
            raise AssertionError("commit verification did not collect commit bytes")
        commit = decode_tencent_cos_commit(data)
        if commit.commit_id != ref.commit_id:
            raise ArchiveIntegrityError("Tencent COS commit id differs from its reference")
        expected_key = self._key("commits", ref.commit_id, "commit.json")
        if ref.key != expected_key:
            raise ArchiveIntegrityError("Tencent COS commit key differs from its digest")
        total_size = 0
        for entry in commit.entries:
            self._download_verified(entry.ref, collect=False)
            total_size += entry.ref.size
        return TencentCosVerificationReceipt(
            commit=commit,
            ref=ref,
            object_count=len(commit.entries),
            total_size=total_size,
            verified=True,
        )

    def restore_commit(
        self,
        ref: TencentCosCommitRef,
        *,
        destination: Path,
    ) -> TencentCosRestoreReceipt:
        receipt = self.verify_commit(ref)
        destination_path = destination.expanduser().absolute()
        _reject_symlink_ancestry(destination_path.parent)
        if destination_path.exists() or destination_path.is_symlink():
            raise ArchiveIntegrityError(
                f"restore destination already exists: {destination_path}"
            )
        destination_path.parent.mkdir(parents=False, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{destination_path.name}.txnopt-cos-restore-",
                dir=destination_path.parent,
            )
        )
        staging_status = staging.stat(follow_symlinks=False)
        try:
            for entry in receipt.commit.entries:
                target = staging.joinpath(*PurePosixPath(entry.relative_path).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as handle:
                    self._download_verified(entry.ref, collect=False, sink=handle)
                    handle.flush()
                    os.fsync(handle.fileno())
            _assert_staging_identity(staging, staging_status)
            _rename_noreplace(staging, destination_path)
        finally:
            if staging.exists() and not staging.is_symlink():
                shutil.rmtree(staging)
        return TencentCosRestoreReceipt(
            commit_ref=ref,
            object_count=receipt.object_count,
            total_size=receipt.total_size,
            verified=True,
        )

    def _upload_verified_source(
        self,
        source: BinaryIO,
        *,
        key: str,
        expected_sha256: str,
        expected_size: int,
        require_absent: bool = False,
    ) -> TencentCosObjectRef:
        if require_absent and self._client.object_exists(Bucket=self._bucket, Key=key):
            raise ArchiveIntegrityError(
                f"Tencent COS commit marker already exists and cannot be overwritten: {key}"
            )
        retention_floor = datetime.now(UTC) + timedelta(
            days=self._minimum_retention_days,
            minutes=-5,
        )
        with tempfile.SpooledTemporaryFile(max_size=self._multipart_threshold) as staged:
            digest = hashlib.sha256()
            observed_size = 0
            while chunk := source.read(1024 * 1024):
                if not isinstance(chunk, bytes):
                    raise TypeError("Tencent COS source must yield bytes")
                staged.write(chunk)
                digest.update(chunk)
                observed_size += len(chunk)
            if observed_size != expected_size or digest.hexdigest() != expected_sha256:
                raise ArchiveIntegrityError(
                    "Tencent COS source differs from its declared digest or size"
                )
            staged.seek(0)
            if expected_size < self._multipart_threshold:
                upload_arguments: dict[str, object] = {
                    "Bucket": self._bucket,
                    "Key": key,
                    "Body": staged,
                    "ContentLength": expected_size,
                    "Metadata": {
                        "txnopt-sha256": expected_sha256,
                        "txnopt-size": str(expected_size),
                    },
                    "StorageClass": "STANDARD",
                    "ServerSideEncryption": "AES256",
                }
                if require_absent:
                    upload_arguments["IfNoneMatch"] = "*"
                response = self._client.put_object(
                    **upload_arguments,
                )
            else:
                response = self._multipart_upload(
                    cast(BinaryIO, staged),
                    key=key,
                    sha256=expected_sha256,
                    size=expected_size,
                )
        version_id = _required_version_id(response)
        retain_until = self._read_retention_until(key=key, version_id=version_id)
        if _parse_timestamp(retain_until) < retention_floor:
            raise ArchiveIntegrityError(
                "Tencent COS object retention is shorter than the required minimum"
            )
        ref = TencentCosObjectRef(
            key=key,
            version_id=version_id,
            sha256=expected_sha256,
            size=expected_size,
            retain_until=retain_until,
        )
        self._download_verified(ref, collect=False)
        return ref

    def _multipart_upload(
        self,
        staged: BinaryIO,
        *,
        key: str,
        sha256: str,
        size: int,
    ) -> Mapping[str, Any]:
        created = self._client.create_multipart_upload(
            Bucket=self._bucket,
            Key=key,
            Metadata={"txnopt-sha256": sha256, "txnopt-size": str(size)},
            StorageClass="STANDARD",
            ServerSideEncryption="AES256",
        )
        upload_id = created.get("UploadId")
        if not isinstance(upload_id, str) or not upload_id:
            raise ArchiveIntegrityError("Tencent COS multipart upload lacks UploadId")
        parts: list[dict[str, object]] = []
        try:
            part_number = 1
            while chunk := staged.read(self._multipart_part_size):
                if not isinstance(chunk, bytes):
                    raise TypeError("Tencent COS multipart source must yield bytes")
                response = self._client.upload_part(
                    Bucket=self._bucket,
                    Key=key,
                    Body=chunk,
                    PartNumber=part_number,
                    UploadId=upload_id,
                    ContentLength=len(chunk),
                )
                etag = response.get("ETag")
                if not isinstance(etag, str) or not etag:
                    raise ArchiveIntegrityError("Tencent COS multipart part lacks ETag")
                parts.append({"ETag": etag, "PartNumber": part_number})
                part_number += 1
            return self._client.complete_multipart_upload(
                Bucket=self._bucket,
                Key=key,
                UploadId=upload_id,
                MultipartUpload={"Part": parts},
            )
        except Exception:
            self._client.abort_multipart_upload(
                Bucket=self._bucket,
                Key=key,
                UploadId=upload_id,
            )
            raise

    def _download_verified(
        self,
        ref: TencentCosObjectRef,
        *,
        collect: bool,
        sink: BinaryIO | None = None,
    ) -> bytes | None:
        if collect and sink is not None:
            raise ValueError("Tencent COS download cannot collect and stream simultaneously")
        head = _lower_keys(
            self._client.head_object(
                Bucket=self._bucket,
                Key=ref.key,
                VersionId=ref.version_id,
            )
        )
        if head.get("x-cos-version-id") != ref.version_id:
            raise ArchiveIntegrityError("Tencent COS HEAD returned a different VersionId")
        if _integer(head.get("content-length"), "COS Content-Length") != ref.size:
            raise ArchiveIntegrityError("Tencent COS object size metadata differs")
        if head.get("x-cos-meta-txnopt-sha256") != ref.sha256:
            raise ArchiveIntegrityError("Tencent COS object digest metadata differs")
        if head.get("x-cos-meta-txnopt-size") != str(ref.size):
            raise ArchiveIntegrityError("Tencent COS object size metadata differs")
        storage_class = head.get("x-cos-storage-class", "STANDARD")
        if storage_class != "STANDARD":
            raise ArchiveIntegrityError("Tencent COS evidence object is not immediately readable")
        if head.get("x-cos-server-side-encryption") != "AES256":
            raise ArchiveIntegrityError("Tencent COS server-side encryption differs")
        self._verify_retention(ref)
        response = self._client.get_object(
            Bucket=self._bucket,
            Key=ref.key,
            VersionId=ref.version_id,
        )
        normalized = _lower_keys(response)
        if normalized.get("x-cos-version-id") != ref.version_id:
            raise ArchiveIntegrityError("Tencent COS GET returned a different VersionId")
        stream = _response_stream(response)
        digest = hashlib.sha256()
        size = 0
        collected = bytearray() if collect else None
        read_error: TencentCosError | None = None
        close_error: TencentCosError | None = None
        try:
            while True:
                try:
                    chunk = stream.read(1024 * 1024)
                except Exception as error:
                    read_error = _redacted_sdk_error("response_body.read", error)
                    break
                if chunk in {b"", ""}:
                    break
                if not isinstance(chunk, bytes):
                    raise TypeError("Tencent COS response body must yield bytes")
                digest.update(chunk)
                size += len(chunk)
                if collected is not None:
                    collected.extend(chunk)
                if sink is not None:
                    written = sink.write(chunk)
                    if written != len(chunk):
                        raise OSError("Tencent COS restore write was incomplete")
        finally:
            try:
                close = getattr(stream, "close", None)
                if callable(close):
                    close()
            except Exception as error:
                close_error = _redacted_sdk_error("response_body.close", error)
        if read_error is not None:
            raise read_error
        if close_error is not None:
            raise close_error
        if size != ref.size or digest.hexdigest() != ref.sha256:
            raise ArchiveIntegrityError(
                "Tencent COS object bytes differ from their exact version reference"
            )
        return bytes(collected) if collected is not None else None

    def _verify_retention(self, ref: TencentCosObjectRef) -> None:
        until = self._read_retention_until(key=ref.key, version_id=ref.version_id)
        if _parse_timestamp(until) <= datetime.now(UTC):
            raise ArchiveIntegrityError("Tencent COS object retention has expired")
        if until != ref.retain_until:
            raise ArchiveIntegrityError("Tencent COS object retain-until differs")

    def _read_retention_until(self, *, key: str, version_id: str) -> str:
        payload = self._client.get_object_retention(
            Bucket=self._bucket,
            Key=key,
            VersionId=version_id,
        )
        retention = _required_mapping(payload.get("Retention", payload), "object retention")
        if retention.get("Mode") != "COMPLIANCE":
            raise ArchiveIntegrityError("Tencent COS object retention is not COMPLIANCE")
        until = retention.get("RetainUntilDate")
        if not isinstance(until, str):
            raise ArchiveIntegrityError("Tencent COS object retention date is absent")
        _parse_timestamp(until)
        return until

    def _key(self, *parts: str) -> str:
        return "/".join((self._prefix, *parts))


class _QcloudCosSdkBridge:
    """Thin isolation layer for the official ``cos-python-sdk-v5`` client."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def head_bucket(self, **kwargs: object) -> Mapping[str, Any]:
        return cast(
            Mapping[str, Any],
            self._call("head_bucket", self._client.head_bucket, **kwargs),
        )

    def get_bucket_versioning(self, **kwargs: object) -> Mapping[str, Any]:
        return cast(
            Mapping[str, Any],
            self._call("get_bucket_versioning", self._client.get_bucket_versioning, **kwargs),
        )

    def get_bucket_object_lock(self, **kwargs: object) -> Mapping[str, Any]:
        return cast(
            Mapping[str, Any],
            self._call(
                "get_bucket_object_lock",
                self._client.get_bucket_object_lock,
                **kwargs,
            ),
        )

    def put_object(self, **kwargs: object) -> Mapping[str, Any]:
        return cast(
            Mapping[str, Any],
            self._call("put_object", self._client.put_object, **kwargs),
        )

    def create_multipart_upload(self, **kwargs: object) -> Mapping[str, Any]:
        return cast(
            Mapping[str, Any],
            self._call(
                "create_multipart_upload",
                self._client.create_multipart_upload,
                **kwargs,
            ),
        )

    def upload_part(self, **kwargs: object) -> Mapping[str, Any]:
        return cast(
            Mapping[str, Any],
            self._call("upload_part", self._client.upload_part, **kwargs),
        )

    def complete_multipart_upload(self, **kwargs: object) -> Mapping[str, Any]:
        return cast(
            Mapping[str, Any],
            self._call(
                "complete_multipart_upload",
                self._client.complete_multipart_upload,
                **kwargs,
            ),
        )

    def abort_multipart_upload(self, **kwargs: object) -> object:
        return self._call("abort_multipart_upload", self._client.abort_multipart_upload, **kwargs)

    def head_object(self, **kwargs: object) -> Mapping[str, Any]:
        return cast(
            Mapping[str, Any],
            self._call("head_object", self._client.head_object, **kwargs),
        )

    def get_object(self, **kwargs: object) -> Mapping[str, Any]:
        return cast(
            Mapping[str, Any],
            self._call("get_object", self._client.get_object, **kwargs),
        )

    def get_object_retention(self, **kwargs: object) -> Mapping[str, Any]:
        return self._call("get_object_retention", self._get_object_retention, **kwargs)

    def _get_object_retention(self, **kwargs: object) -> Mapping[str, Any]:
        auth_module = importlib.import_module("qcloud_cos.cos_auth")
        comm_module = importlib.import_module("qcloud_cos.cos_comm")
        bucket = cast(str, kwargs["Bucket"])
        key = cast(str, kwargs["Key"])
        version_id = cast(str, kwargs["VersionId"])
        config = self._client._conf
        params = {"retention": "", "versionId": version_id}
        response = self._client.send_request(
            method="GET",
            url=config.uri(bucket=bucket, path=key),
            bucket=bucket,
            auth=auth_module.CosS3Auth(config, key, params=params),
            params=params,
        )
        return cast(Mapping[str, Any], comm_module.xml_to_dict(response.content))

    def object_exists(self, **kwargs: object) -> bool:
        exists = True
        sdk_error: TencentCosError | None = None
        try:
            self._client.head_object(**kwargs)
        except Exception as error:
            status_code: object = None
            status_sdk_error: TencentCosError | None = None
            try:
                status_code = getattr(error, "get_status_code", lambda: None)()
            except Exception as status_error:
                status_sdk_error = _redacted_sdk_error(
                    "object_exists status",
                    status_error,
                )
            if status_sdk_error is not None:
                sdk_error = status_sdk_error
            elif status_code == 404:
                exists = False
            else:
                sdk_error = _redacted_sdk_error("object_exists", error)
        if sdk_error is not None:
            raise sdk_error
        return exists

    def _call(
        self,
        operation: str,
        method: Callable[..., _SdkResult],
        **kwargs: object,
    ) -> _SdkResult:
        result: _SdkResult | None = None
        sdk_error: TencentCosError | None = None
        try:
            result = method(**kwargs)
        except Exception as error:
            sdk_error = _redacted_sdk_error(operation, error)
        if sdk_error is not None:
            raise sdk_error
        return cast(_SdkResult, result)


def encode_tencent_cos_commit(commit: TencentCosCommit) -> bytes:
    return canonical_json_bytes(commit.to_payload(), pretty=True)


def decode_tencent_cos_commit(data: bytes) -> TencentCosCommit:
    try:
        payload: object = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArchiveIntegrityError("Tencent COS commit is not valid JSON") from error
    if canonical_json_bytes(payload, pretty=True) != data:
        raise ArchiveIntegrityError("Tencent COS commit is not canonical JSON")
    return _commit_from_payload(payload)


def _commit_from_payload(payload: object) -> TencentCosCommit:
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "commit_id",
        "source_tree_sha256",
        "objects",
    }:
        raise ArchiveIntegrityError("Tencent COS commit field set differs")
    if payload.get("schema_version") != _COMMIT_SCHEMA:
        raise ArchiveIntegrityError("Tencent COS commit schema differs")
    commit_id = payload.get("commit_id")
    tree = payload.get("source_tree_sha256")
    objects = payload.get("objects")
    if not isinstance(commit_id, str) or not isinstance(tree, str) or not isinstance(objects, list):
        raise ArchiveIntegrityError("Tencent COS commit header types differ")
    entries: list[TencentCosEntry] = []
    for item in objects:
        if not isinstance(item, dict) or set(item) != {
            "relative_path",
            "key",
            "version_id",
            "sha256",
            "size",
            "retain_until",
        }:
            raise ArchiveIntegrityError("Tencent COS commit object field set differs")
        relative_path = item.get("relative_path")
        key = item.get("key")
        version_id = item.get("version_id")
        sha256 = item.get("sha256")
        size = item.get("size")
        retain_until = item.get("retain_until")
        if (
            not isinstance(relative_path, str)
            or not isinstance(key, str)
            or not isinstance(version_id, str)
            or not isinstance(sha256, str)
            or type(size) is not int
            or not isinstance(retain_until, str)
        ):
            raise ArchiveIntegrityError("Tencent COS commit object types differ")
        try:
            entries.append(
                TencentCosEntry(
                    relative_path=relative_path,
                    ref=TencentCosObjectRef(
                        key=key,
                        version_id=version_id,
                        sha256=sha256,
                        size=size,
                        retain_until=retain_until,
                    ),
                )
            )
        except ValueError as error:
            raise ArchiveIntegrityError("Tencent COS commit object identity differs") from error
    try:
        return TencentCosCommit(
            commit_id=commit_id,
            source_tree_sha256=tree,
            entries=tuple(entries),
        )
    except ValueError as error:
        raise ArchiveIntegrityError("Tencent COS commit identity differs") from error


def _response_stream(response: Mapping[str, Any]) -> BinaryIO:
    body = response.get("Body")
    stream: object = None
    sdk_error: TencentCosError | None = None
    try:
        raw_factory = getattr(body, "get_raw_stream", None)
        stream = raw_factory() if callable(raw_factory) else body
    except Exception as error:
        sdk_error = _redacted_sdk_error("response_body.get_raw_stream", error)
    if sdk_error is not None:
        raise sdk_error
    readable = False
    try:
        readable = callable(getattr(stream, "read", None))
    except Exception as error:
        sdk_error = _redacted_sdk_error("response_body.read", error)
    if sdk_error is not None:
        raise sdk_error
    if stream is None or not readable:
        raise ArchiveIntegrityError("Tencent COS response lacks a readable body")
    return cast(BinaryIO, stream)


def _required_version_id(response: Mapping[str, Any]) -> str:
    normalized = _lower_keys(response)
    version_id = normalized.get("x-cos-version-id")
    if not isinstance(version_id, str) or not version_id:
        raise ArchiveIntegrityError("Tencent COS upload response lacks VersionId")
    _validate_version_id(version_id)
    return version_id


def _lower_keys(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key).lower(): value for key, value in payload.items()}


def _redacted_sdk_error(
    operation: str,
    error: Exception,
) -> TencentCosError:
    """Convert an SDK failure without exporting its untrusted message body."""

    return TencentCosError(
        safe_sdk_error_text(
            provider="Tencent COS",
            operation=operation,
            error=error,
        )
    )


def _required_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TencentCosContractError(f"{name} is absent or malformed")
    return cast(Mapping[str, Any], value)


def _retention_days(retention: Mapping[str, Any]) -> int:
    days = retention.get("Days")
    years = retention.get("Years")
    if days is not None and years is not None:
        raise TencentCosContractError("Tencent COS retention declares days and years")
    if days is not None:
        return _integer(days, "Tencent COS retention days")
    if years is not None:
        return _integer(years, "Tencent COS retention years") * 365
    raise TencentCosContractError("Tencent COS default retention duration is absent")


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ArchiveIntegrityError(f"{name} is not an integer")
    try:
        result = int(cast(Any, value))
    except (TypeError, ValueError) as error:
        raise ArchiveIntegrityError(f"{name} is not an integer") from error
    if result < 0:
        raise ArchiveIntegrityError(f"{name} is negative")
    return result


def _parse_timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ArchiveIntegrityError("Tencent COS retention date is malformed") from error
    if parsed.tzinfo is None:
        raise ArchiveIntegrityError("Tencent COS retention date lacks a timezone")
    return parsed.astimezone(UTC)


def _validate_prefix(value: str) -> str:
    normalized = value.strip("/")
    if not normalized or "\\" in normalized:
        raise ValueError("Tencent COS prefix is empty or malformed")
    path = PurePosixPath(normalized)
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("Tencent COS prefix is not canonical")
    return path.as_posix()


def _validate_key(value: str) -> None:
    if not value or value.startswith("/") or "\\" in value:
        raise ValueError("Tencent COS key is not canonical")
    if any(part in {"", ".", ".."} for part in PurePosixPath(value).parts):
        raise ValueError("Tencent COS key is not canonical")


def _validate_version_id(value: str) -> None:
    if not value or len(value) > 1024 or any(ord(character) < 0x20 for character in value):
        raise ValueError("Tencent COS VersionId is absent or malformed")


def _validate_sha256(value: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("SHA-256 identity must be 64 lowercase hexadecimal characters")


def _validate_commit_id(value: str) -> None:
    if _COMMIT_ID.fullmatch(value) is None:
        raise ValueError("Tencent COS commit id is not path-safe")


def _validate_relative_path(value: str) -> None:
    candidate = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or candidate.is_absolute()
        or value != candidate.as_posix()
        or any(part in {"", ".", ".."} for part in candidate.parts)
    ):
        raise ValueError("Tencent COS archive path is not canonical")


def _reject_symlink_ancestry(path: Path) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        status = current.stat(follow_symlinks=False)
        if stat.S_ISLNK(status.st_mode):
            raise ArchiveIntegrityError("restore destination ancestry contains a symlink")
        if not stat.S_ISDIR(status.st_mode):
            raise ArchiveIntegrityError("restore destination parent is not a directory")


def _assert_staging_identity(path: Path, expected: os.stat_result) -> None:
    observed = path.stat(follow_symlinks=False)
    if (
        not stat.S_ISDIR(observed.st_mode)
        or observed.st_dev != expected.st_dev
        or observed.st_ino != expected.st_ino
    ):
        raise ArchiveIntegrityError("restore staging directory identity changed")


def _rename_noreplace(source: Path, destination: Path) -> None:
    function = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if function is None:
        raise ArchiveIntegrityError("atomic no-replace restore is unavailable")
    function.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    function.restype = ctypes.c_int
    result = function(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise ArchiveIntegrityError(f"restore destination already exists: {destination}")
    raise OSError(error_number, os.strerror(error_number), destination)


__all__ = [
    "TencentCosArchive",
    "TencentCosBucketReceipt",
    "TencentCosCommit",
    "TencentCosCommitRef",
    "TencentCosContractError",
    "TencentCosEntry",
    "TencentCosError",
    "TencentCosObjectRef",
    "TencentCosObjectVerificationReceipt",
    "TencentCosRestoreReceipt",
    "TencentCosVerificationReceipt",
    "decode_tencent_cos_commit",
    "encode_tencent_cos_commit",
]
