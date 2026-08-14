"""S3-compatible immutable archive adapter with externally enforced WORM policy."""

from __future__ import annotations

import base64
import hashlib
import importlib
import io
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import PurePosixPath
from typing import Any, BinaryIO, Protocol, Self, cast

from txnopt_evidence.archive import (
    ArchiveCommit,
    ArchiveCommitRef,
    ArchiveConflictError,
    ArchiveError,
    ArchiveIntegrityError,
    ArchiveNotFoundError,
    ArchiveObjectMeta,
    ArchiveObjectRef,
    ArchiveVerificationReceipt,
    decode_archive_commit,
    encode_archive_commit,
)

_CHUNK_SIZE = 1024 * 1024
_MINIMUM_S3_PART_SIZE = 5 * 1024 * 1024
_MAXIMUM_S3_PART_SIZE = 5 * 1024 * 1024 * 1024
_SINGLE_PUT_LIMIT = 5 * 1024 * 1024 * 1024
_METADATA_SCHEMA = "txnopt-archive-object-v1"


class S3Client(Protocol):
    """Small structural interface implemented by boto3's S3 client."""

    def head_bucket(self, **kwargs: object) -> dict[str, Any]: ...

    def get_bucket_versioning(self, **kwargs: object) -> dict[str, Any]: ...

    def get_object_lock_configuration(self, **kwargs: object) -> dict[str, Any]: ...

    def head_object(self, **kwargs: object) -> dict[str, Any]: ...

    def get_object(self, **kwargs: object) -> dict[str, Any]: ...

    def get_object_retention(self, **kwargs: object) -> dict[str, Any]: ...

    def put_object(self, **kwargs: object) -> dict[str, Any]: ...

    def create_multipart_upload(self, **kwargs: object) -> dict[str, Any]: ...

    def upload_part(self, **kwargs: object) -> dict[str, Any]: ...

    def complete_multipart_upload(self, **kwargs: object) -> dict[str, Any]: ...

    def abort_multipart_upload(self, **kwargs: object) -> dict[str, Any]: ...


class _ResponseBody(Protocol):
    def read(self, amount: int = -1) -> bytes: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class S3BucketProtectionReceipt:
    bucket: str
    versioning_enabled: bool
    object_lock_enabled: bool
    retention_mode: str
    retention_days: int
    verified: bool


class S3ArchiveStore:
    """Immutable ArchiveStore over AWS S3 or a compatible object API.

    The adapter never lists, renames, overwrites, or deletes published objects.
    A bucket-level COMPLIANCE retention rule is required before any operation.
    """

    def __init__(
        self,
        client: S3Client,
        *,
        bucket: str,
        prefix: str = "txnopt",
        minimum_retention_days: int = 365,
        multipart_threshold: int = _SINGLE_PUT_LIMIT,
        multipart_part_size: int = 64 * 1024 * 1024,
        send_checksum_sha256: bool = True,
        server_side_encryption: str | None = "AES256",
        storage_class: str | None = None,
        verify_bucket: bool = True,
    ) -> None:
        if not bucket or any(character.isspace() for character in bucket):
            raise ValueError("S3 archive bucket must be a non-empty canonical name")
        if minimum_retention_days < 1:
            raise ValueError("S3 archive retention must be at least one day")
        if multipart_threshold < 1:
            raise ValueError("S3 multipart threshold must be positive")
        if not _MINIMUM_S3_PART_SIZE <= multipart_part_size <= _MAXIMUM_S3_PART_SIZE:
            raise ValueError("S3 multipart part size is outside the service contract")
        if storage_class in {"DEEP_ARCHIVE", "GLACIER"}:
            raise ValueError(
                "S3 archive objects must remain immediately readable until verification"
            )
        self._client = client
        self._bucket = bucket
        self._prefix = _canonical_prefix(prefix)
        self._minimum_retention_days = minimum_retention_days
        self._multipart_threshold = multipart_threshold
        self._multipart_part_size = multipart_part_size
        self._send_checksum_sha256 = send_checksum_sha256
        self._server_side_encryption = server_side_encryption
        self._storage_class = storage_class
        self._bucket_receipt: S3BucketProtectionReceipt | None = None
        if verify_bucket:
            self._bucket_receipt = self.verify_bucket_contract()

    @classmethod
    def from_boto3(
        cls,
        *,
        bucket: str,
        prefix: str = "txnopt",
        region_name: str | None = None,
        endpoint_url: str | None = None,
        minimum_retention_days: int = 365,
        send_checksum_sha256: bool = True,
        server_side_encryption: str | None = "AES256",
        storage_class: str | None = None,
    ) -> Self:
        """Build the adapter from boto3's standard credential provider chain."""

        if endpoint_url is not None and not endpoint_url.startswith("https://"):
            raise ValueError("S3 archive endpoint must use HTTPS")
        try:
            boto3: Any = importlib.import_module("boto3")
        except ImportError as error:
            raise ArchiveError(
                "S3 archive support requires the optional txnopt[s3] dependency"
            ) from error
        client = cast(
            S3Client,
            boto3.client(
                "s3",
                region_name=region_name,
                endpoint_url=endpoint_url,
            ),
        )
        return cls(
            client,
            bucket=bucket,
            prefix=prefix,
            minimum_retention_days=minimum_retention_days,
            send_checksum_sha256=send_checksum_sha256,
            server_side_encryption=server_side_encryption,
            storage_class=storage_class,
        )

    @property
    def bucket_receipt(self) -> S3BucketProtectionReceipt:
        if self._bucket_receipt is None:
            self._bucket_receipt = self.verify_bucket_contract()
        return self._bucket_receipt

    def verify_bucket_contract(self) -> S3BucketProtectionReceipt:
        try:
            self._client.head_bucket(Bucket=self._bucket)
            versioning = self._client.get_bucket_versioning(Bucket=self._bucket)
            lock_response = self._client.get_object_lock_configuration(
                Bucket=self._bucket
            )
        except Exception as error:
            raise ArchiveError("S3 archive bucket protection could not be verified") from error
        if versioning.get("Status") != "Enabled":
            raise ArchiveIntegrityError("S3 archive bucket versioning is not enabled")
        configuration = lock_response.get("ObjectLockConfiguration")
        if not isinstance(configuration, dict) or configuration.get(
            "ObjectLockEnabled"
        ) != "Enabled":
            raise ArchiveIntegrityError("S3 archive bucket Object Lock is not enabled")
        rule = configuration.get("Rule")
        retention = rule.get("DefaultRetention") if isinstance(rule, dict) else None
        if not isinstance(retention, dict) or retention.get("Mode") != "COMPLIANCE":
            raise ArchiveIntegrityError(
                "S3 archive bucket requires COMPLIANCE default retention"
            )
        retention_days = _retention_days(retention)
        if retention_days < self._minimum_retention_days:
            raise ArchiveIntegrityError(
                "S3 archive bucket retention is shorter than the required minimum"
            )
        receipt = S3BucketProtectionReceipt(
            bucket=self._bucket,
            versioning_enabled=True,
            object_lock_enabled=True,
            retention_mode="COMPLIANCE",
            retention_days=retention_days,
            verified=True,
        )
        self._bucket_receipt = receipt
        return receipt

    def put_blob(
        self,
        source: BinaryIO,
        *,
        expected_sha256: str,
        expected_size: int,
    ) -> ArchiveObjectRef:
        ref = ArchiveObjectRef(sha256=expected_sha256, size=expected_size)
        with tempfile.TemporaryFile(mode="w+b") as staged:
            observed_sha256, observed_size = _copy_and_hash(source, staged)
            if observed_sha256 != ref.sha256 or observed_size != ref.size:
                raise ArchiveIntegrityError(
                    "archive blob differs from expected digest or size"
                )
            key = self._blob_key(ref)
            if self._declared_object_exists(key, ref, object_kind="blob"):
                with self.open(ref):
                    return ref
            staged.seek(0)
            created = self._conditional_upload(
                key,
                staged,
                ref=ref,
                object_kind="blob",
            )
            if not created and not self._declared_object_exists(
                key,
                ref,
                object_kind="blob",
            ):
                raise ArchiveConflictError(
                    "S3 archive blob key is bound to another immutable object"
                )
        with self.open(ref):
            return ref

    def open(self, ref: ArchiveObjectRef) -> BinaryIO:
        key = self._blob_key(ref)
        self._require_declared_object(key, ref, object_kind="blob")
        return self._download_verified(key, ref)

    def head(self, ref: ArchiveObjectRef) -> ArchiveObjectMeta:
        with self.open(ref):
            return ArchiveObjectMeta(ref=ref, verified=True)

    def publish_commit(
        self,
        commit: ArchiveCommit,
        *,
        expected_absent: bool = True,
    ) -> ArchiveCommitRef:
        for entry in commit.entries:
            with self.open(entry.ref):
                pass
        data = encode_archive_commit(commit)
        object_ref = ArchiveObjectRef(
            sha256=hashlib.sha256(data).hexdigest(),
            size=len(data),
        )
        ref = ArchiveCommitRef(
            commit_id=commit.commit_id,
            sha256=object_ref.sha256,
            size=object_ref.size,
        )
        key = self._commit_key(commit.commit_id)
        created = self._conditional_upload(
            key,
            io.BytesIO(data),
            ref=object_ref,
            object_kind="commit",
        )
        if not created:
            if expected_absent:
                raise ArchiveConflictError(
                    f"S3 archive commit already exists: {commit.commit_id}"
                )
            self._require_declared_object(
                key,
                object_ref,
                object_kind="commit",
            )
            with self._download_verified(key, object_ref) as existing:
                if existing.read() != data:
                    raise ArchiveConflictError(
                        "S3 archive commit name is bound to other bytes"
                    )
        return ref

    def verify_commit(self, ref: ArchiveCommitRef) -> ArchiveVerificationReceipt:
        object_ref = ArchiveObjectRef(sha256=ref.sha256, size=ref.size)
        key = self._commit_key(ref.commit_id)
        self._require_declared_object(
            key,
            object_ref,
            object_kind="commit",
        )
        with self._download_verified(key, object_ref) as handle:
            commit = decode_archive_commit(handle.read())
        if commit.commit_id != ref.commit_id:
            raise ArchiveIntegrityError(
                "S3 archive commit identity differs from its reference"
            )
        total_size = 0
        for entry in commit.entries:
            with self.open(entry.ref):
                total_size += entry.ref.size
        return ArchiveVerificationReceipt(
            commit=commit,
            ref=ref,
            object_count=len(commit.entries),
            total_size=total_size,
            verified=True,
        )

    def _conditional_upload(
        self,
        key: str,
        source: BinaryIO,
        *,
        ref: ArchiveObjectRef,
        object_kind: str,
    ) -> bool:
        if ref.size < self._multipart_threshold:
            return self._conditional_single_put(
                key,
                source,
                ref=ref,
                object_kind=object_kind,
            )
        return self._conditional_multipart_put(
            key,
            source,
            ref=ref,
            object_kind=object_kind,
        )

    def _conditional_single_put(
        self,
        key: str,
        source: BinaryIO,
        *,
        ref: ArchiveObjectRef,
        object_kind: str,
    ) -> bool:
        if ref.size > _SINGLE_PUT_LIMIT:
            raise ArchiveError("S3 single PUT exceeds the service size limit")
        source.seek(0)
        arguments = self._put_arguments(key, ref, object_kind=object_kind)
        arguments.update({"Body": source, "ContentLength": ref.size, "IfNoneMatch": "*"})
        if self._send_checksum_sha256:
            arguments["ChecksumSHA256"] = _base64_sha256(ref.sha256)
        try:
            self._client.put_object(**arguments)
        except Exception as error:
            if _is_conditional_conflict(error):
                return False
            if self._declared_object_exists(key, ref, object_kind=object_kind):
                return False
            raise ArchiveError("S3 archive object publication outcome is unknown") from error
        return True

    def _conditional_multipart_put(
        self,
        key: str,
        source: BinaryIO,
        *,
        ref: ArchiveObjectRef,
        object_kind: str,
    ) -> bool:
        if ref.size > self._multipart_part_size * 10_000:
            raise ArchiveError("S3 multipart upload would exceed 10,000 parts")
        arguments = self._put_arguments(key, ref, object_kind=object_kind)
        if self._send_checksum_sha256:
            arguments["ChecksumAlgorithm"] = "SHA256"
        try:
            created = self._client.create_multipart_upload(**arguments)
        except Exception as error:
            raise ArchiveError("S3 multipart upload could not be created") from error
        upload_id = created.get("UploadId")
        if not isinstance(upload_id, str) or not upload_id:
            raise ArchiveError("S3 multipart upload did not return an upload id")
        source.seek(0)
        parts: list[dict[str, object]] = []
        try:
            part_number = 1
            while chunk := source.read(self._multipart_part_size):
                if not isinstance(chunk, bytes):
                    raise TypeError("archive blob source must yield bytes")
                part_arguments: dict[str, object] = {
                    "Bucket": self._bucket,
                    "Key": key,
                    "UploadId": upload_id,
                    "PartNumber": part_number,
                    "Body": chunk,
                    "ContentLength": len(chunk),
                }
                checksum = base64.b64encode(hashlib.sha256(chunk).digest()).decode("ascii")
                if self._send_checksum_sha256:
                    part_arguments["ChecksumSHA256"] = checksum
                uploaded = self._client.upload_part(**part_arguments)
                etag = uploaded.get("ETag")
                if not isinstance(etag, str) or not etag:
                    raise ArchiveError("S3 multipart upload part lacks an ETag")
                part: dict[str, object] = {"ETag": etag, "PartNumber": part_number}
                if self._send_checksum_sha256:
                    part["ChecksumSHA256"] = checksum
                parts.append(part)
                part_number += 1
            complete_arguments: dict[str, object] = {
                "Bucket": self._bucket,
                "Key": key,
                "UploadId": upload_id,
                "MultipartUpload": {"Parts": parts},
                "IfNoneMatch": "*",
            }
            self._client.complete_multipart_upload(**complete_arguments)
            return True
        except Exception as error:
            with suppress(Exception):
                self._client.abort_multipart_upload(
                    Bucket=self._bucket,
                    Key=key,
                    UploadId=upload_id,
                )
            if _is_conditional_conflict(error):
                return False
            if self._declared_object_exists(key, ref, object_kind=object_kind):
                return False
            raise ArchiveError("S3 multipart publication outcome is unknown") from error

    def _put_arguments(
        self,
        key: str,
        ref: ArchiveObjectRef,
        *,
        object_kind: str,
    ) -> dict[str, object]:
        arguments: dict[str, object] = {
            "Bucket": self._bucket,
            "Key": key,
            "ContentType": "application/octet-stream",
            "Metadata": _object_metadata(ref, object_kind=object_kind),
            "ObjectLockMode": "COMPLIANCE",
            "ObjectLockRetainUntilDate": datetime.now(UTC)
            + timedelta(days=self._minimum_retention_days),
        }
        if self._server_side_encryption is not None:
            arguments["ServerSideEncryption"] = self._server_side_encryption
        if self._storage_class is not None:
            arguments["StorageClass"] = self._storage_class
        return arguments

    def _declared_object_exists(
        self,
        key: str,
        ref: ArchiveObjectRef,
        *,
        object_kind: str,
    ) -> bool:
        try:
            response = self._client.head_object(Bucket=self._bucket, Key=key)
        except Exception as error:
            if _is_not_found(error):
                return False
            raise ArchiveError("S3 archive object metadata could not be read") from error
        _validate_head_response(response, ref, object_kind=object_kind)
        self._validate_remote_retention(key)
        return True

    def _validate_remote_retention(self, key: str) -> None:
        try:
            response = self._client.get_object_retention(
                Bucket=self._bucket,
                Key=key,
            )
        except Exception as error:
            raise ArchiveError("S3 archive object retention could not be read") from error
        retention = response.get("Retention")
        if not isinstance(retention, dict) or retention.get("Mode") != "COMPLIANCE":
            raise ArchiveIntegrityError(
                "S3 archive object lacks COMPLIANCE retention"
            )
        retain_until = retention.get("RetainUntilDate")
        if not isinstance(retain_until, datetime):
            raise ArchiveIntegrityError("S3 archive object retention date is malformed")
        normalized = (
            retain_until.replace(tzinfo=UTC)
            if retain_until.tzinfo is None
            else retain_until.astimezone(UTC)
        )
        if normalized <= datetime.now(UTC):
            raise ArchiveIntegrityError("S3 archive object retention has expired")

    def _require_declared_object(
        self,
        key: str,
        ref: ArchiveObjectRef,
        *,
        object_kind: str,
    ) -> None:
        if not self._declared_object_exists(key, ref, object_kind=object_kind):
            raise ArchiveNotFoundError(f"S3 archive {object_kind} is absent: {key}")

    def _download_verified(self, key: str, ref: ArchiveObjectRef) -> BinaryIO:
        try:
            response = self._client.get_object(Bucket=self._bucket, Key=key)
        except Exception as error:
            if _is_not_found(error):
                raise ArchiveNotFoundError(f"S3 archive object is absent: {key}") from error
            raise ArchiveError("S3 archive object could not be downloaded") from error
        body = response.get("Body")
        if body is None or not hasattr(body, "read") or not hasattr(body, "close"):
            raise ArchiveError("S3 archive response lacks a readable body")
        readable = cast(_ResponseBody, body)
        staged = tempfile.TemporaryFile(mode="w+b")  # noqa: SIM115 - returned to caller
        digest = hashlib.sha256()
        size = 0
        try:
            while chunk := readable.read(_CHUNK_SIZE):
                if not isinstance(chunk, bytes):
                    raise TypeError("S3 archive body must yield bytes")
                staged.write(chunk)
                digest.update(chunk)
                size += len(chunk)
        except Exception:
            staged.close()
            raise
        finally:
            readable.close()
        if digest.hexdigest() != ref.sha256 or size != ref.size:
            staged.close()
            raise ArchiveIntegrityError(
                "S3 archive object differs from its immutable reference"
            )
        staged.seek(0)
        return staged

    def _blob_key(self, ref: ArchiveObjectRef) -> str:
        return self._key(f"blobs/sha256/{ref.sha256[:2]}/{ref.sha256}")

    def _commit_key(self, commit_id: str) -> str:
        ArchiveCommit(commit_id=commit_id, entries=())
        return self._key(f"commits/{commit_id}.json")

    def _key(self, suffix: str) -> str:
        return f"{self._prefix}/{suffix}" if self._prefix else suffix


def _copy_and_hash(source: BinaryIO, destination: BinaryIO) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    while chunk := source.read(_CHUNK_SIZE):
        if not isinstance(chunk, bytes):
            raise TypeError("archive blob source must yield bytes")
        destination.write(chunk)
        digest.update(chunk)
        size += len(chunk)
    destination.flush()
    destination.seek(0)
    return digest.hexdigest(), size


def _canonical_prefix(value: str) -> str:
    if value == "":
        return value
    candidate = PurePosixPath(value)
    if (
        value.startswith("/")
        or value.endswith("/")
        or "\\" in value
        or value != candidate.as_posix()
        or any(part in {"", ".", ".."} for part in candidate.parts)
    ):
        raise ValueError("S3 archive prefix must be canonical and traversal-free")
    return value


def _retention_days(retention: dict[str, object]) -> int:
    days = retention.get("Days")
    years = retention.get("Years")
    if type(days) is int and days > 0 and years is None:
        return days
    if type(years) is int and years > 0 and days is None:
        return years * 365
    raise ArchiveIntegrityError("S3 archive default retention duration is malformed")


def _object_metadata(ref: ArchiveObjectRef, *, object_kind: str) -> dict[str, str]:
    return {
        "txnopt-schema": _METADATA_SCHEMA,
        "txnopt-kind": object_kind,
        "txnopt-sha256": ref.sha256,
        "txnopt-size": str(ref.size),
    }


def _validate_head_response(
    response: dict[str, Any],
    ref: ArchiveObjectRef,
    *,
    object_kind: str,
) -> None:
    if response.get("ContentLength") != ref.size:
        raise ArchiveIntegrityError("S3 archive object size metadata differs")
    metadata = response.get("Metadata")
    if not isinstance(metadata, dict) or {
        str(key).lower(): str(value) for key, value in metadata.items()
    } != _object_metadata(ref, object_kind=object_kind):
        raise ArchiveIntegrityError("S3 archive immutable metadata differs")


def _base64_sha256(hexadecimal: str) -> str:
    return base64.b64encode(bytes.fromhex(hexadecimal)).decode("ascii")


def _service_error_code(error: Exception) -> str:
    response = getattr(error, "response", None)
    if not isinstance(response, dict):
        return ""
    detail = response.get("Error")
    if isinstance(detail, dict):
        code = detail.get("Code")
        if isinstance(code, str):
            return code
    metadata = response.get("ResponseMetadata")
    if isinstance(metadata, dict):
        status = metadata.get("HTTPStatusCode")
        if type(status) is int:
            return str(status)
    return ""


def _is_not_found(error: Exception) -> bool:
    return _service_error_code(error) in {"404", "NoSuchKey", "NotFound"}


def _is_conditional_conflict(error: Exception) -> bool:
    return _service_error_code(error) in {
        "409",
        "412",
        "ConditionalRequestConflict",
        "PreconditionFailed",
    }


__all__ = ["S3ArchiveStore", "S3BucketProtectionReceipt", "S3Client"]
