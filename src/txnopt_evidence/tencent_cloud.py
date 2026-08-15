"""Offline Tencent Cloud capacity, provisioning, and live-host checks.

This module is deliberately Tencent-specific.  It does not expose a generic
cloud-provider seam and it never constructs a purchasing request: the only CVM
request shape it can produce has ``DryRun`` fixed to ``True``.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import shutil
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from txnopt_evidence._safe_sdk_error import safe_sdk_error_text
from txnopt_evidence.codec import (
    canonical_json_bytes,
    read_signed_json,
    verify_sidecar,
)
from txnopt_evidence.level1_campaign_common import (
    linux_host_identity,
    physical_core_count,
)
from txnopt_evidence.runtime_estimate import estimate_cloud_window


@dataclass(frozen=True, slots=True)
class TencentCloudRequirements:
    minimum_physical_cores: int = 64
    minimum_provider_memory_gb: int = 128
    scheduler_efficiency: float = 0.8
    maximum_matrix_days: int = 10

    def __post_init__(self) -> None:
        if self.minimum_physical_cores < 64:
            raise ValueError("Tencent cloud profile must require at least 64 physical cores")
        if self.minimum_provider_memory_gb < 128:
            raise ValueError("Tencent cloud profile must require at least 128 GB")
        if not 0.0 < self.scheduler_efficiency <= 1.0:
            raise ValueError("scheduler efficiency must be in (0, 1]")


@dataclass(frozen=True, slots=True)
class TencentCvmSelection:
    """Live Tencent CVM values that may not have repository defaults."""

    region: str
    zone: str
    instance_type: str
    image_id: str
    vpc_id: str
    subnet_id: str
    security_group_id: str

    def __post_init__(self) -> None:
        for name in (
            "region",
            "zone",
            "instance_type",
            "image_id",
            "vpc_id",
            "subnet_id",
            "security_group_id",
        ):
            value = getattr(self, name)
            if not value or value != value.strip():
                raise ValueError(f"Tencent CVM {name} is required and must be canonical")
@dataclass(frozen=True, slots=True)
class TencentProviderInstanceSpec:
    """Tencent API or product-spec evidence for one exact CVM instance type."""

    instance_type: str
    physical_cores: int
    memory_gb: int

    def __post_init__(self) -> None:
        if not self.instance_type or self.instance_type != self.instance_type.strip():
            raise ValueError("Tencent instance type is required and must be canonical")
        if isinstance(self.physical_cores, bool) or self.physical_cores <= 0:
            raise ValueError("Tencent provider physical-core count must be positive")
        if isinstance(self.memory_gb, bool) or self.memory_gb <= 0:
            raise ValueError("Tencent provider memory must be positive")


@dataclass(frozen=True, slots=True)
class TencentProviderApiEvidence:
    """Provider facts acknowledged by one exact Tencent CVM DryRun receipt."""

    region: str
    zone: str
    instance_type: str
    physical_cores: int
    thread_per_core: int
    vcpu: int
    memory_gb: int
    describe_request_id: str
    dry_run_request_id: str

    @property
    def instance_spec(self) -> TencentProviderInstanceSpec:
        return TencentProviderInstanceSpec(
            instance_type=self.instance_type,
            physical_cores=self.physical_cores,
            memory_gb=self.memory_gb,
        )


class TencentCvmDryRunClient(Protocol):
    """The only admitted live CVM operation."""

    def describe_instance_type(
        self,
        *,
        region: str,
        zone: str,
        instance_type: str,
    ) -> Mapping[str, object]: ...

    def run_instances(
        self,
        *,
        region: str,
        request: dict[str, object],
    ) -> Mapping[str, object]: ...


def create_tencent_provisioning_spec() -> dict[str, object]:
    """Return the frozen offline resource contract without a region default."""

    return {
        "schema_version": "txnopt-tencent-provisioning-spec-v1",
        "provider": "tencent-cloud",
        "minimum_physical_cores": 64,
        "minimum_provider_memory_gb": 128,
        "core_count": 64,
        "thread_per_core": 1,
        "region": None,
        "region_required_live_input": True,
        "exclusive_linux_required": True,
        "maximum_consecutive_window_days": 14,
        "predicted_completion_days_max": 10,
        "cloud_purchase_authorized": False,
        "formal_matrix_started": False,
    }


def build_cvm_dry_run_payload(selection: TencentCvmSelection) -> dict[str, object]:
    """Build the only admitted RunInstances payload: a non-mutating dry run."""

    selection_digest = hashlib.sha256(
        canonical_json_bytes(asdict(selection))
    ).hexdigest()
    return {
        "Placement": {"Zone": selection.zone},
        "ImageId": selection.image_id,
        "InstanceType": selection.instance_type,
        "InstanceChargeType": "POSTPAID_BY_HOUR",
        "InstanceCount": 1,
        "ClientToken": f"txnopt-{selection_digest[:32]}",
        "VirtualPrivateCloud": {
            "VpcId": selection.vpc_id,
            "SubnetId": selection.subnet_id,
        },
        "SecurityGroupIds": [selection.security_group_id],
        "InternetAccessible": {
            "PublicIpAssigned": False,
            "InternetMaxBandwidthOut": 0,
        },
        "CpuTopology": {"CoreCount": 64, "ThreadPerCore": 1},
        "DryRun": True,
        "DisableApiTermination": True,
    }


def prepare_cvm_dry_run_envelope(selection: TencentCvmSelection) -> dict[str, object]:
    """Bind the required SDK region to a dry-run-only request body."""

    return {
        "schema_version": "txnopt-tencent-cvm-dry-run-request-v1",
        "provider": "tencent-cloud",
        "region": selection.region,
        "provider_instance_spec_source": "live_describe_zone_instance_config_infos",
        "request": build_cvm_dry_run_payload(selection),
        "submitted": False,
        "cloud_purchase_authorized": False,
    }


def execute_cvm_dry_run(
    selection: TencentCvmSelection,
    *,
    client: TencentCvmDryRunClient | None = None,
) -> dict[str, object]:
    """Submit exactly one safe RunInstances preflight and never create an instance."""

    payload = build_cvm_dry_run_payload(selection)
    _validate_dry_run_payload(payload)
    bridge = client or _TencentCvmSdkBridge.from_environment_or_cvm_role()
    described = bridge.describe_instance_type(
        region=selection.region,
        zone=selection.zone,
        instance_type=selection.instance_type,
    )
    provider_instance = _validated_provider_instance(
        described,
        selection=selection,
    )
    response = bridge.run_instances(region=selection.region, request=payload)
    request_id = response.get("RequestId")
    if not isinstance(request_id, str) or not request_id:
        raise RuntimeError("Tencent CVM DryRun response lacks its RequestId")
    instance_ids = response.get("InstanceIdSet")
    if instance_ids not in (None, []):
        raise RuntimeError("Tencent CVM DryRun unexpectedly reported an instance id")
    return {
        "schema_version": "txnopt-tencent-cvm-dry-run-receipt-v1",
        "status": "DRY_RUN_PASS_NO_INSTANCE_CREATED",
        "provider": "tencent-cloud",
        "region": selection.region,
        "instance_type": selection.instance_type,
        "provider_instance": provider_instance,
        "request_id": request_id,
        "dry_run": True,
        "instance_created": False,
        "cloud_purchase_authorized": False,
    }


def _validate_dry_run_payload(payload: Mapping[str, object]) -> None:
    if payload.get("DryRun") is not True:
        raise ValueError("Tencent CVM boundary requires DryRun=true")
    topology = payload.get("CpuTopology")
    if topology != {"CoreCount": 64, "ThreadPerCore": 1}:
        raise ValueError("Tencent CVM boundary requires CpuTopology 64 physical cores/1 thread")
    required = (
        "Placement",
        "ImageId",
        "InstanceType",
        "VirtualPrivateCloud",
        "SecurityGroupIds",
    )
    if any(not payload.get(field) for field in required):
        raise ValueError("Tencent CVM DryRun request lacks a required live selection")


class _TencentCvmSdkBridge:
    """Pinned Tencent SDK bridge using environment credentials or a CVM CAM role."""

    def __init__(self, credential: object) -> None:
        self._credential = credential

    @classmethod
    def from_environment_or_cvm_role(cls) -> _TencentCvmSdkBridge:
        try:
            if importlib.metadata.version("tencentcloud-sdk-python-common") != "3.1.156":
                raise RuntimeError("Tencent common SDK version differs from 3.1.156")
            if importlib.metadata.version("tencentcloud-sdk-python-cvm") != "3.1.156":
                raise RuntimeError("Tencent CVM SDK version differs from 3.1.156")
            credential_module = importlib.import_module("tencentcloud.common.credential")
        except (ImportError, importlib.metadata.PackageNotFoundError) as error:
            raise RuntimeError("pinned Tencent CVM/Common SDKs are not installed") from error
        secret_id = os.environ.get("TENCENTCLOUD_SECRET_ID")
        secret_key = os.environ.get("TENCENTCLOUD_SECRET_KEY")
        token = os.environ.get("TENCENTCLOUD_SESSION_TOKEN")
        dynamic = cast(Any, credential_module)
        if secret_id and secret_key:
            operation = "environment credential construction"
        elif secret_id or secret_key or token:
            raise RuntimeError("Tencent credential environment is incomplete")
        else:
            operation = "CVM CAM role credential acquisition"
        credential: object | None = None
        credential_error: RuntimeError | None = None
        try:
            if secret_id and secret_key:
                credential = dynamic.Credential(secret_id, secret_key, token)
            else:
                credential = dynamic.CVMRoleCredential().get_credential()
        except Exception as error:
            credential_error = RuntimeError(
                _redacted_sdk_error(error, operation=operation)
            )
        if credential_error is not None:
            raise credential_error
        if credential is None:
            raise RuntimeError("Tencent environment credentials and CVM CAM role are absent")
        return cls(credential)

    def run_instances(
        self,
        *,
        region: str,
        request: dict[str, object],
    ) -> Mapping[str, object]:
        _validate_dry_run_payload(request)
        payload: object = None
        request_error: RuntimeError | None = None
        try:
            cvm_client = cast(
                Any,
                importlib.import_module("tencentcloud.cvm.v20170312.cvm_client"),
            )
            models = cast(
                Any,
                importlib.import_module("tencentcloud.cvm.v20170312.models"),
            )
            sdk_request = models.RunInstancesRequest()
            sdk_request.from_json_string(json.dumps(request, sort_keys=True))
            response = cvm_client.CvmClient(self._credential, region).RunInstances(
                sdk_request
            )
            payload = json.loads(response.to_json_string())
        except Exception as error:
            request_error = RuntimeError(
                _redacted_sdk_error(error, operation="RunInstances DryRun")
            )
        if request_error is not None:
            raise request_error
        if not isinstance(payload, dict):
            raise RuntimeError("Tencent CVM DryRun response is not a JSON object")
        return cast(dict[str, object], payload)

    def describe_instance_type(
        self,
        *,
        region: str,
        zone: str,
        instance_type: str,
    ) -> Mapping[str, object]:
        payload: object = None
        request_error: RuntimeError | None = None
        try:
            cvm_client = cast(
                Any,
                importlib.import_module("tencentcloud.cvm.v20170312.cvm_client"),
            )
            models = cast(
                Any,
                importlib.import_module("tencentcloud.cvm.v20170312.models"),
            )
            sdk_request = models.DescribeZoneInstanceConfigInfosRequest()
            sdk_request.from_json_string(
                json.dumps(
                    {
                        "Filters": [
                            {"Name": "zone", "Values": [zone]},
                            {"Name": "instance-type", "Values": [instance_type]},
                        ]
                    },
                    sort_keys=True,
                )
            )
            response = cvm_client.CvmClient(
                self._credential,
                region,
            ).DescribeZoneInstanceConfigInfos(sdk_request)
            payload = json.loads(response.to_json_string())
        except Exception as error:
            request_error = RuntimeError(
                _redacted_sdk_error(
                    error,
                    operation="DescribeZoneInstanceConfigInfos",
                )
            )
        if request_error is not None:
            raise request_error
        if not isinstance(payload, dict):
            raise RuntimeError("Tencent CVM instance-config response is not a JSON object")
        items = payload.get("InstanceTypeQuotaSet")
        if not isinstance(items, list) or len(items) != 1:
            raise RuntimeError(
                "Tencent CVM instance-config query did not return exactly one SKU"
            )
        item = items[0]
        if not isinstance(item, dict):
            raise RuntimeError("Tencent CVM instance-config item is not a JSON object")
        normalized = dict(item)
        normalized["RequestId"] = payload.get("RequestId")
        return normalized


def _validated_provider_instance(
    payload: Mapping[str, object],
    *,
    selection: TencentCvmSelection,
) -> dict[str, object]:
    request_id = payload.get("RequestId")
    if not isinstance(request_id, str) or not request_id:
        raise RuntimeError("Tencent CVM instance-config response lacks its RequestId")
    zone = payload.get("Zone")
    if zone != selection.zone:
        raise RuntimeError("Tencent CVM instance-config zone differs from the request")
    instance_type = payload.get("InstanceType")
    if instance_type != selection.instance_type:
        raise RuntimeError("Tencent CVM instance type differs from the request")
    memory = payload.get("Memory")
    if isinstance(memory, bool) or not isinstance(memory, (int, float)):
        raise RuntimeError("Tencent CVM instance-config memory is not numeric")
    if memory < 128:
        raise RuntimeError("Tencent CVM instance type must provide at least 128 GB")
    cpu = payload.get("Cpu")
    if isinstance(cpu, bool) or not isinstance(cpu, int) or cpu <= 0:
        raise RuntimeError("Tencent CVM instance-config vCPU count is invalid")
    status = payload.get("Status")
    if status != "SELL":
        raise RuntimeError("Tencent CVM instance type is not currently sellable")
    return {
        "describe_request_id": request_id,
        "zone": zone,
        "instance_type": instance_type,
        "vcpu": cpu,
        "memory_gb": memory,
        "status": status,
        "requested_physical_cores": 64,
        "requested_thread_per_core": 1,
    }


def _redacted_sdk_error(
    error: Exception,
    *,
    operation: str = "request",
) -> str:
    return safe_sdk_error_text(
        provider="Tencent CVM",
        operation=operation,
        error=error,
    )


def assess_tencent_capacity(
    protocol: Mapping[str, Any],
    calibration: Mapping[str, Any],
    *,
    physical_cores: int = 64,
    provider_memory_gb: int = 128,
    scheduler_efficiency: float = 0.8,
) -> dict[str, object]:
    """Recompute the formal-matrix estimate for an exact physical-core profile."""

    requirements = TencentCloudRequirements(
        minimum_physical_cores=64,
        minimum_provider_memory_gb=128,
        scheduler_efficiency=scheduler_efficiency,
    )
    if physical_cores < requirements.minimum_physical_cores:
        raise ValueError("Tencent profile has fewer than 64 physical cores")
    if provider_memory_gb < requirements.minimum_provider_memory_gb:
        raise ValueError("Tencent profile has less than 128 GB provider memory")
    estimate = estimate_cloud_window(
        protocol,
        calibration,
        physical_cores=physical_cores,
        scheduler_efficiency=scheduler_efficiency,
    )
    predicted_days = estimate.get("predicted_matrix_days")
    if not isinstance(predicted_days, float):
        raise ValueError("cloud estimate did not produce a day count")
    capacity_pass = predicted_days <= requirements.maximum_matrix_days
    protocol_v2 = protocol.get("schema_version") == "txnopt-level1-protocol-v2"
    peak_rss_bytes: int | None = None
    calibration_attempt: int | None = None
    if protocol_v2:
        if calibration.get("schema_version") != "txnopt-local-runtime-calibration-v2":
            raise ValueError("protocol v2 requires the v2 calibration schema")
        candidate = calibration.get("peak_rss_bytes")
        if isinstance(candidate, bool) or not isinstance(candidate, int) or candidate <= 0:
            raise ValueError("calibration peak RSS is missing")
        raw_attempt = calibration.get("attempt")
        if (
            isinstance(raw_attempt, bool)
            or not isinstance(raw_attempt, int)
            or raw_attempt <= 0
        ):
            raise ValueError("calibration attempt identity is missing")
        peak_rss_bytes = candidate
        calibration_attempt = raw_attempt
    return {
        "schema_version": "txnopt-tencent-capacity-assessment-v2",
        "status": (
            "CAPACITY_ESTIMATE_PASS_LIVE_HOST_NOT_VERIFIED"
            if capacity_pass
            else "CAPACITY_ESTIMATE_FAIL"
        ),
        "provider": "tencent-cloud",
        "physical_cores": physical_cores,
        "provider_memory_gb": provider_memory_gb,
        "physical_core_requirement_satisfied": True,
        "memory_requirement_satisfied": True,
        "attempt27_peak_rss_bytes": (
            peak_rss_bytes if calibration_attempt == 27 else None
        ),
        "calibration_attempt": calibration_attempt,
        "calibration_peak_rss_bytes": peak_rss_bytes,
        "memory_margin_live_host_verified": False,
        "usable_core_tokens": estimate["usable_cores"],
        "scheduler_efficiency": scheduler_efficiency,
        "predicted_matrix_seconds": estimate["predicted_matrix_seconds"],
        "predicted_matrix_days": predicted_days,
        "maximum_matrix_days": requirements.maximum_matrix_days,
        "capacity_estimate_pass": capacity_pass,
        "requires_cvm_api_core_count_at_least": 64,
        "requires_linux_physical_core_count_at_least": 64,
        "accepts_vcpu_count_as_physical_core_evidence": False,
        "build_portability_verified": False,
        "live_host_verified": False,
        "cloud_account_configured": _credential_environment_present(),
        "cloud_purchase_performed": False,
        "formal_matrix_started": False,
    }


def check_tencent_host(
    *,
    provider_instance: TencentProviderInstanceSpec,
    minimum_physical_cores: int = 64,
    minimum_provider_memory_gb: int = 128,
    expected_peak_rss_bytes: int | None = None,
    work_directory: Path | None = None,
) -> dict[str, object]:
    """Enforce the Tencent resource contract after producing an observation."""

    receipt = inspect_tencent_host(
        provider_instance=provider_instance,
        minimum_physical_cores=minimum_physical_cores,
        minimum_provider_memory_gb=minimum_provider_memory_gb,
        expected_peak_rss_bytes=expected_peak_rss_bytes,
        work_directory=work_directory,
    )
    if receipt["capacity_pass"] is True:
        return receipt
    linux_topology = cast(Mapping[str, object], receipt["linux_topology"])
    observed_cores = cast(int, linux_topology["physical_cores"])
    if provider_instance.memory_gb < minimum_provider_memory_gb:
        raise RuntimeError(
            f"Tencent SKU declares {provider_instance.memory_gb} GB memory; "
            f"{minimum_provider_memory_gb} GB are required"
        )
    if provider_instance.physical_cores < minimum_physical_cores:
        raise RuntimeError(
            f"Tencent API reports {provider_instance.physical_cores} physical cores; "
            f"{minimum_physical_cores} are required"
        )
    if observed_cores < minimum_physical_cores:
        raise RuntimeError(
            f"Tencent host has {observed_cores} physical cores; "
            f"{minimum_physical_cores} are required"
        )
    raise RuntimeError("Tencent host does not satisfy the recorded resource contract")


def inspect_tencent_host_from_provider_receipt(
    provider_receipt_path: Path,
    *,
    expected_peak_rss_bytes: int | None = None,
    work_directory: Path | None = None,
) -> dict[str, object]:
    """Cross-check Linux topology against one signed live Tencent API receipt."""

    provider_receipt_digest = verify_sidecar(provider_receipt_path)
    provider_receipt = read_signed_json(provider_receipt_path)
    evidence = _validated_provider_api_evidence(provider_receipt)
    receipt = inspect_tencent_host(
        provider_instance=evidence.instance_spec,
        minimum_physical_cores=64,
        minimum_provider_memory_gb=128,
        expected_peak_rss_bytes=expected_peak_rss_bytes,
        work_directory=work_directory,
    )
    failures = receipt.get("failures")
    if not isinstance(failures, list):
        raise RuntimeError("Tencent host receipt lacks its failure ledger")
    receipt["schema_version"] = "txnopt-tencent-host-check-v2"
    receipt["provider_resource_input_source"] = "signed_live_cvm_dry_run_receipt"
    receipt["provider_receipt_path"] = str(
        provider_receipt_path.expanduser().resolve(strict=True)
    )
    receipt["provider_receipt_sha256"] = provider_receipt_digest
    receipt["provider_api_evidence"] = asdict(evidence)
    receipt["provider_linux_cross_check_pass"] = not any(
        failure
        in {
            "provider_physical_cores_below_64",
            "linux_physical_cores_below_64",
            "provider_linux_physical_core_cross_check_failed",
            "thread_per_core_is_not_one",
        }
        for failure in failures
    )
    return receipt


def _validated_provider_api_evidence(
    receipt: Mapping[str, Any],
) -> TencentProviderApiEvidence:
    expected_fields = {
        "schema_version",
        "status",
        "provider",
        "region",
        "instance_type",
        "provider_instance",
        "request_id",
        "dry_run",
        "instance_created",
        "cloud_purchase_authorized",
    }
    if set(receipt) != expected_fields:
        raise ValueError("Tencent provider receipt field set differs")
    if (
        receipt.get("schema_version") != "txnopt-tencent-cvm-dry-run-receipt-v1"
        or receipt.get("status") != "DRY_RUN_PASS_NO_INSTANCE_CREATED"
        or receipt.get("provider") != "tencent-cloud"
        or receipt.get("dry_run") is not True
        or receipt.get("instance_created") is not False
        or receipt.get("cloud_purchase_authorized") is not False
    ):
        raise ValueError("Tencent provider receipt is not a successful DryRun")
    region = receipt.get("region")
    if not isinstance(region, str) or not region or region != region.strip():
        raise ValueError("Tencent provider receipt region is missing")
    instance_type = receipt.get("instance_type")
    if (
        not isinstance(instance_type, str)
        or not instance_type
        or instance_type != instance_type.strip()
    ):
        raise ValueError("Tencent provider receipt instance type is missing")
    dry_run_request_id = receipt.get("request_id")
    if not isinstance(dry_run_request_id, str) or not dry_run_request_id:
        raise ValueError("Tencent provider receipt request identity is missing")
    provider_instance = receipt.get("provider_instance")
    expected_instance_fields = {
        "describe_request_id",
        "zone",
        "instance_type",
        "vcpu",
        "memory_gb",
        "status",
        "requested_physical_cores",
        "requested_thread_per_core",
    }
    if (
        not isinstance(provider_instance, dict)
        or set(provider_instance) != expected_instance_fields
    ):
        raise ValueError("Tencent provider instance receipt field set differs")
    describe_request_id = provider_instance.get("describe_request_id")
    if not isinstance(describe_request_id, str) or not describe_request_id:
        raise ValueError("Tencent provider describe request identity is missing")
    zone = provider_instance.get("zone")
    if not isinstance(zone, str) or not zone or zone != zone.strip():
        raise ValueError("Tencent provider receipt zone is missing")
    if provider_instance.get("instance_type") != instance_type:
        raise ValueError("Tencent provider receipt instance type differs")
    physical_cores = provider_instance.get("requested_physical_cores")
    thread_per_core = provider_instance.get("requested_thread_per_core")
    vcpu = provider_instance.get("vcpu")
    memory_gb = provider_instance.get("memory_gb")
    if physical_cores != 64 or thread_per_core != 1 or vcpu != 64:
        raise ValueError(
            "Tencent provider DryRun must acknowledge 64 physical cores and one thread per core"
        )
    if isinstance(memory_gb, bool) or not isinstance(memory_gb, (int, float)):
        raise ValueError("Tencent provider receipt memory is not numeric")
    if memory_gb < 128:
        raise ValueError("Tencent provider receipt memory is below 128 GB")
    if provider_instance.get("status") != "SELL":
        raise ValueError("Tencent provider receipt SKU is not sellable")
    return TencentProviderApiEvidence(
        region=region,
        zone=zone,
        instance_type=instance_type,
        physical_cores=physical_cores,
        thread_per_core=thread_per_core,
        vcpu=vcpu,
        memory_gb=int(memory_gb),
        describe_request_id=describe_request_id,
        dry_run_request_id=dry_run_request_id,
    )


def inspect_tencent_host(
    *,
    provider_instance: TencentProviderInstanceSpec,
    minimum_physical_cores: int = 64,
    minimum_provider_memory_gb: int = 128,
    expected_peak_rss_bytes: int | None = None,
    work_directory: Path | None = None,
) -> dict[str, object]:
    """Collect a complete host receipt before any resource decision is enforced."""

    observed_cores = physical_core_count()
    logical_processors = _logical_processor_count()
    topology_pairs = _cpu_topology_pairs()
    observed_memory_bytes = memory_bytes()
    observed_memory_gib = observed_memory_bytes / 1024**3
    target_directory = (work_directory or Path.cwd()).expanduser().absolute()
    active = _active_host_observations(target_directory)
    failures: list[str] = []
    if provider_instance.physical_cores < minimum_physical_cores:
        failures.append("provider_physical_cores_below_64")
    if provider_instance.memory_gb < minimum_provider_memory_gb:
        failures.append("provider_memory_below_128_gb")
    if observed_cores < minimum_physical_cores:
        failures.append("linux_physical_cores_below_64")
    if provider_instance.physical_cores != observed_cores:
        failures.append("provider_linux_physical_core_cross_check_failed")
    if logical_processors != observed_cores:
        failures.append("thread_per_core_is_not_one")
    if expected_peak_rss_bytes is not None:
        if expected_peak_rss_bytes < 0:
            raise ValueError("expected peak RSS must be non-negative")
        if expected_peak_rss_bytes > observed_memory_bytes * 0.8:
            failures.append("calibration_peak_rss_exceeds_visible_memory_margin")
    for field in (
        "writer_count",
        "lease_count",
        "process_count",
        "socket_count",
        "shared_memory_owner_count",
    ):
        if cast(int, active[field]) > 0:
            failures.append(f"active_{field.removesuffix('_count')}_detected")
    usable_tokens = max(1, int(observed_cores * 0.8))
    return {
        "schema_version": "txnopt-tencent-host-check-v1",
        "status": (
            "HOST_CAPACITY_PASS_EXCLUSIVITY_NOT_AUTHORIZED"
            if not failures
            else "HOST_CAPACITY_FAIL"
        ),
        "capacity_pass": not failures,
        "provider": "tencent-cloud",
        "provider_instance": asdict(provider_instance),
        "host": linux_host_identity(
            physical_cores=observed_cores,
            memory_gib=int(observed_memory_gib),
            usable_core_tokens=usable_tokens,
            exclusive_linux=False,
        ),
        "linux_topology": {
            "physical_cores": observed_cores,
            "logical_processors": logical_processors,
            "physical_core_pairs": [list(pair) for pair in topology_pairs],
            "smt_enabled": logical_processors != observed_cores,
            "numa_cpu_map": _numa_topology(),
        },
        "frequency_policy": _frequency_policy(),
        "kernel": {
            "release": platform.release(),
            "machine": platform.machine(),
            "distribution": _os_release(),
        },
        "work_directory": str(target_directory),
        "work_directory_free_bytes": shutil.disk_usage(target_directory).free,
        "active_owners": active,
        "minimum_physical_cores": minimum_physical_cores,
        "minimum_provider_memory_gb": minimum_provider_memory_gb,
        "provider_memory_gb": provider_instance.memory_gb,
        "observed_memory_bytes": observed_memory_bytes,
        "observed_memory_gib": observed_memory_gib,
        "linux_visible_memory_is_admission_gate": False,
        "expected_peak_rss_bytes": expected_peak_rss_bytes,
        "peak_rss_within_visible_memory_fraction": (
            None
            if expected_peak_rss_bytes is None
            else expected_peak_rss_bytes <= observed_memory_bytes * 0.8
        ),
        "vcpu_count_used_as_evidence": False,
        "exclusive_linux_authorized": False,
        "formal_host_gate_pass": False,
        "failures": failures,
        "cloud_purchase_performed": False,
        "formal_matrix_started": False,
    }


def _cpu_topology_pairs(path: Path = Path("/proc/cpuinfo")) -> tuple[tuple[str, str], ...]:
    pairs: set[tuple[str, str]] = set()
    for block in path.read_text(encoding="utf-8").split("\n\n"):
        fields = _colon_fields(block)
        physical_id = fields.get("physical id")
        core_id = fields.get("core id")
        if physical_id is not None and core_id is not None:
            pairs.add((physical_id, core_id))
    if not pairs:
        raise RuntimeError("Linux physical-core topology is unavailable")
    return tuple(sorted(pairs))


def _logical_processor_count(path: Path = Path("/proc/cpuinfo")) -> int:
    count = sum(
        1
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith("processor") and ":" in line
    )
    if count <= 0:
        raise RuntimeError("Linux logical-processor topology is unavailable")
    return count


def _colon_fields(block: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in block.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            fields[key.strip()] = value.strip()
    return fields


def _numa_topology(root: Path = Path("/sys/devices/system/node")) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for node in sorted(root.glob("node[0-9]*")):
        cpulist = node / "cpulist"
        if cpulist.is_file():
            mapping[node.name] = cpulist.read_text(encoding="utf-8").strip()
    if not mapping:
        raise RuntimeError("Linux NUMA topology is unavailable")
    return mapping


def _frequency_policy(root: Path = Path("/sys/devices/system/cpu/cpufreq")) -> dict[str, object]:
    governors: set[str] = set()
    drivers: set[str] = set()
    for policy in sorted(root.glob("policy[0-9]*")):
        governor = policy / "scaling_governor"
        driver = policy / "scaling_driver"
        if governor.is_file():
            governors.add(governor.read_text(encoding="utf-8").strip())
        if driver.is_file():
            drivers.add(driver.read_text(encoding="utf-8").strip())
    return {
        "governors": sorted(governors),
        "drivers": sorted(drivers),
        "available": bool(governors or drivers),
    }


def _os_release(path: Path = Path("/etc/os-release")) -> dict[str, str]:
    if not path.is_file():
        raise RuntimeError("Linux distribution identity is unavailable")
    payload: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.startswith("#"):
            key, value = line.split("=", 1)
            payload[key] = value.strip().strip('"')
    return payload


def _active_host_observations(work_directory: Path) -> dict[str, object]:
    target = str(work_directory)
    ignored_pids = _current_process_ancestry()
    process_records: list[dict[str, object]] = []
    writer_pids: set[int] = set()
    socket_pids: set[int] = set()
    shared_memory_pids: set[int] = set()
    open_paths: dict[int, list[str]] = {}
    for process_root in Path("/proc").iterdir():
        if not process_root.name.isdigit() or int(process_root.name) in ignored_pids:
            continue
        pid = int(process_root.name)
        try:
            command = (
                (process_root / "cmdline").read_bytes().replace(b"\0", b" ").decode().strip()
            )
            descriptors = tuple((process_root / "fd").iterdir())
        except (FileNotFoundError, PermissionError, ProcessLookupError, UnicodeDecodeError):
            continue
        relevant_paths: list[str] = []
        has_socket = False
        for descriptor in descriptors:
            try:
                linked = os.readlink(descriptor)
            except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
                continue
            if linked.startswith("socket:["):
                has_socket = True
            if linked == target or linked.startswith(target + os.sep):
                relevant_paths.append(linked)
        try:
            mappings = (process_root / "maps").read_text(
                encoding="utf-8",
                errors="replace",
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            mappings = ""
        is_txnopt_process = "txnopt" in command.lower() or target in command
        if relevant_paths:
            writer_pids.add(pid)
            open_paths[pid] = sorted(set(relevant_paths))
        if is_txnopt_process:
            process_records.append({"pid": pid, "command": command})
            if has_socket:
                socket_pids.add(pid)
            if "/dev/shm" in mappings or "SYSV" in mappings:
                shared_memory_pids.add(pid)
    lease_paths = tuple(
        sorted(
            {
                path
                for paths in open_paths.values()
                for path in paths
                if Path(path).name == "writer.lock"
                or Path(path).name.endswith((".lock", ".lease"))
            }
        )
    )
    return {
        "writer_count": len(writer_pids),
        "writer_pids": sorted(writer_pids),
        "open_paths": {str(pid): paths for pid, paths in sorted(open_paths.items())},
        "lease_count": len(lease_paths),
        "lease_paths": list(lease_paths),
        "process_count": len(process_records),
        "processes": sorted(
            process_records,
            key=lambda item: cast(int, item["pid"]),
        ),
        "socket_count": len(socket_pids),
        "socket_owner_pids": sorted(socket_pids),
        "shared_memory_owner_count": len(shared_memory_pids),
        "shared_memory_owner_pids": sorted(shared_memory_pids),
    }


def _current_process_ancestry() -> set[int]:
    ancestry: set[int] = set()
    pid = os.getpid()
    while pid > 1 and pid not in ancestry:
        ancestry.add(pid)
        try:
            lines = Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            break
        parent = next((line for line in lines if line.startswith("PPid:")), None)
        if parent is None:
            break
        pid = int(parent.split()[1])
    return ancestry


def memory_bytes(path: Path = Path("/proc/meminfo")) -> int:
    """Return Linux MemTotal as observed bytes without rounding it into a gate."""

    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("MemTotal:"):
            kib = int(line.split()[1])
            return kib * 1024
    raise RuntimeError("host memory identity is unavailable")


def _credential_environment_present() -> bool:
    return bool(
        os.environ.get("TENCENTCLOUD_SECRET_ID")
        and os.environ.get("TENCENTCLOUD_SECRET_KEY")
    )


__all__ = [
    "TencentCvmSelection",
    "TencentCloudRequirements",
    "TencentProviderInstanceSpec",
    "TencentProviderApiEvidence",
    "assess_tencent_capacity",
    "build_cvm_dry_run_payload",
    "check_tencent_host",
    "create_tencent_provisioning_spec",
    "execute_cvm_dry_run",
    "inspect_tencent_host",
    "inspect_tencent_host_from_provider_receipt",
    "memory_bytes",
    "prepare_cvm_dry_run_envelope",
]
