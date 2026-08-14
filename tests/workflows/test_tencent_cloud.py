from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import txnopt_evidence.tencent_cloud as tencent_cloud
from txnopt_evidence.cli import main
from txnopt_evidence.codec import read_signed_json, write_signed_json
from txnopt_evidence.tencent_cloud import (
    TencentCvmSelection,
    TencentProviderInstanceSpec,
    assess_tencent_capacity,
    build_cvm_dry_run_payload,
    check_tencent_host,
    create_tencent_provisioning_spec,
    execute_cvm_dry_run,
    inspect_tencent_host,
)

_AXES = ["serial_1", "txnopt_1", "txnopt_4", "barrier_4"]
_BUDGETS = ["fixed_work", "fixed_time"]
ROOT = Path(__file__).resolve().parents[2]


def test_host_activity_scan_excludes_doctor_process_ancestry() -> None:
    ancestry = tencent_cloud._current_process_ancestry()

    assert os.getpid() in ancestry
    assert all(pid > 1 for pid in ancestry)


def _protocol() -> dict[str, object]:
    return {
        "schema_version": "txnopt-level1-protocol-v1",
        "minimum_physical_cores": 32,
        "predicted_formal_matrix_days_max": 10,
        "cloud_window_days": 14,
        "formal_axes": _AXES,
        "budgets": _BUDGETS,
        "seeds": [2014, 2015],
        "domains": {
            "evrptw": {"pilot": ["c101"], "validation": ["r101"]},
            "rcpsp": {"pilot": ["j1201_1"], "validation": ["j1201_2"]},
        },
    }


def _calibration(
    *,
    schema_version: str = "txnopt-local-runtime-calibration-v1",
) -> dict[str, object]:
    return {
        "schema_version": schema_version,
        **({"peak_rss_bytes": 64 * 1024**2} if schema_version.endswith("v2") else {}),
        "observations": [
            {
                "domain": domain,
                "axis": axis,
                "budget": budget,
                "seconds_per_run_p95": 2.0,
            }
            for domain in ("evrptw", "rcpsp")
            for axis in _AXES
            for budget in _BUDGETS
        ],
    }


def _mock_linux_host(
    monkeypatch: pytest.MonkeyPatch,
    *,
    physical_cores: int,
    logical_processors: int,
    visible_memory_gib: int,
) -> None:
    monkeypatch.setattr(tencent_cloud, "physical_core_count", lambda: physical_cores)
    monkeypatch.setattr(
        tencent_cloud,
        "memory_bytes",
        lambda: visible_memory_gib * 1024**3,
    )
    monkeypatch.setattr(
        tencent_cloud,
        "_logical_processor_count",
        lambda: logical_processors,
    )
    monkeypatch.setattr(
        tencent_cloud,
        "_cpu_topology_pairs",
        lambda: tuple(("0", str(core)) for core in range(physical_cores)),
    )
    monkeypatch.setattr(tencent_cloud, "_numa_topology", lambda: {"node0": "0-63"})
    monkeypatch.setattr(
        tencent_cloud,
        "_frequency_policy",
        lambda: {"governors": ["performance"]},
    )
    monkeypatch.setattr(
        tencent_cloud,
        "_active_host_observations",
        lambda _: {
            "writer_count": 0,
            "lease_count": 0,
            "process_count": 0,
            "socket_count": 0,
            "shared_memory_owner_count": 0,
        },
    )
    monkeypatch.setattr(
        tencent_cloud,
        "linux_host_identity",
        lambda **kwargs: {"physical_cores": kwargs["physical_cores"]},
    )


def test_tencent_capacity_requires_64_real_cores_and_128_provider_gb() -> None:
    assessment = assess_tencent_capacity(
        _protocol(),
        _calibration(),
        physical_cores=64,
        provider_memory_gb=128,
    )

    assert assessment["status"] == "CAPACITY_ESTIMATE_PASS_LIVE_HOST_NOT_VERIFIED"
    assert assessment["usable_core_tokens"] == 51
    assert assessment["accepts_vcpu_count_as_physical_core_evidence"] is False
    assert assessment["provider_memory_gb"] == 128
    assert assessment["live_host_verified"] is False
    assert assessment["cloud_purchase_performed"] is False

    with pytest.raises(ValueError, match="64 physical"):
        assess_tencent_capacity(
            _protocol(),
            _calibration(),
            physical_cores=63,
            provider_memory_gb=128,
        )


def test_tencent_capacity_accepts_protocol_v2_resource_contract() -> None:
    protocol = json.loads(
        (ROOT / "experiments/txnopt/level1-protocol-v2.json").read_bytes()
    )

    assessment = assess_tencent_capacity(
        protocol,
        _calibration(schema_version="txnopt-local-runtime-calibration-v2"),
        physical_cores=64,
        provider_memory_gb=128,
    )

    assert assessment["schema_version"] == "txnopt-tencent-capacity-assessment-v2"
    assert assessment["usable_core_tokens"] == 51
    assert assessment["attempt27_peak_rss_bytes"] == 64 * 1024**2
    assert assessment["memory_margin_live_host_verified"] is False
    with pytest.raises(ValueError, match="128 GB"):
        assess_tencent_capacity(
            _protocol(),
            _calibration(),
            physical_cores=64,
            provider_memory_gb=127,
        )


def test_tencent_host_check_uses_physical_topology_not_vcpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_linux_host(
        monkeypatch,
        physical_cores=64,
        logical_processors=64,
        visible_memory_gib=120,
    )

    receipt = check_tencent_host(
        provider_instance=TencentProviderInstanceSpec(
            instance_type="TEST.64CORE128GB",
            physical_cores=64,
            memory_gb=128,
        )
    )

    assert receipt["status"] == "HOST_CAPACITY_PASS_EXCLUSIVITY_NOT_AUTHORIZED"
    assert receipt["vcpu_count_used_as_evidence"] is False
    assert receipt["provider_memory_gb"] == 128
    assert receipt["observed_memory_gib"] == 120.0
    assert receipt["linux_visible_memory_is_admission_gate"] is False
    assert receipt["exclusive_linux_authorized"] is False
    assert receipt["formal_host_gate_pass"] is False


def test_tencent_host_check_rejects_64_vcpu_style_32_core_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_linux_host(
        monkeypatch,
        physical_cores=32,
        logical_processors=64,
        visible_memory_gib=128,
    )

    with pytest.raises(RuntimeError, match="32 physical cores"):
        check_tencent_host(
            provider_instance=TencentProviderInstanceSpec(
                instance_type="TEST.64CORE128GB",
                physical_cores=64,
                memory_gb=128,
            )
        )


def test_tencent_host_accepts_platform_reserved_memory_but_rejects_small_sku(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_linux_host(
        monkeypatch,
        physical_cores=64,
        logical_processors=64,
        visible_memory_gib=119,
    )

    receipt = check_tencent_host(
        provider_instance=TencentProviderInstanceSpec(
            instance_type="TEST.64CORE128GB",
            physical_cores=64,
            memory_gb=128,
        )
    )
    assert receipt["observed_memory_gib"] == 119.0

    with pytest.raises(RuntimeError, match="128 GB"):
        check_tencent_host(
            provider_instance=TencentProviderInstanceSpec(
                instance_type="TEST.64CORE127GB",
                physical_cores=64,
                memory_gb=127,
            )
        )


def test_tencent_host_inspection_records_failure_before_enforcement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tencent_cloud, "physical_core_count", lambda: 32)
    monkeypatch.setattr(tencent_cloud, "memory_bytes", lambda: 120 * 1024**3)
    monkeypatch.setattr(tencent_cloud, "_logical_processor_count", lambda: 64)
    monkeypatch.setattr(tencent_cloud, "_cpu_topology_pairs", lambda: (("0", "0"),) * 32)
    monkeypatch.setattr(tencent_cloud, "_numa_topology", lambda: {"node0": "0-63"})
    monkeypatch.setattr(tencent_cloud, "_frequency_policy", lambda: {"governors": ["performance"]})
    monkeypatch.setattr(tencent_cloud, "_active_host_observations", lambda _: {
        "writer_count": 0,
        "lease_count": 0,
        "process_count": 0,
        "socket_count": 0,
        "shared_memory_owner_count": 0,
    })
    monkeypatch.setattr(
        tencent_cloud,
        "linux_host_identity",
        lambda **kwargs: {"physical_cores": kwargs["physical_cores"]},
    )

    receipt = inspect_tencent_host(
        provider_instance=TencentProviderInstanceSpec(
            instance_type="TEST.64CORE128GB",
            physical_cores=64,
            memory_gb=128,
        ),
        work_directory=tmp_path,
        expected_peak_rss_bytes=1 * 1024**3,
    )

    assert receipt["status"] == "HOST_CAPACITY_FAIL"
    assert receipt["capacity_pass"] is False
    assert receipt["provider_instance"]["physical_cores"] == 64
    assert receipt["linux_topology"]["physical_cores"] == 32
    assert receipt["linux_topology"]["logical_processors"] == 64
    assert receipt["linux_topology"]["smt_enabled"] is True
    assert receipt["observed_memory_gib"] == 120.0
    assert receipt["linux_visible_memory_is_admission_gate"] is False
    assert "linux_physical_cores_below_64" in receipt["failures"]


def test_tencent_provisioning_spec_has_no_region_and_dry_run_is_not_optional() -> None:
    spec = create_tencent_provisioning_spec()

    assert spec["schema_version"] == "txnopt-tencent-provisioning-spec-v1"
    assert spec["core_count"] == 64
    assert spec["thread_per_core"] == 1
    assert spec["minimum_provider_memory_gb"] == 128
    assert spec["region"] is None
    assert spec["region_required_live_input"] is True
    assert spec["cloud_purchase_authorized"] is False

    selection = TencentCvmSelection(
        region="ap-test",
        zone="ap-test-1",
        instance_type="TEST.64CORE128GB",
        image_id="img-test",
        vpc_id="vpc-test",
        subnet_id="subnet-test",
        security_group_id="sg-test",
    )
    request = build_cvm_dry_run_payload(selection)
    assert request["DryRun"] is True
    assert request["CpuTopology"] == {"CoreCount": 64, "ThreadPerCore": 1}
    assert request["Placement"] == {"Zone": "ap-test-1"}
    assert request["InternetAccessible"] == {
        "PublicIpAssigned": False,
        "InternetMaxBandwidthOut": 0,
    }

    with pytest.raises(ValueError, match="region"):
        TencentCvmSelection(
            region="",
            zone="ap-test-1",
            instance_type="TEST.64CORE128GB",
            image_id="img-test",
            vpc_id="vpc-test",
            subnet_id="subnet-test",
            security_group_id="sg-test",
        )


def test_tencent_cvm_sdk_boundary_verifies_provider_spec_before_dry_run() -> None:
    calls: list[tuple[str, str, object]] = []

    class FakeCvmClient:
        def describe_instance_type(
            self,
            *,
            region: str,
            zone: str,
            instance_type: str,
        ) -> dict[str, object]:
            calls.append(("describe", region, (zone, instance_type)))
            return {
                "RequestId": "offline-fixture-describe",
                "Zone": zone,
                "InstanceType": instance_type,
                "Cpu": 64,
                "Memory": 128,
                "Status": "SELL",
            }

        def run_instances(
            self,
            *,
            region: str,
            request: dict[str, object],
        ) -> dict[str, object]:
            calls.append(("dry-run", region, request))
            return {"RequestId": "offline-fixture-request"}

    selection = TencentCvmSelection(
        region="ap-test",
        zone="ap-test-1",
        instance_type="TEST.64CORE128GB",
        image_id="img-test",
        vpc_id="vpc-test",
        subnet_id="subnet-test",
        security_group_id="sg-test",
    )
    receipt = execute_cvm_dry_run(selection, client=FakeCvmClient())

    assert calls[0] == (
        "describe",
        "ap-test",
        ("ap-test-1", "TEST.64CORE128GB"),
    )
    assert calls[1][0] == "dry-run"
    assert calls[1][1] == "ap-test"
    assert isinstance(calls[1][2], dict)
    assert calls[1][2]["DryRun"] is True
    assert calls[1][2]["CpuTopology"] == {"CoreCount": 64, "ThreadPerCore": 1}
    assert receipt == {
        "schema_version": "txnopt-tencent-cvm-dry-run-receipt-v1",
        "status": "DRY_RUN_PASS_NO_INSTANCE_CREATED",
        "provider": "tencent-cloud",
        "region": "ap-test",
        "instance_type": "TEST.64CORE128GB",
        "provider_instance": {
            "describe_request_id": "offline-fixture-describe",
            "zone": "ap-test-1",
            "instance_type": "TEST.64CORE128GB",
            "vcpu": 64,
            "memory_gb": 128,
            "status": "SELL",
            "requested_physical_cores": 64,
            "requested_thread_per_core": 1,
        },
        "request_id": "offline-fixture-request",
        "dry_run": True,
        "instance_created": False,
        "cloud_purchase_authorized": False,
    }

    unsafe = build_cvm_dry_run_payload(selection)
    unsafe["DryRun"] = False
    with pytest.raises(ValueError, match="DryRun=true"):
        tencent_cloud._validate_dry_run_payload(unsafe)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"Memory": 127}, "at least 128 GB"),
        ({"Status": "SOLD_OUT"}, "not currently sellable"),
        ({"Zone": "ap-other-1"}, "zone differs"),
        ({"InstanceType": "TEST.OTHER"}, "instance type differs"),
    ],
)
def test_tencent_cvm_dry_run_rejects_unacceptable_provider_spec(
    overrides: dict[str, object],
    message: str,
) -> None:
    class FakeCvmClient:
        def describe_instance_type(
            self,
            *,
            region: str,
            zone: str,
            instance_type: str,
        ) -> dict[str, object]:
            payload: dict[str, object] = {
                "RequestId": "describe-request",
                "Zone": zone,
                "InstanceType": instance_type,
                "Cpu": 64,
                "Memory": 128,
                "Status": "SELL",
            }
            payload.update(overrides)
            return payload

        def run_instances(
            self,
            *,
            region: str,
            request: dict[str, object],
        ) -> dict[str, object]:
            raise AssertionError("RunInstances must not be called after a bad spec")

    selection = TencentCvmSelection(
        region="ap-test",
        zone="ap-test-1",
        instance_type="TEST.64CORE128GB",
        image_id="img-test",
        vpc_id="vpc-test",
        subnet_id="subnet-test",
        security_group_id="sg-test",
    )
    with pytest.raises(RuntimeError, match=message):
        execute_cvm_dry_run(selection, client=FakeCvmClient())


def test_cli_tencent_assessment_and_offline_archive_inventory(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    protocol = tmp_path / "protocol.json"
    calibration = tmp_path / "calibration.json"
    write_signed_json(protocol, _protocol())
    write_signed_json(calibration, _calibration())

    assert main(
        [
            "cloud",
            "tencent",
            "assess",
            "--protocol",
            str(protocol),
            "--calibration",
            str(calibration),
            "--physical-cores",
            "64",
            "--provider-memory-gb",
            "128",
        ]
    ) == 0
    assessment = json.loads(capsys.readouterr().out)
    assert assessment["physical_cores"] == 64
    assert assessment["provider_memory_gb"] == 128

    source = tmp_path / "source"
    source.mkdir()
    (source / "evidence.json").write_text('{"status":"SEALED"}\n', encoding="utf-8")
    assert main(["archive", "inventory", str(source)]) == 0
    inventory = json.loads(capsys.readouterr().out)
    assert inventory["file_count"] == 1


def test_cli_tencent_spec_and_offline_dry_run_request(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec_path = tmp_path / "tencent-spec.json"
    assert main(
        ["cloud", "tencent", "spec", "--output", str(spec_path)]
    ) == 0
    spec_result = json.loads(capsys.readouterr().out)
    assert spec_result["status"] == "SPEC_MATERIALIZED_NOT_AUTHORIZED"
    assert read_signed_json(spec_path)["region"] is None

    request_path = tmp_path / "dry-run-request.json"
    receipt_path = tmp_path / "dry-run-receipt.json"
    monkeypatch.setattr(
        tencent_cloud,
        "execute_cvm_dry_run",
        lambda _selection: {
            "schema_version": "txnopt-tencent-cvm-dry-run-receipt-v1",
            "status": "DRY_RUN_PASS_NO_INSTANCE_CREATED",
            "request_id": "offline-fixture-request",
            "dry_run": True,
            "instance_created": False,
        },
    )
    assert main(
        [
            "cloud",
            "tencent",
            "dry-run",
            "--region",
            "ap-test",
            "--zone",
            "ap-test-1",
            "--instance-type",
            "TEST.64CORE128GB",
            "--image-id",
            "img-test",
            "--vpc-id",
            "vpc-test",
            "--subnet-id",
            "subnet-test",
            "--security-group-id",
            "sg-test",
            "--request-output",
            str(request_path),
            "--receipt-output",
            str(receipt_path),
        ]
    ) == 0
    request_result = json.loads(capsys.readouterr().out)
    assert request_result["status"] == "DRY_RUN_PASS_NO_INSTANCE_CREATED"
    assert request_result["instance_created"] is False
    request = read_signed_json(request_path)
    assert request["region"] == "ap-test"
    assert request["request"]["DryRun"] is True
    assert request["request"]["CpuTopology"] == {
        "CoreCount": 64,
        "ThreadPerCore": 1,
    }
    assert "SecretId" not in json.dumps(request)
    assert read_signed_json(receipt_path)["request_id"] == "offline-fixture-request"


def test_cli_tencent_cos_fails_closed_without_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TENCENTCLOUD_SECRET_ID", raising=False)
    monkeypatch.delenv("TENCENTCLOUD_SECRET_KEY", raising=False)
    monkeypatch.delenv("TENCENTCLOUD_USE_CVM_ROLE", raising=False)
    source = tmp_path / "source"
    source.mkdir()
    (source / "evidence.json").write_text("{}\n", encoding="utf-8")

    assert main(
        [
            "cloud",
            "tencent",
            "cos",
            "mirror",
            str(source),
            "--cos-bucket",
            "txnopt-evidence-1250000000",
            "--cos-region",
            "ap-guangzhou",
            "--commit-id",
            "offline-not-authorized",
        ]
    ) == 2


def test_cli_generic_archive_storage_commands_are_removed(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()

    with pytest.raises(SystemExit) as removed_command:
        main(
            [
                "archive",
                "mirror",
                str(source),
                "--commit-id",
                "removed-generic-store",
            ]
        )
    assert removed_command.value.code == 2

    with pytest.raises(SystemExit) as removed_s3_flag:
        main(
            [
                "cloud",
                "tencent",
                "cos",
                "mirror",
                str(source),
                "--s3-bucket",
                "removed-s3-store",
                "--commit-id",
                "removed-s3-store",
            ]
        )
    assert removed_s3_flag.value.code == 2
