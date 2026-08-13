from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_level1_case_set_and_holdout_boundary_are_exact() -> None:
    path = ROOT / "experiments/txnopt/level1-protocol.json"
    digest, filename = (
        path.with_suffix(".json.sha256").read_text(encoding="utf-8").strip().split()
    )
    assert filename == path.name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    protocol = json.loads(path.read_bytes())
    domains = protocol["domains"]

    assert len(domains["evrptw"]["pilot"]) == 8
    assert len(domains["evrptw"]["validation"]) == 4
    assert len(domains["rcpsp"]["pilot"]) == 16
    assert len(domains["rcpsp"]["validation"]) == 8
    assert len(protocol["seeds"]) == 10
    assert len(set(protocol["seeds"])) == 10
    assert protocol["formal_axes"] == [
        "serial_1",
        "txnopt_1",
        "txnopt_4",
        "barrier_4",
    ]
    assert protocol["level2_holdout_opened"] is False
    assert protocol["level3_holdout_opened"] is False


def test_internal_build_manifest_is_signed_and_does_not_claim_readiness() -> None:
    path = (
        ROOT
        / "experiments/txnopt/manifests/txnopt_level1_build_attempt01.json"
    )
    digest, filename = (
        path.with_suffix(".json.sha256").read_text(encoding="utf-8").strip().split()
    )
    assert filename == path.name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    manifest = json.loads(path.read_bytes())

    assert manifest["status"] == "BUILD_COMPLETE_NOT_LEVEL1_READY"
    assert manifest["producer"]["source_dirty"] is False
    assert manifest["validation"]["wheel_surface"]["fallback_count"] == 0
    assert manifest["claim_boundary"]["level1_ready"] is False
    assert manifest["claim_boundary"]["cloud_purchase_authorized"] is False
    assert manifest["claim_boundary"]["pending_gates"]


def test_example_catalog_names_exactly_the_preregistered_cases() -> None:
    protocol = json.loads((ROOT / "experiments/txnopt/level1-protocol.json").read_bytes())
    catalog = json.loads(
        (ROOT / "experiments/txnopt/case-catalog.example.json").read_bytes()
    )

    for domain in ("evrptw", "rcpsp"):
        scope = protocol["domains"][domain]
        expected = {*scope["pilot"], *scope["validation"]}
        assert set(catalog["domains"][domain]) == expected


def test_first_resource_soak_failure_is_signed_and_not_promoted() -> None:
    path = (
        ROOT
        / "experiments/txnopt/manifests/txnopt_resource_soak_attempt01_failure.json"
    )
    digest, filename = (
        path.with_suffix(".json.sha256").read_text(encoding="utf-8").strip().split()
    )
    failure = json.loads(path.read_bytes())

    assert filename == path.name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert failure["status"] == "HARNESS_FAILED_BEFORE_RECEIPT"
    assert failure["raw_receipt_written"] is False
    assert failure["disposition"] == "FAILURE_RETAINED_NEW_ATTEMPT_REQUIRED"


def test_second_build_receipt_binds_clean_resource_soak_without_readiness() -> None:
    path = ROOT / "experiments/txnopt/manifests/txnopt_level1_build_attempt02.json"
    digest, filename = (
        path.with_suffix(".json.sha256").read_text(encoding="utf-8").strip().split()
    )
    manifest = json.loads(path.read_bytes())

    assert filename == path.name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert manifest["status"] == (
        "BUILD_AND_LOCAL_RESOURCE_GATES_COMPLETE_NOT_LEVEL1_READY"
    )
    assert manifest["producer"]["source_dirty"] is False
    assert manifest["artifacts"]["resource_soak"]["completed_rounds"] == 100000
    assert manifest["artifacts"]["resource_soak"]["fallback_count"] == 0
    assert manifest["failure_lineage"]["same_attempt_reused"] is False
    assert manifest["claim_boundary"]["level1_ready"] is False


def test_third_build_receipt_binds_campaign_identity_without_readiness() -> None:
    path = ROOT / "experiments/txnopt/manifests/txnopt_level1_build_attempt03.json"
    digest, filename = (
        path.with_suffix(".json.sha256").read_text(encoding="utf-8").strip().split()
    )
    manifest = json.loads(path.read_bytes())

    assert filename == path.name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert manifest["producer"]["source_dirty"] is False
    assert manifest["artifacts"]["wheel"]["entry_count"] == 52
    assert manifest["artifacts"]["resource_soak"]["completed_rounds"] == 100000
    assert manifest["exploratory_calibration"]["status"] == "NOT_FORMAL_EVIDENCE"
    assert manifest["claim_boundary"]["level1_ready"] is False


def test_fourth_build_receipt_binds_deadline_fix_without_readiness() -> None:
    path = ROOT / "experiments/txnopt/manifests/txnopt_level1_build_attempt04.json"
    digest, filename = (
        path.with_suffix(".json.sha256").read_text(encoding="utf-8").strip().split()
    )
    manifest = json.loads(path.read_bytes())

    assert filename == path.name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert manifest["producer"]["source_dirty"] is False
    assert manifest["validation"]["pytest"]["passed"] == 95
    assert manifest["failure_lineage"]["same_attempt_reused"] is False
    assert manifest["claim_boundary"]["level1_ready"] is False


def test_fifth_build_receipt_binds_bounded_screen_cache_without_readiness() -> None:
    path = ROOT / "experiments/txnopt/manifests/txnopt_level1_build_attempt05.json"
    digest, filename = (
        path.with_suffix(".json.sha256").read_text(encoding="utf-8").strip().split()
    )
    manifest = json.loads(path.read_bytes())

    assert filename == path.name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert manifest["producer"]["source_dirty"] is False
    assert manifest["producer"]["development_override"] is False
    assert manifest["artifacts"]["resource_soak"]["completed_rounds"] == 100000
    assert manifest["change"]["capacity"] == 65_536
    assert manifest["change"]["semantic_effect"].startswith("none")
    assert manifest["claim_boundary"]["level1_ready"] is False


def test_local_calibration_records_failed_evrptw_gate_without_procurement() -> None:
    path = (
        ROOT
        / "experiments/txnopt/manifests/txnopt_level1_local_calibration_attempt14.json"
    )
    digest, filename = (
        path.with_suffix(".json.sha256").read_text(encoding="utf-8").strip().split()
    )
    manifest = json.loads(path.read_bytes())

    assert filename == path.name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert manifest["calibration"]["raw_run_count"] == 96
    assert manifest["calibration"]["independent_review_pass_count"] == 96
    assert manifest["fixed_work_physical_results"]["evrptw"]["speedup_gate"] == "FAIL"
    assert manifest["fixed_work_physical_results"]["rcpsp"]["speedup_gate"] == (
        "PASS_REPRESENTATIVE_ONLY"
    )
    assert manifest["cloud_estimate"]["purchase_authorized"] is False
    assert manifest["decision"]["formal_cloud_matrix_authorized"] is False


def test_active_pytest_collection_failure_is_retained_and_not_reused() -> None:
    path = (
        ROOT
        / "experiments/txnopt/manifests/txnopt_active_pytest_attempt01_failure.json"
    )
    digest, filename = (
        path.with_suffix(".json.sha256").read_text(encoding="utf-8").strip().split()
    )
    failure = json.loads(path.read_bytes())

    assert filename == path.name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert failure["status"] == "FAILED_DURING_COLLECTION"
    assert failure["collection_error_count"] == 59
    assert failure["historical_test_files_changed"] is False
    assert failure["same_attempt_reused"] is False


def test_sixth_build_receipt_keeps_failed_gates_open() -> None:
    path = ROOT / "experiments/txnopt/manifests/txnopt_level1_build_attempt06.json"
    digest, filename = (
        path.with_suffix(".json.sha256").read_text(encoding="utf-8").strip().split()
    )
    manifest = json.loads(path.read_bytes())

    assert filename == path.name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert manifest["producer"]["source_dirty"] is False
    assert manifest["validation"]["pytest"]["passed"] == 102
    assert manifest["active_test_boundary"]["historical_test_paths_moved_or_deleted"] is False
    assert manifest["formal_review"]["t3_t4_gate_passed"] is False
    assert manifest["predecessor_calibration"]["binding_status"] == (
        "NOT_REBOUND_TO_BUILD06"
    )
    assert manifest["claim_boundary"]["level1_ready"] is False


def test_seventh_build_receipt_binds_formal_correction_without_approving_it() -> None:
    path = ROOT / "experiments/txnopt/manifests/txnopt_level1_build_attempt07.json"
    digest, filename = (
        path.with_suffix(".json.sha256").read_text(encoding="utf-8").strip().split()
    )
    manifest = json.loads(path.read_bytes())

    assert filename == path.name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert manifest["producer"]["source_dirty"] is False
    assert manifest["validation"]["pytest"]["passed"] == 111
    assert manifest["validation"]["property_based_tests"]["status"] == "PASS"
    assert manifest["formal_correction"]["status"] == "READY_FOR_INDEPENDENT_REVIEW"
    assert manifest["formal_correction"]["independent_review_completed"] is False
    assert manifest["formal_correction"]["t3_t4_gate_passed"] is False
    assert manifest["predecessor_calibration"]["binding_status"] == (
        "NOT_REBOUND_TO_BUILD07"
    )
    assert manifest["claim_boundary"]["level1_ready"] is False


def test_eighth_build_receipt_keeps_the_evrptw_speedup_gate_open() -> None:
    path = ROOT / "experiments/txnopt/manifests/txnopt_level1_build_attempt08.json"
    digest, filename = (
        path.with_suffix(".json.sha256").read_text(encoding="utf-8").strip().split()
    )
    manifest = json.loads(path.read_bytes())

    assert filename == path.name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert manifest["producer"]["source_dirty"] is False
    assert manifest["validation"]["pytest"]["passed"] == 117
    assert manifest["validation"]["route_validation_cache_differential"]["status"] == (
        "PASS"
    )
    assert manifest["performance_change"]["semantic_trace_changed"] is False
    assert manifest["performance_change"]["evrptw_speedup_gate"] == "FAIL"
    assert manifest["predecessor_calibration"]["binding_status"] == (
        "NOT_REBOUND_TO_BUILD08"
    )
    assert manifest["claim_boundary"]["level1_ready"] is False


def test_build08_calibration_is_complete_but_does_not_open_the_cloud_matrix() -> None:
    path = ROOT / (
        "experiments/txnopt/manifests/txnopt_level1_local_calibration_attempt16.json"
    )
    digest, filename = (
        path.with_suffix(".json.sha256").read_text(encoding="utf-8").strip().split()
    )
    manifest = json.loads(path.read_bytes())

    assert filename == path.name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert manifest["producer_build"]["revision"].startswith("ae98bf8")
    assert manifest["calibration"]["raw_run_count"] == 96
    assert manifest["calibration"]["independent_review_pass_count"] == 96
    assert manifest["calibration"]["fallback_count"] == 0
    assert manifest["fixed_work_physical_results"]["evrptw"]["speedup_gate"] == (
        "FAIL"
    )
    assert manifest["fixed_work_physical_results"]["rcpsp"]["speedup_gate"] == (
        "PASS_REPRESENTATIVE_ONLY"
    )
    assert manifest["decision"]["formal_cloud_matrix_authorized"] is False


def test_tenth_build_receipt_keeps_the_successor_formal_gate_open() -> None:
    path = ROOT / "experiments/txnopt/manifests/txnopt_level1_build_attempt10.json"
    digest, filename = (
        path.with_suffix(".json.sha256").read_text(encoding="utf-8").strip().split()
    )
    manifest = json.loads(path.read_bytes())

    assert filename == path.name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert manifest["producer"]["revision"].startswith("3e18b99")
    assert manifest["producer"]["source_dirty"] is False
    assert manifest["validation"]["pytest_wheel_installed"]["passed"] == 174
    assert manifest["validation"]["asan_ubsan"]["status"] == "PASS"
    assert manifest["validation"]["tsan"]["status"] == "PASS"
    assert manifest["formal_successor"]["prior_review_binding_status"] == (
        "PRIOR_SOURCE_ONLY"
    )
    assert manifest["formal_successor"]["successor_status"] == (
        "REVIEW_PENDING_BUILD10"
    )
    assert manifest["claim_boundary"]["cloud_purchase_authorized"] is False
    assert manifest["claim_boundary"]["level1_ready"] is False


def test_build10_calibration_passes_representative_gates_without_authorization() -> None:
    path = ROOT / (
        "experiments/txnopt/manifests/"
        "txnopt_level1_local_calibration_attempt20.json"
    )
    digest, filename = (
        path.with_suffix(".json.sha256").read_text(encoding="utf-8").strip().split()
    )
    manifest = json.loads(path.read_bytes())

    assert filename == path.name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert manifest["producer_build"]["revision"].startswith("3e18b99")
    assert manifest["calibration"]["raw_run_count"] == 96
    assert manifest["calibration"]["raw_manifest_hash_match_count"] == 96
    assert manifest["calibration"]["independent_review_pass_count"] == 96
    assert manifest["calibration"]["independent_review_hash_match_count"] == 96
    assert manifest["fixed_work_physical_results"]["evrptw"]["speedup_gate"] == (
        "PASS_REPRESENTATIVE_ONLY"
    )
    assert manifest["fixed_work_physical_results"]["rcpsp"]["speedup_gate"] == (
        "PASS_REPRESENTATIVE_ONLY"
    )
    assert manifest["formal_successor"]["independent_review_completed"] is False
    assert manifest["cloud_estimate"]["purchase_authorized"] is False
    assert manifest["decision"]["formal_cloud_matrix_authorized"] is False
    assert manifest["decision"]["level1_ready"] is False


def test_build10_precloud_receipt_blocks_procurement_pending_review() -> None:
    path = ROOT / (
        "experiments/txnopt/manifests/txnopt_level1_precloud_gate_attempt02.json"
    )
    digest, filename = (
        path.with_suffix(".json.sha256").read_text(encoding="utf-8").strip().split()
    )
    manifest = json.loads(path.read_bytes())

    assert filename == path.name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert manifest["status"] == "BLOCKED_FORMAL_SUCCESSOR_REVIEW_PENDING"
    assert manifest["formal_plan"]["config_count"] == 2880
    assert manifest["formal_plan"]["raw_output_root_absent"] is True
    assert manifest["analysis_preregistration"]["holdout_opened"] is False
    assert manifest["clean_cli_preflight"]["status"] == (
        "PASS_NOT_AUTHORIZED_TO_EXECUTE"
    )
    assert manifest["decision"]["precloud_gate"] == "BLOCKED"
    assert manifest["execution_boundary"]["procurement_authorized"] is False
    assert manifest["execution_boundary"]["formal_matrix_started"] is False
    assert manifest["execution_boundary"]["level1_ready"] is False


def test_first_build11_fault_gate_failure_is_retained_and_not_promoted() -> None:
    path = ROOT / (
        "experiments/txnopt/manifests/"
        "txnopt_level1_fault_gate_attempt01_failure.json"
    )
    digest, filename = (
        path.with_suffix(".json.sha256").read_text(encoding="utf-8").strip().split()
    )
    manifest = json.loads(path.read_bytes())

    assert filename == path.name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert manifest["status"] == "FAILED_ORCHESTRATION_ENVIRONMENT_IDENTITY"
    assert manifest["retained_raw_evidence"]["fault_tests_started"] == 0
    assert manifest["root_cause"]["producer_runtime_fault"] is False
    assert manifest["claim_boundary"]["reusable_as_pass_evidence"] is False
    assert manifest["claim_boundary"]["cloud_purchase_authorized"] is False
    assert manifest["claim_boundary"]["level1_ready"] is False


def test_second_build11_fault_gate_passes_locally_without_claiming_level1() -> None:
    path = ROOT / (
        "experiments/txnopt/manifests/txnopt_level1_fault_gate_attempt02.json"
    )
    digest, filename = (
        path.with_suffix(".json.sha256").read_text(encoding="utf-8").strip().split()
    )
    manifest = json.loads(path.read_bytes())

    assert filename == path.name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert manifest["status"] == "LOCAL_FAULT_GATE_PASS_NOT_LEVEL1_READY"
    assert manifest["matrix"]["fault_category_count"] == 24
    assert manifest["matrix"]["pytest_case_count"] == 25
    assert manifest["matrix"]["failed"] == 0
    assert manifest["matrix"]["errors"] == 0
    assert manifest["matrix"]["fallback_count"] == 0
    assert set(manifest["gate_results"].values()) == {"PASS"}
    assert manifest["immutability"]["failed_attempt01_retained"] is True
    assert manifest["immutability"]["build11_review_request_hash_mismatch_count"] == 0
    assert manifest["claim_boundary"]["local_fault_gate"] == "PASS"
    assert manifest["claim_boundary"]["cloud_purchase_authorized"] is False
    assert manifest["claim_boundary"]["formal_matrix_started"] is False
    assert manifest["claim_boundary"]["level1_ready"] is False


def test_build11_precloud_receipt_passes_local_gates_but_blocks_procurement() -> None:
    path = ROOT / (
        "experiments/txnopt/manifests/txnopt_level1_precloud_gate_attempt04.json"
    )
    digest, filename = (
        path.with_suffix(".json.sha256").read_text(encoding="utf-8").strip().split()
    )
    manifest = json.loads(path.read_bytes())

    assert filename == path.name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert manifest["status"] == "BLOCKED_BUILD11_INDEPENDENT_REVIEW_PENDING"
    assert manifest["local_precloud_gates"]["local_precloud_gate_complete"] is True
    assert manifest["local_precloud_gates"]["fault_and_prefix_gate"] == "PASS"
    assert manifest["local_precloud_gates"]["fallback_count"] == 0
    assert manifest["formal_plan"]["config_count"] == 2880
    assert manifest["formal_plan"]["raw_output_root_absent"] is True
    assert manifest["formal_and_legacy_reviews"][
        "build11_review_request_hash_mismatch_count"
    ] == 0
    assert manifest["formal_and_legacy_reviews"][
        "build11_independent_successor_review_completed"
    ] is False
    assert manifest["decision"]["local_precloud_gate"] == "PASS"
    assert manifest["decision"]["precloud_gate"] == "BLOCKED"
    assert manifest["execution_boundary"]["procurement_authorized"] is False
    assert manifest["execution_boundary"]["formal_matrix_started"] is False
    assert manifest["execution_boundary"]["level1_ready"] is False


def test_third_build11_fault_gate_exhausts_bounded_runtime_microstates() -> None:
    path = ROOT / (
        "experiments/txnopt/manifests/txnopt_level1_fault_gate_attempt03.json"
    )
    digest, filename = (
        path.with_suffix(".json.sha256").read_text(encoding="utf-8").strip().split()
    )
    manifest = json.loads(path.read_bytes())

    assert filename == path.name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert manifest["status"] == "LOCAL_EXHAUSTIVE_FAULT_GATE_PASS_NOT_LEVEL1_READY"
    assert manifest["matrix"]["fault_category_count"] == 27
    assert manifest["matrix"]["pytest_case_count"] == 75
    assert manifest["matrix"]["auditor_owned_exhaustive_microstate_case_count"] == 50
    assert manifest["exhaustive_microstates"]["completion_permutation_count"] == 6
    assert manifest["exhaustive_microstates"]["completion_order_case_count"] == 12
    assert manifest["exhaustive_microstates"][
        "worker_failure_cross_product_case_count"
    ] == 36
    assert manifest["exhaustive_microstates"]["deadline_checkpoint_case_count"] == 2
    assert set(manifest["gate_results"].values()) == {"PASS"}
    assert manifest["claim_boundary"]["general_unbounded_proof_claimed"] is False
    assert manifest["claim_boundary"]["cloud_purchase_authorized"] is False
    assert manifest["claim_boundary"]["formal_matrix_started"] is False
    assert manifest["claim_boundary"]["level1_ready"] is False


def test_fifth_precloud_receipt_binds_the_exhaustive_fault_gate() -> None:
    path = ROOT / (
        "experiments/txnopt/manifests/txnopt_level1_precloud_gate_attempt05.json"
    )
    digest, filename = (
        path.with_suffix(".json.sha256").read_text(encoding="utf-8").strip().split()
    )
    manifest = json.loads(path.read_bytes())

    assert filename == path.name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert manifest["status"] == "BLOCKED_BUILD11_INDEPENDENT_REVIEW_PENDING"
    local = manifest["local_precloud_gates"]
    assert local["local_precloud_gate_complete"] is True
    assert local["exhaustive_bounded_microstate_fault_gate"] == "PASS_ATTEMPT03"
    assert local["fault_category_count"] == 27
    assert local["fault_pytest_case_count"] == 75
    assert local["auditor_owned_exhaustive_microstate_case_count"] == 50
    assert local["fallback_count"] == 0
    assert manifest["formal_and_legacy_reviews"][
        "build11_independent_successor_review_completed"
    ] is False
    assert manifest["decision"]["local_precloud_gate"] == "PASS"
    assert manifest["decision"]["precloud_gate"] == "BLOCKED"
    assert manifest["execution_boundary"]["procurement_authorized"] is False
    assert manifest["execution_boundary"]["formal_matrix_started"] is False
    assert manifest["execution_boundary"]["level1_ready"] is False
