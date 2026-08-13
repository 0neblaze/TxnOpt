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
