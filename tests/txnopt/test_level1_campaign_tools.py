from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from tools.review_txnopt_level1_campaign import (
    _aggregate,
    _audit_t4_waste_events,
    _gate_results,
    _semantic_exact_work_started,
)
from tools.run_txnopt_level1_campaign import (
    _physical_core_count,
    preflight_campaign,
)
from tools.txnopt_level1_campaign_common import (
    AXES,
    campaign_claim_path,
    canonical_json_bytes,
    load_analysis_protocol,
    load_campaign_plan,
    load_prebound_expected_identity,
    require_clean_repository,
    run_isolated_process,
    sha256_bytes,
    sha256_file,
    validate_authorization,
    write_signed_object,
)
from txnopt_evidence.identity import ExpectedEvidenceIdentity


def _write(path: Path, payload: object) -> str:
    return write_signed_object(path, payload)


def _rebind_plan_to_build(
    plan_path: Path,
    plan: dict[str, object],
    build_path: Path,
    build_sha256: str,
) -> None:
    entries = plan["entries"]
    assert isinstance(entries, list)
    for raw_entry in entries:
        assert isinstance(raw_entry, dict)
        config_path = plan_path.parent / str(raw_entry["path"])
        config = json.loads(config_path.read_bytes())
        config["build_manifest"]["sha256"] = build_sha256
        config_bytes = canonical_json_bytes(config, pretty=True)
        config_path.write_bytes(config_bytes)
        raw_entry["sha256"] = sha256_bytes(config_bytes)
        identity_path = plan_path.parent / str(raw_entry["expected_identity_path"])
        identity_path.unlink()
        identity_path.with_suffix(".json.sha256").unlink()
        identity = ExpectedEvidenceIdentity.from_plan_inputs(
            config_path,
            build_manifest_path=build_path,
        )
        raw_entry["expected_identity_sha256"] = _write(
            identity_path,
            identity.to_payload(),
        )
    plan["build_manifest_sha256"] = build_sha256
    plan["config_tree_sha256"] = sha256_bytes(
        canonical_json_bytes(
            [{"path": entry["path"], "sha256": entry["sha256"]} for entry in entries]
        )
    )
    plan["expected_identity_tree_sha256"] = sha256_bytes(
        canonical_json_bytes(
            [
                {
                    "path": entry["expected_identity_path"],
                    "sha256": entry["expected_identity_sha256"],
                }
                for entry in entries
            ]
        )
    )
    plan_path.unlink()
    plan_path.with_suffix(".json.sha256").unlink()
    _write(plan_path, plan)


def _campaign(tmp_path: Path) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    protocol_path = tmp_path / "level1-protocol.json"
    protocol = {
        "schema_version": "txnopt-level1-protocol-v1",
        "holdout_opened": False,
        "level2_holdout_opened": False,
        "level3_holdout_opened": False,
        "domains": {
            "evrptw": {"pilot": ["c101C5"], "validation": []},
            "rcpsp": {"pilot": ["j1201_1"], "validation": []},
        },
        "seeds": [2014],
        "formal_axes": list(AXES),
        "budgets": ["fixed_work", "fixed_time"],
    }
    protocol_sha256 = _write(protocol_path, protocol)
    build_path = tmp_path / "build.json"
    build = {
        "schema_version": "txnopt-level1-build-manifest-v1",
        "run_label": "txnopt_level1_build_attempt14",
        "status": "BUILD_COMPLETE_ANCHORED_EVIDENCE_REVIEW_PENDING_NOT_LEVEL1_READY",
        "producer": {
            "revision": "a" * 40,
            "git_tree": "b" * 40,
            "source_manifest_sha256": "c" * 64,
            "tracked_file_count": 1,
            "source_dirty": False,
            "development_override": False,
        },
        "artifacts": {
            "native_extension": {
                "protocol": "txnopt-native-round-v1",
                "sha256": "d" * 64,
            },
            "wheel": {"sha256": "e" * 64},
            "resource_soak": {"status": "PASS", "fallback_count": 0},
        },
        "formal_successor": {
            "prior_review_binding_status": "PRIOR_SOURCE_ONLY",
            "successor_status": "REVIEW_PENDING_BUILD14",
            "independent_successor_review_completed": False,
            "level1_formal_gate_passed": False,
        },
        "validation": {
            **{
                gate: {"status": "PASS", "exit_code": 0}
                for gate in (
                    "ruff",
                    "strict_mypy",
                    "pytest_wheel_installed",
                    "property_based_tests",
                    "wheel_surface_and_record",
                    "legacy_verify",
                    "asan_ubsan",
                    "tsan",
                )
            },
            "historical_protected_paths": {"status": "UNCHANGED", "exit_code": 0},
        },
    }
    build_sha256 = _write(build_path, build)
    catalog_path = tmp_path / "catalog.json"
    catalog_sha256 = _write(
        catalog_path,
        {
            "schema_version": "txnopt-level1-case-catalog-v1",
            "domains": {
                "evrptw": {"c101C5": str(tmp_path / "c101C5.txt")},
                "rcpsp": {"j1201_1": str(tmp_path / "j1201_1.txt")},
            },
        },
    )
    source_digests: dict[str, str] = {}
    for case_id in ("c101C5", "j1201_1"):
        source = tmp_path / f"{case_id}.txt"
        source.write_text(f"fixture {case_id}\n", encoding="utf-8")
        source_digests[case_id] = sha256_file(source)
    raw_root = tmp_path / "raw"
    plan_root = tmp_path / "plan"
    configs = plan_root / "configs"
    configs.mkdir(parents=True)
    identities_root = plan_root / "expected-identities"
    identities_root.mkdir()
    entries: list[dict[str, str]] = []
    for domain, case_id in (("evrptw", "c101C5"), ("rcpsp", "j1201_1")):
        for axis, (mode, workers, speculation) in AXES.items():
            for budget in ("fixed_work", "fixed_time"):
                label = f"txnopt_level1_{domain}_{case_id.lower()}_2014_{axis}_{budget}_attempt18"
                config = {
                    "schema_version": "txnopt-run-config-v1",
                    "run_label": label,
                    "output_root": str(raw_root),
                    "build_manifest": {
                        "path": str(build_path),
                        "sha256": build_sha256,
                    },
                    "run_config": {
                        "seed": 2014,
                        "workers": workers,
                        "execution_mode": mode,
                        "fixed_work": 1200 if budget == "fixed_work" else None,
                        "deadline_seconds": 3.0 if budget == "fixed_time" else None,
                        "speculation_window": speculation,
                        "trace_policy": "semantic_and_physical",
                        "max_rounds": 10,
                    },
                    "case": {
                        "domain": domain,
                        "source_instance_path": str(tmp_path / f"{case_id}.txt"),
                        "source_instance_sha256": source_digests[case_id],
                        "oracle_seed": 2014,
                    },
                }
                relative = f"configs/{label}.json"
                data = canonical_json_bytes(config, pretty=True)
                config_path = plan_root / relative
                config_path.write_bytes(data)
                expected_identity = ExpectedEvidenceIdentity.from_plan_inputs(
                    config_path,
                    build_manifest_path=build_path,
                )
                identity_relative = f"expected-identities/{label}.json"
                identity_sha256 = _write(
                    plan_root / identity_relative,
                    expected_identity.to_payload(),
                )
                entries.append(
                    {
                        "path": relative,
                        "sha256": sha256_bytes(data),
                        "expected_identity_path": identity_relative,
                        "expected_identity_sha256": identity_sha256,
                    }
                )
    config_tree_entries = [
        {"path": entry["path"], "sha256": entry["sha256"]} for entry in entries
    ]
    identity_tree_entries = [
        {
            "path": entry["expected_identity_path"],
            "sha256": entry["expected_identity_sha256"],
        }
        for entry in entries
    ]
    plan = {
        "schema_version": "txnopt-level1-campaign-plan-v2",
        "status": "PLANNED_NOT_STARTED",
        "protocol_path": str(protocol_path),
        "protocol_sha256": protocol_sha256,
        "catalog_path": str(catalog_path),
        "catalog_sha256": catalog_sha256,
        "raw_output_root": str(raw_root),
        "build_manifest_path": str(build_path),
        "build_manifest_sha256": build_sha256,
        "config_count": len(entries),
        "config_tree_sha256": sha256_bytes(canonical_json_bytes(config_tree_entries)),
        "expected_identity_tree_sha256": sha256_bytes(
            canonical_json_bytes(identity_tree_entries)
        ),
        "fixed_work": 1200,
        "fixed_time_seconds": 3.0,
        "max_rounds": 10,
        "attempt": 18,
        "holdout_opened": False,
        "cloud_purchase_authorized": False,
        "entries": entries,
    }
    plan_path = plan_root / "manifest.json"
    plan_sha256 = _write(plan_path, plan)
    analysis_path = tmp_path / "analysis.json"
    _write(
        analysis_path,
        {
            "schema_version": "txnopt-level1-analysis-protocol-v1",
            "campaign_plan_sha256": plan_sha256,
            "level1_protocol_sha256": protocol_sha256,
            "holdout_opened": False,
            "status": "PREREGISTERED_BEFORE_FORMAL_MATRIX",
            "fixed_work_quality_gate": {
                "comparison": (
                    "exact canonical objective and semantic digest equality across "
                    "serial_1, txnopt_1, txnopt_4, and barrier_4"
                ),
                "maximum_regressions": 0,
            },
            "pairing_unit": ["domain", "case_id", "seed"],
            "performance_gates": {
                "four_worker_geomean_minimum": 1.3,
                "four_worker_ci_lower_minimum_exclusive": 1.05,
                "one_worker_maximum_overhead_fraction": 0.15,
            },
            "bootstrap": {
                "method": "paired_percentile_bootstrap_of_log_speedup",
                "confidence_level": 0.95,
                "quantile_rule": "nearest_rank",
                "resamples": 20000,
                "seed": 20260813,
            },
            "performance_measure": {
                "budget_axis": "fixed_work",
                "duration_field": "txnopt-physical-trace-v1.run_observation.duration_ns",
                "four_worker_ratio": "serial_1_duration_ns / txnopt_4_duration_ns",
                "one_worker_overhead": ("max(0, txnopt_1_duration_ns / serial_1_duration_ns - 1)"),
            },
            "t4_cmax_gate": {
                "require_positive_observed_cmax_for_every_run_with_semantic_exact_work": True,
                "require_independent_waste_bound_recomputation": True,
            },
            "resource_environment": {
                "exclusive_linux_required": True,
                "maximum_consecutive_window_days": 14,
                "minimum_memory_gib": 128,
                "minimum_physical_cores": 32,
            },
        },
    )
    return plan_path, analysis_path


def test_preflight_verifies_the_complete_plan_without_creating_raw_output(
    tmp_path: Path,
) -> None:
    plan_path, analysis_path = _campaign(tmp_path)

    receipt = preflight_campaign(plan_path, analysis_path)

    assert receipt["status"] == "PASS_NOT_AUTHORIZED_TO_EXECUTE"
    assert receipt["config_count"] == 16
    assert receipt["expected_identity_tree_sha256"] == json.loads(
        plan_path.read_bytes()
    )["expected_identity_tree_sha256"]
    assert receipt["cloud_purchase_authorized"] is False
    assert receipt["formal_matrix_started"] is False
    assert not Path(receipt["raw_output_root"]).exists()


def test_preflight_rejects_legacy_plan_without_prebound_identity_tree(
    tmp_path: Path,
) -> None:
    plan_path, analysis_path = _campaign(tmp_path)
    plan = json.loads(plan_path.read_bytes())
    plan["schema_version"] = "txnopt-level1-campaign-plan-v1"
    plan.pop("expected_identity_tree_sha256")
    plan["entries"] = [
        {"path": entry["path"], "sha256": entry["sha256"]}
        for entry in plan["entries"]
    ]
    plan_path.unlink()
    plan_path.with_suffix(".json.sha256").unlink()
    _write(plan_path, plan)
    analysis = json.loads(analysis_path.read_bytes())
    analysis["campaign_plan_sha256"] = sha256_file(plan_path)
    analysis_path.unlink()
    analysis_path.with_suffix(".json.sha256").unlink()
    _write(analysis_path, analysis)

    with pytest.raises(ValueError, match="prebound expected identity"):
        preflight_campaign(plan_path, analysis_path)


def test_plan_loader_accepts_build10_only_with_its_pending_formal_boundary(
    tmp_path: Path,
) -> None:
    plan_path, _analysis_path = _campaign(tmp_path)
    plan_payload = json.loads(plan_path.read_bytes())
    build_path = Path(plan_payload["build_manifest_path"])
    build = json.loads(build_path.read_bytes())
    build.update(
        {
            "run_label": "txnopt_level1_build_attempt10",
            "status": "BUILD_COMPLETE_FORMAL_SUCCESSOR_REVIEW_PENDING_NOT_LEVEL1_READY",
            "formal_successor": {
                "prior_review_binding_status": "PRIOR_SOURCE_ONLY",
                "successor_status": "REVIEW_PENDING_BUILD10",
                "independent_successor_review_completed": False,
                "level1_formal_gate_passed": False,
            },
        }
    )
    build_path.unlink()
    build_path.with_suffix(".json.sha256").unlink()
    build_sha256 = _write(build_path, build)
    _rebind_plan_to_build(plan_path, plan_payload, build_path, build_sha256)

    with pytest.raises(ValueError, match="requires the approved Build14 producer"):
        load_campaign_plan(plan_path)

    build["formal_successor"]["independent_successor_review_completed"] = True
    build_path.unlink()
    build_path.with_suffix(".json.sha256").unlink()
    build_sha256 = _write(build_path, build)
    _rebind_plan_to_build(plan_path, plan_payload, build_path, build_sha256)
    with pytest.raises(ValueError, match="formal-successor boundary"):
        load_campaign_plan(plan_path)


def test_plan_loader_accepts_build11_only_with_its_pending_lifecycle_boundary(
    tmp_path: Path,
) -> None:
    plan_path, _analysis_path = _campaign(tmp_path)
    plan_payload = json.loads(plan_path.read_bytes())
    build_path = Path(plan_payload["build_manifest_path"])
    build = json.loads(build_path.read_bytes())
    build.update(
        {
            "run_label": "txnopt_level1_build_attempt11",
            "status": (
                "BUILD_COMPLETE_EVIDENCE_LIFECYCLE_REVIEW_PENDING_NOT_LEVEL1_READY"
            ),
            "formal_successor": {
                "prior_review_binding_status": "PRIOR_SOURCE_ONLY",
                "successor_status": "REVIEW_PENDING_BUILD11",
                "independent_successor_review_completed": False,
                "level1_formal_gate_passed": False,
            },
        }
    )
    build_path.unlink()
    build_path.with_suffix(".json.sha256").unlink()
    build_sha256 = _write(build_path, build)
    _rebind_plan_to_build(plan_path, plan_payload, build_path, build_sha256)

    with pytest.raises(ValueError, match="requires the approved Build14 producer"):
        load_campaign_plan(plan_path)

    build["formal_successor"]["successor_status"] = "REVIEW_PENDING_BUILD10"
    build_path.unlink()
    build_path.with_suffix(".json.sha256").unlink()
    build_sha256 = _write(build_path, build)
    _rebind_plan_to_build(plan_path, plan_payload, build_path, build_sha256)
    with pytest.raises(ValueError, match="Build11 formal-successor boundary"):
        load_campaign_plan(plan_path)


def test_plan_loader_accepts_build14_only_as_an_anchored_review_successor(
    tmp_path: Path,
) -> None:
    plan_path, _analysis_path = _campaign(tmp_path)
    plan = json.loads(plan_path.read_bytes())
    build_path = Path(plan["build_manifest_path"])
    build = json.loads(build_path.read_bytes())
    build.update(
        {
            "run_label": "txnopt_level1_build_attempt14",
            "status": "BUILD_COMPLETE_ANCHORED_EVIDENCE_REVIEW_PENDING_NOT_LEVEL1_READY",
            "formal_successor": {
                "prior_review_binding_status": "PRIOR_SOURCE_ONLY",
                "successor_status": "REVIEW_PENDING_BUILD14",
                "independent_successor_review_completed": False,
                "level1_formal_gate_passed": False,
            },
        }
    )
    build_path.unlink()
    build_path.with_suffix(".json.sha256").unlink()
    build_sha256 = _write(build_path, build)
    _rebind_plan_to_build(plan_path, plan, build_path, build_sha256)

    assert load_campaign_plan(plan_path).build_manifest_sha256 == build_sha256

    build["formal_successor"]["successor_status"] = "REVIEW_PENDING_BUILD12"
    build_path.unlink()
    build_path.with_suffix(".json.sha256").unlink()
    build_sha256 = _write(build_path, build)
    _rebind_plan_to_build(plan_path, plan, build_path, build_sha256)
    with pytest.raises(ValueError, match="Build14 formal-successor boundary"):
        load_campaign_plan(plan_path)


def test_plan_v2_rejects_failed_build13_even_with_its_original_boundary(
    tmp_path: Path,
) -> None:
    plan_path, _analysis_path = _campaign(tmp_path)
    plan = json.loads(plan_path.read_bytes())
    build_path = Path(plan["build_manifest_path"])
    build = json.loads(build_path.read_bytes())
    build.update(
        {
            "run_label": "txnopt_level1_build_attempt13",
            "status": "BUILD_COMPLETE_ANCHORED_EVIDENCE_REVIEW_PENDING_NOT_LEVEL1_READY",
            "formal_successor": {
                "prior_review_binding_status": "PRIOR_SOURCE_ONLY",
                "successor_status": "REVIEW_PENDING_BUILD13",
                "independent_successor_review_completed": False,
                "level1_formal_gate_passed": False,
            },
        }
    )
    build_path.unlink()
    build_path.with_suffix(".json.sha256").unlink()
    build_sha256 = _write(build_path, build)
    _rebind_plan_to_build(plan_path, plan, build_path, build_sha256)

    with pytest.raises(ValueError, match="not an approved Level 1 producer"):
        load_campaign_plan(plan_path)


def test_preflight_rejects_config_tampering_and_an_existing_raw_root(
    tmp_path: Path,
) -> None:
    plan_path, analysis_path = _campaign(tmp_path)
    plan = json.loads(plan_path.read_bytes())
    first = plan_path.parent / plan["entries"][0]["path"]
    first.write_bytes(first.read_bytes() + b" ")
    with pytest.raises(ValueError, match="config digest differs"):
        preflight_campaign(plan_path, analysis_path)


def test_plan_loader_rejects_an_expected_identity_resigned_after_planning(
    tmp_path: Path,
) -> None:
    plan_path, _analysis_path = _campaign(tmp_path)
    plan = json.loads(plan_path.read_bytes())
    first = plan["entries"][0]
    identity_path = plan_path.parent / first["expected_identity_path"]
    identity = json.loads(identity_path.read_bytes())
    identity["expected_oracle"] = "forged.oracle"
    identity_path.unlink()
    identity_path.with_suffix(".json.sha256").unlink()
    first["expected_identity_sha256"] = _write(identity_path, identity)
    plan["expected_identity_tree_sha256"] = sha256_bytes(
        canonical_json_bytes(
            [
                {
                    "path": entry["expected_identity_path"],
                    "sha256": entry["expected_identity_sha256"],
                }
                for entry in plan["entries"]
            ]
        )
    )
    plan_path.unlink()
    plan_path.with_suffix(".json.sha256").unlink()
    _write(plan_path, plan)

    with pytest.raises(ValueError, match="expected identity differs from plan inputs"):
        load_campaign_plan(plan_path)


def test_plan_loader_rejects_noncanonical_expected_identity_bytes(
    tmp_path: Path,
) -> None:
    plan_path, _analysis_path = _campaign(tmp_path)
    plan = json.loads(plan_path.read_bytes())
    first = plan["entries"][0]
    identity_path = plan_path.parent / first["expected_identity_path"]
    identity = json.loads(identity_path.read_bytes())
    noncanonical = json.dumps(identity, sort_keys=False).encode()
    identity_path.write_bytes(noncanonical)
    identity_digest = sha256_bytes(noncanonical)
    identity_path.with_suffix(".json.sha256").write_text(
        f"{identity_digest}  {identity_path.name}\n",
        encoding="utf-8",
    )
    first["expected_identity_sha256"] = identity_digest
    plan["expected_identity_tree_sha256"] = sha256_bytes(
        canonical_json_bytes(
            [
                {
                    "path": entry["expected_identity_path"],
                    "sha256": entry["expected_identity_sha256"],
                }
                for entry in plan["entries"]
            ]
        )
    )
    plan_path.unlink()
    plan_path.with_suffix(".json.sha256").unlink()
    _write(plan_path, plan)

    with pytest.raises(ValueError, match="canonical bytes"):
        load_campaign_plan(plan_path)


def test_prebound_identity_is_rechecked_immediately_before_use(tmp_path: Path) -> None:
    plan_path, _analysis_path = _campaign(tmp_path)
    entry = load_campaign_plan(plan_path).entries[0]
    entry.config_path.write_bytes(entry.config_path.read_bytes() + b" ")

    with pytest.raises(ValueError, match="config changed after plan validation"):
        load_prebound_expected_identity(entry)

    plan_path, analysis_path = _campaign(tmp_path / "second")
    raw_root = Path(json.loads(plan_path.read_bytes())["raw_output_root"])
    raw_root.mkdir()
    with pytest.raises(FileExistsError, match="new attempt"):
        preflight_campaign(plan_path, analysis_path)


def test_execution_requires_an_exact_separate_authorization(tmp_path: Path) -> None:
    plan_path, analysis_path = _campaign(tmp_path)
    plan = load_campaign_plan(plan_path)
    analysis_sha256 = sha256_file(analysis_path)
    build = json.loads(plan.build_manifest_path.read_bytes())
    authorization = {
        "schema_version": "txnopt-level1-procurement-authorization-v1",
        "authorized": True,
        "authorization_scope": "execute_exact_level1_attempt18_only",
        "campaign_plan_sha256": plan.manifest_sha256,
        "analysis_protocol_sha256": analysis_sha256,
        "config_tree_sha256": plan.payload["config_tree_sha256"],
        "expected_identity_tree_sha256": plan.expected_identity_tree_sha256,
        "build_manifest_sha256": plan.build_manifest_sha256,
        "wheel_sha256": build["artifacts"]["wheel"]["sha256"],
        "raw_output_root": str(plan.raw_output_root),
        "holdout_opened": False,
        "maximum_window_days": 14,
        "exclusive_linux": True,
    }
    validate_authorization(authorization, plan, analysis_sha256)
    authorization["campaign_plan_sha256"] = "0" * 64
    with pytest.raises(PermissionError, match="exact campaign"):
        validate_authorization(authorization, plan, analysis_sha256)
    authorization["campaign_plan_sha256"] = plan.manifest_sha256
    authorization["expected_identity_tree_sha256"] = "0" * 64
    with pytest.raises(PermissionError, match="exact campaign"):
        validate_authorization(authorization, plan, analysis_sha256)


def test_cli_preflight_requires_a_clean_orchestration_repository(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Result:
        stdout = "?? untracked-tool.py\n"

    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: _Result())
    with pytest.raises(RuntimeError, match="must be clean"):
        require_clean_repository(Path("/repository"))


def test_reviewer_requires_a_clean_committed_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools.review_txnopt_level1_campaign import _reviewer_identity

    monkeypatch.setattr(
        "tools.review_txnopt_level1_campaign.require_clean_repository",
        lambda root: (_ for _ in ()).throw(RuntimeError("dirty reviewer")),
    )
    with pytest.raises(RuntimeError, match="dirty reviewer"):
        _reviewer_identity()


def test_campaign_statistics_use_paired_fixed_work_and_fail_closed() -> None:
    records: list[dict[str, object]] = []
    ordinal = 0
    for domain in ("evrptw", "rcpsp"):
        for case_index in range(3):
            for axis, duration in {
                "serial_1": 200,
                "txnopt_1": 210,
                "txnopt_4": 100,
                "barrier_4": 130,
            }.items():
                ordinal += 1
                records.append(
                    {
                        "ordinal": ordinal,
                        "status": "PASS",
                        "domain": domain,
                        "case_id": f"case{case_index}",
                        "seed": 2014,
                        "axis": axis,
                        "budget": "fixed_work",
                        "duration_ns": duration,
                        "objective": [1, 2, 3],
                        "semantic_digest": f"{domain}-{case_index}",
                        "semantic_exact_work_started": True,
                        "observed_cmax_upper_ns": 50,
                        "prefix_safety_replay": "PASS",
                        "aggregate_refinement_replay": "PASS",
                        "physical_t4_recomputation": "PASS",
                        "t4_waste_event_count": 0,
                    }
                )
    analysis = {
        "bootstrap": {
            "resamples": 100,
            "seed": 7,
            "confidence_level": 0.95,
        },
        "performance_gates": {
            "four_worker_geomean_minimum": 1.3,
            "four_worker_ci_lower_minimum_exclusive": 1.05,
            "one_worker_maximum_overhead_fraction": 0.15,
        },
    }

    metrics = _aggregate(records, analysis)
    gates = _gate_results(metrics, [])

    assert metrics["fixed_work"]["domains"]["evrptw"][
        "four_worker_geomean_speedup"
    ] == pytest.approx(2.0)
    assert metrics["fixed_work"]["domains"]["rcpsp"][
        "maximum_one_worker_overhead_fraction"
    ] == pytest.approx(0.05)
    assert set(gates.values()) == {"PASS"}

    records[1]["objective"] = [9, 9, 9]
    failed_metrics = _aggregate(records, analysis)
    assert (
        _gate_results(failed_metrics, [])["fixed_work_semantic_digest_and_objective_parity"]
        == "FAIL"
    )


def test_review_rejects_an_execution_receipt_without_clean_tool_identity(
    tmp_path: Path,
) -> None:
    from tools.review_txnopt_level1_campaign import _validate_execution_receipt

    plan_path, analysis_path = _campaign(tmp_path)
    plan = load_campaign_plan(plan_path)
    analysis_sha256 = sha256_file(analysis_path)
    receipt: dict[str, object] = {
        "status": "COMPLETE_RAW_ONLY_NOT_REVIEWED",
        "plan_manifest_sha256": plan.manifest_sha256,
        "analysis_protocol_sha256": analysis_sha256,
        "completed_run_count": len(plan.entries),
        "failed_run_count": 0,
        "not_started_run_count": 0,
        "independent_review_performed": False,
        "readiness_decision": None,
        "fallback_count": 0,
        "runs": [],
    }
    with pytest.raises(ValueError, match="runner orchestration identity"):
        _validate_execution_receipt(
            receipt,
            plan,
            analysis_sha256,
            expected_runtime_identity={},
        )


def test_analysis_protocol_file_digest_is_bound_by_the_plan_fixture(tmp_path: Path) -> None:
    plan_path, analysis_path = _campaign(tmp_path)
    assert (
        sha256_file(analysis_path)
        == analysis_path.with_suffix(".json.sha256").read_text().split()[0]
    )
    assert (
        load_campaign_plan(plan_path).manifest_sha256
        == plan_path.with_suffix(".json.sha256").read_text().split()[0]
    )


def test_analysis_preregistration_cannot_be_relaxed(tmp_path: Path) -> None:
    plan_path, analysis_path = _campaign(tmp_path)
    plan = load_campaign_plan(plan_path)
    payload = json.loads(analysis_path.read_bytes())
    payload["performance_gates"]["four_worker_geomean_minimum"] = 1.0
    data = canonical_json_bytes(payload, pretty=True)
    analysis_path.write_bytes(data)
    analysis_path.with_suffix(".json.sha256").write_text(
        f"{sha256_bytes(data)}  {analysis_path.name}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exact preregistration"):
        load_analysis_protocol(analysis_path, plan=plan)


def test_preflight_rejects_an_existing_atomic_launch_claim(tmp_path: Path) -> None:
    plan_path, analysis_path = _campaign(tmp_path)
    plan = load_campaign_plan(plan_path)
    campaign_claim_path(plan).mkdir()

    with pytest.raises(FileExistsError, match="launch is already claimed"):
        preflight_campaign(plan_path, analysis_path)


def test_campaign_safety_gates_require_every_independent_replay() -> None:
    metrics = {
        "fixed_work": {
            "domains": {
                domain: {
                    "one_worker_overhead_gate": "PASS",
                    "geomean_gate": "PASS",
                    "ci_lower_gate": "PASS",
                }
                for domain in ("evrptw", "rcpsp")
            },
            "semantic_and_objective_parity_failure_count": 0,
        },
        "cmax": {
            "semantic_exact_work_run_count": 2,
            "measured_positive_cmax_run_count": 2,
        },
        "raw_to_review_traceability": {"expected_run_count": 2},
        "safety_replay": {
            "prefix_safety_pass_count": 2,
            "aggregate_refinement_pass_count": 2,
            "physical_t4_recomputation_pass_count": 1,
            "build09_fault_and_formal_gate": "PASS",
        },
    }

    gates = _gate_results(metrics, [])

    assert gates["deadline_and_fault_prefix_safety"] == "PASS"
    assert gates["measured_waste_within_t4_bound"] == "FAIL"


def test_interrupted_transaction_still_requires_measured_cmax() -> None:
    events = (
        {
            "event": "candidate_transaction",
            "phase": "INTERRUPTED",
            "started_work": 3,
        },
    )

    assert _semantic_exact_work_started(events) is True


def test_campaign_recomputes_t4_bounds_and_requires_fixed_work_abort_audit() -> None:
    semantic = (
        {
            "event": "candidate_transaction",
            "phase": "INTERRUPTED",
            "started_work": 2,
        },
    )
    observation = {
        "event": "run_observation",
        "observed_cmax_upper_ns": 10,
    }
    waste = {
        "event": "t4_waste_observation",
        "trace": "txnopt-physical-trace-v1",
        "bound_satisfied": True,
        "remaining_budget_before": 4,
        "uncommitted_window": 2,
        "max_requests_per_candidate": 1,
        "post_boundary_capacity_units": 4,
        "observed_discarded_work_units": 2,
        "observed_post_boundary_work_units": 1,
        "discarded_work_bound_units": 2,
        "post_boundary_work_bound_units": 2,
        "measured_cmax_ns": 10,
        "observed_discarded_cost_upper_ns": 20,
        "discarded_cost_bound_ns": 20,
        "observed_post_boundary_cost_upper_ns": 10,
        "post_boundary_cost_bound_ns": 20,
    }

    receipt = _audit_t4_waste_events(
        (observation, waste),
        semantic_events=semantic,
        budget="fixed_work",
    )
    assert receipt["independently_recomputed_t4_event_count"] == 1

    tampered = {**waste, "discarded_work_bound_units": 3}
    with pytest.raises(ValueError, match="recomputed bound"):
        _audit_t4_waste_events(
            (observation, tampered),
            semantic_events=semantic,
            budget="fixed_work",
        )
    with pytest.raises(ValueError, match="lacks its T4"):
        _audit_t4_waste_events(
            (observation,),
            semantic_events=semantic,
            budget="fixed_work",
        )


def test_subprocess_timeout_terminates_the_complete_process_group(tmp_path: Path) -> None:
    script = (
        "import subprocess,sys,time; "
        "subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
        "time.sleep(30)"
    )

    result = run_isolated_process(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        timeout_seconds=0.1,
    )

    assert result.timed_out is True
    assert result.descendant_cleanup_performed is True
    assert result.descendant_processes_remaining == ()


def test_reviewer_does_not_import_the_raw_producer_module() -> None:
    reviewer = Path("tools/review_txnopt_level1_campaign.py").read_text(encoding="utf-8")
    assert "from tools.run_txnopt_level1_campaign import" not in reviewer


def test_linux_resource_floor_measurements_are_positive() -> None:
    assert _physical_core_count() > 0
    from tools.run_txnopt_level1_campaign import _memory_gib

    assert _memory_gib() > 0


def test_precloud_gate_is_not_an_execution_or_level1_readiness_claim() -> None:
    path = Path("experiments/txnopt/manifests/txnopt_level1_precloud_gate_attempt01.json")
    payload = json.loads(path.read_bytes())

    assert payload["status"] == "READY_FOR_SEPARATE_PROCUREMENT_AUTHORIZATION"
    assert payload["clean_cli_preflight"]["status"] == ("PASS_NOT_AUTHORIZED_TO_EXECUTE")
    assert payload["execution_boundary"] == {
        "procurement_authorized": False,
        "cloud_purchase_performed": False,
        "formal_matrix_started": False,
        "holdout_opened": False,
        "level1_ready": False,
        "internal_level1_seal_created": False,
        "level2_entry_authorized": False,
        "push_authorized": False,
        "public_release_authorized": False,
    }
    assert sha256_file(path) == path.with_suffix(".json.sha256").read_text().split()[0]
