"""Materialize the fixed Level 1 matrix without opening a holdout or running it."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from txnopt_cases.evrptw import Instance, construct_initial_plan, parse_schneider
from txnopt_cases.rcpsp import (
    RCPSPInstance,
    parse_psplib_sm,
    precedence_feasible_initial_state,
)
from txnopt_evidence.codec import (
    canonical_json_bytes,
    read_signed_json,
    sha256_bytes,
    verify_sidecar,
    write_sidecar,
)
from txnopt_evidence.identity import ExpectedEvidenceIdentity
from txnopt_evidence.level1_protocol import validate_level1_protocol_v2

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_]{1,63}")
_AXES: Mapping[str, tuple[str, int, int]] = {
    "serial_1": ("serial", 1, 0),
    "txnopt_1": ("ordered", 1, 1),
    "txnopt_4": ("ordered", 4, 4),
    "barrier_4": ("barrier", 4, 0),
}


def materialize_level1_plan(
    protocol_path: Path,
    catalog_path: Path,
    *,
    destination: Path,
    raw_output_root: Path,
    build_manifest_path: Path,
    fixed_work: int,
    fixed_time_seconds: float,
    max_rounds: int,
    evrptw_max_candidates: int,
    rcpsp_max_candidates: int,
    attempt: int = 1,
) -> Path:
    """Create signed run configs for the protocol; never execute or buy resources."""

    if (
        min(
            fixed_work,
            max_rounds,
            evrptw_max_candidates,
            rcpsp_max_candidates,
            attempt,
        )
        <= 0
        or fixed_time_seconds <= 0.0
    ):
        raise ValueError("campaign budgets, rounds, and attempt must be positive")
    protocol_source = protocol_path.resolve(strict=True)
    catalog_source = catalog_path.resolve(strict=True)
    build_manifest_source = build_manifest_path.resolve(strict=True)
    build_manifest_sha256 = verify_sidecar(build_manifest_source)
    build_manifest = read_signed_json(build_manifest_source)
    _validate_build_manifest(build_manifest)
    protocol = _object(json.loads(protocol_source.read_bytes()), "protocol")
    catalog = _object(json.loads(catalog_source.read_bytes()), "catalog")
    protocol_schema = protocol.get("schema_version")
    if protocol_schema not in {
        "txnopt-level1-protocol-v1",
        "txnopt-level1-protocol-v2",
    }:
        raise ValueError("unsupported Level 1 protocol schema")
    if (
        protocol_schema == "txnopt-level1-protocol-v2"
        and build_manifest.get("run_label") != "txnopt_level1_build_attempt16"
    ):
        raise ValueError("Level 1 protocol v2 requires the Build16 producer")
    if protocol.get("holdout_opened") is not False:
        raise ValueError("Level 1 planning requires a closed holdout")
    if catalog.get("schema_version") != "txnopt-level1-case-catalog-v1":
        raise ValueError("unsupported Level 1 case catalog schema")
    axes = tuple(_strings(protocol, "formal_axes"))
    if axes != tuple(_AXES):
        raise ValueError("Level 1 formal axes differ from the canonical order")
    if tuple(_strings(protocol, "budgets")) != ("fixed_work", "fixed_time"):
        raise ValueError("Level 1 requires fixed_work and fixed_time budgets")
    seeds = tuple(_integers(protocol, "seeds"))
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Level 1 seeds must be unique")
    resource_contract: dict[str, Any] | None = None
    if protocol_schema == "txnopt-level1-protocol-v2":
        resource_contract = validate_level1_protocol_v2(protocol)
        if (
            fixed_work != 1200
            or fixed_time_seconds != 3
            or max_rounds != 10
            or evrptw_max_candidates != 64
            or rcpsp_max_candidates != 64
        ):
            raise ValueError("Level 1 protocol v2 execution arguments differ")

    expected = _expected_case_ids(protocol)
    catalog_domains = _object(catalog.get("domains"), "catalog domains")
    catalog_paths: dict[str, dict[str, Path]] = {}
    for domain, expected_ids in expected.items():
        raw_paths = _object(catalog_domains.get(domain), f"{domain} catalog")
        if set(raw_paths) != set(expected_ids):
            raise ValueError(f"{domain} catalog does not exactly match the protocol")
        catalog_paths[domain] = {
            case_id: Path(_string_value(raw_paths[case_id])).resolve(strict=True)
            for case_id in expected_ids
        }
    if set(catalog_domains) != set(expected):
        raise ValueError("case catalog contains an unregistered domain")

    output = destination.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"campaign plan destination already exists: {output}")
    configs_dir = output / "configs"
    configs_dir.mkdir(parents=True, exist_ok=False)
    identities_dir = output / "expected-identities"
    identities_dir.mkdir(exist_ok=False)
    raw_root = raw_output_root.resolve()
    entries: list[dict[str, str]] = []
    case_payloads = _load_cases(
        catalog_paths,
        evrptw_max_candidates=evrptw_max_candidates,
        rcpsp_max_candidates=rcpsp_max_candidates,
    )
    for domain in ("evrptw", "rcpsp"):
        for case_id in expected[domain]:
            case_payload = case_payloads[domain][case_id]
            for seed in seeds:
                for axis in axes:
                    execution_mode, workers, speculation_window = _AXES[axis]
                    for budget in ("fixed_work", "fixed_time"):
                        label = (
                            f"txnopt_level1_{domain}_{case_id.lower()}_{seed}_{axis}_{budget}_"
                            f"attempt{attempt:02d}"
                        )
                        config = {
                            "schema_version": "txnopt-run-config-v1",
                            "run_label": label,
                            "output_root": str(raw_root),
                            "build_manifest": {
                                "path": str(build_manifest_source),
                                "sha256": build_manifest_sha256,
                            },
                            "run_config": {
                                "seed": seed,
                                "workers": workers,
                                "execution_mode": execution_mode,
                                "fixed_work": fixed_work if budget == "fixed_work" else None,
                                "deadline_seconds": (
                                    fixed_time_seconds if budget == "fixed_time" else None
                                ),
                                "speculation_window": speculation_window,
                                "trace_policy": "semantic_and_physical",
                                "max_rounds": max_rounds,
                            },
                            "case": {
                                **case_payload,
                                "oracle_seed": seed,
                            },
                        }
                        relative = f"configs/{label}.json"
                        config_bytes = canonical_json_bytes(config, pretty=True)
                        config_path = output / relative
                        config_path.write_bytes(config_bytes)
                        expected_identity = ExpectedEvidenceIdentity.from_plan_inputs(
                            config_path,
                            build_manifest_path=build_manifest_source,
                        )
                        identity_relative = f"expected-identities/{label}.json"
                        identity_path = output / identity_relative
                        identity_bytes = canonical_json_bytes(
                            expected_identity.to_payload(),
                            pretty=True,
                        )
                        identity_path.write_bytes(identity_bytes)
                        identity_sha256 = sha256_bytes(identity_bytes)
                        write_sidecar(identity_path, identity_sha256)
                        entries.append(
                            {
                                "path": relative,
                                "sha256": sha256_bytes(config_bytes),
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
    tree_sha256 = sha256_bytes(canonical_json_bytes(config_tree_entries))
    identity_tree_sha256 = sha256_bytes(canonical_json_bytes(identity_tree_entries))
    manifest = {
        "schema_version": "txnopt-level1-campaign-plan-v2",
        "status": "PLANNED_NOT_STARTED",
        "protocol_path": str(protocol_source),
        "protocol_sha256": _sha256_file(protocol_source),
        "catalog_path": str(catalog_source),
        "catalog_sha256": _sha256_file(catalog_source),
        "raw_output_root": str(raw_root),
        "build_manifest_path": str(build_manifest_source),
        "build_manifest_sha256": build_manifest_sha256,
        "config_count": len(entries),
        "config_tree_sha256": tree_sha256,
        "expected_identity_tree_sha256": identity_tree_sha256,
        "fixed_work": fixed_work,
        "fixed_time_seconds": fixed_time_seconds,
        "max_rounds": max_rounds,
        "max_candidates": {
            "evrptw": evrptw_max_candidates,
            "rcpsp": rcpsp_max_candidates,
        },
        "attempt": attempt,
        "holdout_opened": False,
        "cloud_purchase_authorized": False,
        "formal_matrix_started": False,
        "entries": entries,
    }
    if resource_contract is not None:
        manifest["resource_contract"] = resource_contract
        manifest["region"] = None
        manifest["region_required_live_input"] = True
    manifest_path = output / "manifest.json"
    manifest_bytes = canonical_json_bytes(manifest, pretty=True)
    manifest_path.write_bytes(manifest_bytes)
    write_sidecar(manifest_path, sha256_bytes(manifest_bytes))
    return manifest_path


def _validate_build_manifest(manifest: Mapping[str, Any]) -> None:
    producer = _object(manifest.get("producer"), "build producer")
    artifacts = _object(manifest.get("artifacts"), "build artifacts")
    native = _object(artifacts.get("native_extension"), "native extension")
    if manifest.get("schema_version") != "txnopt-level1-build-manifest-v1":
        raise ValueError("unsupported TxnOpt build manifest schema")
    if (
        producer.get("source_dirty") is not False
        or producer.get("development_override") is not False
    ):
        raise ValueError("campaign build manifest must bind a clean producer")
    if native.get("protocol") != "txnopt-native-round-v1":
        raise ValueError("campaign build manifest uses an unsupported native protocol")


def _expected_case_ids(protocol: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    domains = _object(protocol.get("domains"), "protocol domains")
    expected: dict[str, tuple[str, ...]] = {}
    for domain in ("evrptw", "rcpsp"):
        scope = _object(domains.get(domain), f"{domain} scope")
        identifiers = (*_strings(scope, "pilot"), *_strings(scope, "validation"))
        if not identifiers or len(set(identifiers)) != len(identifiers):
            raise ValueError(f"{domain} identifiers must be unique")
        if any(_IDENTIFIER.fullmatch(identifier) is None for identifier in identifiers):
            raise ValueError(f"{domain} identifier is not canonical")
        expected[domain] = tuple(identifiers)
    if set(domains) != set(expected):
        raise ValueError("protocol contains an unregistered domain")
    return expected


def _load_cases(
    paths: Mapping[str, Mapping[str, Path]],
    *,
    evrptw_max_candidates: int,
    rcpsp_max_candidates: int,
) -> dict[str, dict[str, dict[str, Any]]]:
    evrptw = {
        case_id: _evrptw_case(
            parse_schneider(path),
            path,
            max_candidates=evrptw_max_candidates,
        )
        for case_id, path in paths["evrptw"].items()
    }
    rcpsp = {
        case_id: _rcpsp_case(
            parse_psplib_sm(path),
            path,
            max_candidates=rcpsp_max_candidates,
        )
        for case_id, path in paths["rcpsp"].items()
    }
    return {"evrptw": evrptw, "rcpsp": rcpsp}


def _evrptw_case(
    instance: Instance,
    source: Path,
    *,
    max_candidates: int,
) -> dict[str, Any]:
    initial_plan = construct_initial_plan(instance)
    return {
        "domain": "evrptw",
        "backend": "native",
        "max_candidates": max_candidates,
        "source_instance_path": str(source),
        "source_instance_sha256": _sha256_file(source),
        "initialization_policy": "ortools-vrptw-vehicle-first-exact-split-v1",
        "initial_plan": initial_plan.customer_routes,
        "instance": {
            "name": instance.name,
            "nodes": [
                {
                    "name": node.name,
                    "kind": node.kind.value,
                    "x": node.x,
                    "y": node.y,
                    "demand": node.demand,
                    "ready_time": node.ready_time,
                    "due_date": node.due_date,
                    "service_time": node.service_time,
                }
                for node in instance.nodes
            ],
            "vehicle": {
                "battery_capacity": instance.vehicle.battery_capacity,
                "load_capacity": instance.vehicle.load_capacity,
                "consumption_rate": instance.vehicle.consumption_rate,
                "inverse_refueling_rate": instance.vehicle.inverse_refueling_rate,
                "average_velocity": instance.vehicle.average_velocity,
            },
        },
    }


def _rcpsp_case(
    instance: RCPSPInstance,
    source: Path,
    *,
    max_candidates: int,
) -> dict[str, Any]:
    initial = precedence_feasible_initial_state(instance)
    return {
        "domain": "rcpsp",
        "max_block_size": 3,
        "max_candidates": max_candidates,
        "source_instance_path": str(source),
        "source_instance_sha256": _sha256_file(source),
        "initial_state": {
            "activity_order": initial.activity_order,
            "mode_vector": initial.mode_vector,
        },
        "instance": {
            "name": instance.name,
            "renewable_capacities": instance.renewable_capacities,
            "activities": [
                {
                    "activity_id": activity.activity_id,
                    "predecessors": activity.predecessors,
                    "modes": [
                        {
                            "duration": mode.duration,
                            "renewable_demands": mode.renewable_demands,
                        }
                        for mode in activity.modes
                    ],
                }
                for activity in instance.activities
            ],
        },
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be an object with string keys")
    return value


def _strings(payload: Mapping[str, Any], key: str) -> Sequence[str]:
    value = payload.get(key)
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{key} must be a string list")
    return value


def _integers(payload: Mapping[str, Any], key: str) -> Sequence[int]:
    value = payload.get(key)
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in value
    ):
        raise ValueError(f"{key} must be a non-negative integer list")
    return value


def _string_value(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("catalog paths must be non-empty strings")
    return value


__all__ = ["materialize_level1_plan"]
