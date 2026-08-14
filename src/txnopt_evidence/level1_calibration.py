"""Run a bounded local sample and write one signed p95 calibration receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from txnopt_evidence.codec import (
    canonical_json_bytes,
    read_signed_json,
    sha256_bytes,
    sha256_file,
    write_sidecar,
    write_signed_json,
)
from txnopt_evidence.identity import ExpectedEvidenceIdentity
from txnopt_evidence.level1_campaign_common import (
    load_campaign_plan,
    load_prebound_expected_identity,
    require_clean_repository,
    require_prebound_expected_identities,
)
from txnopt_evidence.level1_tencent_successors import (
    TencentLevel1Successor,
    require_calibration_attempt,
    successor_from_build_manifest,
)

_ATTEMPT_SUFFIX = re.compile(r"_attempt[0-9]+$")


@dataclass(frozen=True, slots=True)
class CalibrationSelection:
    source_config_path: Path
    domain: str
    case_id: str
    axis: str
    seed: int
    budget: str


@dataclass(frozen=True, slots=True)
class CalibrationExecution:
    config_path: Path
    expected_identity_path: Path
    domain: str
    case_id: str
    axis: str
    seed: int
    budget: str


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_sidecar(path: Path) -> str:
    digest = _sha256(path)
    fields = path.with_suffix(path.suffix + ".sha256").read_text().strip().split()
    if fields != [digest, path.name]:
        raise ValueError(f"signed sidecar differs: {path}")
    return digest


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be an object")
    return value


def _run(command: list[str]) -> tuple[dict[str, Any], float, int]:
    started = time.perf_counter()
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"PYTHONHOME", "PYTHONPATH"}
    }
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    )
    peak_rss_bytes = 0
    while True:
        peak_rss_bytes = max(peak_rss_bytes, _process_peak_rss_bytes(process.pid))
        try:
            stdout, stderr = process.communicate(timeout=0.01)
            break
        except subprocess.TimeoutExpired:
            continue
    elapsed = time.perf_counter() - started
    if process.returncode != 0:
        raise RuntimeError(
            f"calibration command failed ({process.returncode}): {stderr.strip()}"
        )
    output: object = json.loads(stdout)
    if peak_rss_bytes <= 0:
        raise RuntimeError("calibration command peak RSS was not observed")
    return _object(output, "command output"), elapsed, peak_rss_bytes


def _process_peak_rss_bytes(pid: int) -> int:
    status = Path(f"/proc/{pid}/status")
    try:
        lines = status.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return 0
    for line in lines:
        if line.startswith("VmHWM:"):
            fields = line.split()
            if len(fields) != 3 or fields[2] != "kB":
                raise RuntimeError("Linux VmHWM has an unexpected format")
            return int(fields[1]) * 1024
    return 0


def _materialize_calibration_plan(
    *,
    source_plan_sha256: str,
    build_manifest_path: Path,
    selections: tuple[CalibrationSelection, ...],
    destination: Path,
    raw_root: Path,
    attempt: int,
) -> tuple[Path, tuple[CalibrationExecution, ...]]:
    """Derive a prebound local-only plan without touching the formal raw root."""

    if (
        len(source_plan_sha256) != 64
        or any(character not in "0123456789abcdef" for character in source_plan_sha256)
    ):
        raise ValueError("source formal plan digest is invalid")
    build = build_manifest_path.expanduser().resolve(strict=True)
    successor = _calibration_successor(build, attempt)
    if not selections:
        raise ValueError("calibration plan requires at least one selection")
    output = destination.expanduser().resolve()
    raw_output = raw_root.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"calibration plan destination already exists: {output}")
    if raw_output.exists() or raw_output.is_symlink():
        raise FileExistsError(f"calibration raw root already exists: {raw_output}")
    build_sha256 = _verify_sidecar(build)
    configs = output / "configs"
    identities = output / "expected-identities"
    configs.mkdir(parents=True, exist_ok=False)
    identities.mkdir(exist_ok=False)
    entries: list[dict[str, object]] = []
    execution: list[CalibrationExecution] = []
    seen_labels: set[str] = set()
    for selection in selections:
        source = selection.source_config_path.resolve(strict=True)
        if source.is_symlink():
            raise ValueError("calibration source config cannot be a symlink")
        payload: object = json.loads(source.read_bytes())
        if not isinstance(payload, dict) or payload.get("schema_version") != "txnopt-run-config-v1":
            raise ValueError("calibration source config schema differs")
        original_label = payload.get("run_label")
        if not isinstance(original_label, str) or _ATTEMPT_SUFFIX.search(original_label) is None:
            raise ValueError("calibration source run label lacks an attempt suffix")
        label = _ATTEMPT_SUFFIX.sub(f"_attempt{attempt:02d}", original_label)
        if label == original_label or label in seen_labels:
            raise ValueError("calibration run label is not a unique successor")
        seen_labels.add(label)
        derived = json.loads(json.dumps(payload))
        if not isinstance(derived, dict):  # pragma: no cover - JSON object invariant
            raise TypeError("calibration config must remain an object")
        derived["run_label"] = label
        derived["output_root"] = str(raw_output)
        config_path = configs / f"{label}.json"
        config_bytes = canonical_json_bytes(derived, pretty=True)
        config_path.write_bytes(config_bytes)
        config_sha256 = sha256_bytes(config_bytes)
        identity = ExpectedEvidenceIdentity.from_plan_inputs(
            config_path,
            build_manifest_path=build,
        )
        identity_path = identities / f"{label}.json"
        identity_bytes = canonical_json_bytes(identity.to_payload(), pretty=True)
        identity_path.write_bytes(identity_bytes)
        identity_sha256 = sha256_bytes(identity_bytes)
        write_sidecar(identity_path, identity_sha256)
        entries.append(
            {
                "path": f"configs/{label}.json",
                "sha256": config_sha256,
                "expected_identity_path": f"expected-identities/{label}.json",
                "expected_identity_sha256": identity_sha256,
                "domain": selection.domain,
                "case_id": selection.case_id,
                "axis": selection.axis,
                "seed": selection.seed,
                "budget": selection.budget,
            }
        )
        execution.append(
            CalibrationExecution(
                config_path=config_path,
                expected_identity_path=identity_path,
                domain=selection.domain,
                case_id=selection.case_id,
                axis=selection.axis,
                seed=selection.seed,
                budget=selection.budget,
            )
        )
    config_tree = [
        {"path": entry["path"], "sha256": entry["sha256"]} for entry in entries
    ]
    identity_tree = [
        {
            "path": entry["expected_identity_path"],
            "sha256": entry["expected_identity_sha256"],
        }
        for entry in entries
    ]
    manifest = output / "manifest.json"
    write_signed_json(
        manifest,
        {
            "schema_version": "txnopt-local-calibration-plan-v1",
            "status": "PLANNED_NOT_STARTED",
            "build": successor.build_name,
            "attempt": attempt,
            "source_formal_plan_sha256": source_plan_sha256,
            "build_manifest_path": str(build),
            "build_manifest_sha256": build_sha256,
            "raw_output_root": str(raw_output),
            "config_count": len(entries),
            "config_tree_sha256": sha256_bytes(canonical_json_bytes(config_tree)),
            "expected_identity_tree_sha256": sha256_bytes(
                canonical_json_bytes(identity_tree)
            ),
            "entries": entries,
            "cloud_purchase_authorized": False,
            "formal_matrix_started": False,
            "holdout_opened": False,
        },
    )
    return manifest, tuple(execution)


def _load_calibration_plan(
    manifest_path: Path,
    *,
    source_plan_sha256: str,
    build_manifest_path: Path,
    raw_root: Path,
    attempt: int,
) -> tuple[Path, tuple[CalibrationExecution, ...]]:
    """Resume an already materialized plan after a pre-run launcher failure."""

    manifest = manifest_path.resolve(strict=True)
    payload = read_signed_json(manifest)
    build = build_manifest_path.resolve(strict=True)
    successor = _calibration_successor(build, attempt)
    raw_output = raw_root.resolve()
    if (
        payload.get("schema_version") != "txnopt-local-calibration-plan-v1"
        or payload.get("status") != "PLANNED_NOT_STARTED"
        or payload.get("build") != successor.build_name
        or payload.get("attempt") != attempt
        or payload.get("source_formal_plan_sha256") != source_plan_sha256
        or payload.get("build_manifest_path") != str(build)
        or payload.get("build_manifest_sha256") != _verify_sidecar(build)
        or payload.get("raw_output_root") != str(raw_output)
        or payload.get("cloud_purchase_authorized") is not False
        or payload.get("formal_matrix_started") is not False
        or payload.get("holdout_opened") is not False
    ):
        raise ValueError("materialized calibration plan identity differs")
    raw_entries = payload.get("entries")
    if not isinstance(raw_entries, list) or payload.get("config_count") != len(raw_entries):
        raise ValueError("materialized calibration plan entries differ")
    root = manifest.parent
    execution: list[CalibrationExecution] = []
    config_tree: list[dict[str, object]] = []
    identity_tree: list[dict[str, object]] = []
    for raw_entry in raw_entries:
        expected_fields = {
            "path",
            "sha256",
            "expected_identity_path",
            "expected_identity_sha256",
            "domain",
            "case_id",
            "axis",
            "seed",
            "budget",
        }
        if not isinstance(raw_entry, dict) or set(raw_entry) != expected_fields:
            raise ValueError("materialized calibration entry field set differs")
        config_relative = _relative_plan_path(raw_entry["path"])
        identity_relative = _relative_plan_path(raw_entry["expected_identity_path"])
        config_path = root / config_relative
        identity_path = root / identity_relative
        config_sha256 = sha256_file(config_path)
        identity_sha256 = sha256_file(identity_path)
        if (
            config_sha256 != raw_entry["sha256"]
            or identity_sha256 != raw_entry["expected_identity_sha256"]
            or _verify_sidecar(identity_path) != identity_sha256
        ):
            raise ValueError("materialized calibration entry digest differs")
        config = _object(json.loads(config_path.read_bytes()), "calibration config")
        label = config.get("run_label")
        if (
            not isinstance(label, str)
            or not label.endswith(f"_attempt{attempt:02d}")
            or config.get("output_root") != str(raw_output)
        ):
            raise ValueError("materialized calibration run identity differs")
        identity = ExpectedEvidenceIdentity.from_payload(read_signed_json(identity_path))
        if (
            identity.run_label != label
            or identity.input_config_sha256 != config_sha256
            or identity.config_artifact_sha256 != config_sha256
        ):
            raise ValueError("materialized calibration expected identity differs")
        domain = raw_entry["domain"]
        case_id = raw_entry["case_id"]
        axis = raw_entry["axis"]
        seed = raw_entry["seed"]
        budget = raw_entry["budget"]
        if (
            domain not in {"evrptw", "rcpsp"}
            or not isinstance(case_id, str)
            or not case_id
            or axis not in {"serial_1", "txnopt_1", "txnopt_4", "barrier_4"}
            or isinstance(seed, bool)
            or not isinstance(seed, int)
            or budget not in {"fixed_work", "fixed_time"}
        ):
            raise ValueError("materialized calibration metadata differs")
        config_tree.append({"path": config_relative, "sha256": config_sha256})
        identity_tree.append({"path": identity_relative, "sha256": identity_sha256})
        execution.append(
            CalibrationExecution(
                config_path=config_path,
                expected_identity_path=identity_path,
                domain=domain,
                case_id=case_id,
                axis=axis,
                seed=seed,
                budget=budget,
            )
        )
    if (
        sha256_bytes(canonical_json_bytes(config_tree))
        != payload.get("config_tree_sha256")
        or sha256_bytes(canonical_json_bytes(identity_tree))
        != payload.get("expected_identity_tree_sha256")
    ):
        raise ValueError("materialized calibration tree digest differs")
    return manifest, tuple(execution)


def _calibration_successor(
    build_manifest_path: Path,
    attempt: int,
) -> TencentLevel1Successor:
    build = read_signed_json(build_manifest_path)
    return require_calibration_attempt(
        successor_from_build_manifest(build),
        attempt,
    )


def _relative_plan_path(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("calibration plan path is invalid")
    candidate = PurePosixPath(value)
    if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
        raise ValueError("calibration plan path is invalid")
    return candidate.as_posix()


def _orchestration_identity(repository: Path) -> dict[str, object]:
    root = repository.expanduser().resolve(strict=True)
    require_clean_repository(root)
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tree = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD^{tree}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    module = root / "src" / "txnopt_evidence" / "level1_calibration.py"
    if module.resolve(strict=True) != Path(__file__).resolve(strict=True):
        raise RuntimeError("calibration module is not running from its bound repository")
    return {
        "repository_root": str(root),
        "revision": revision,
        "git_tree": tree,
        "source_dirty": False,
        "calibration_tool_sha256": _sha256(module),
    }


def _execution_python_path(path: Path) -> Path:
    """Validate a Python launcher without collapsing its virtualenv identity."""

    candidate = path.expanduser().absolute()
    target = candidate.resolve(strict=True)
    if not target.is_file() or not os.access(candidate, os.X_OK):
        raise ValueError("calibration Python must resolve to an executable file")
    return candidate


def _protocol_calibration_schema(protocol_path: Path) -> str:
    payload: object = json.loads(protocol_path.resolve(strict=True).read_bytes())
    if not isinstance(payload, dict):
        raise ValueError("Level 1 protocol must be an object")
    schema = payload.get("schema_version")
    if schema == "txnopt-level1-protocol-v2":
        return "txnopt-local-runtime-calibration-v2"
    if schema == "txnopt-level1-protocol-v1":
        return "txnopt-local-runtime-calibration-v1"
    raise ValueError("unsupported Level 1 protocol schema for calibration")


def materialize_attempt27_v2_correction(source: Path, output: Path) -> str:
    """Correct a completed v1-classified Attempt27 receipt without rewriting it."""

    source_path = source.resolve(strict=True)
    source_sha256 = _verify_sidecar(source_path)
    payload = read_signed_json(source_path)
    if (
        payload.get("schema_version") != "txnopt-local-runtime-calibration-v1"
        or payload.get("status") != "LOCAL_CALIBRATION_COMPLETE_NOT_LEVEL1_EVIDENCE"
        or payload.get("build") != "Build16"
        or payload.get("attempt") != 27
        or payload.get("cloud_purchase_performed") is not False
        or payload.get("formal_matrix_started") is not False
        or payload.get("holdout_opened") is not False
    ):
        raise ValueError("Attempt27 source receipt is not eligible for v2 correction")
    plan_value = payload.get("plan_manifest_path")
    if not isinstance(plan_value, str) or not plan_value:
        raise ValueError("Attempt27 source receipt lacks its formal plan")
    plan = read_signed_json(Path(plan_value))
    protocol_value = plan.get("protocol_path")
    if not isinstance(protocol_value, str) or not protocol_value:
        raise ValueError("Attempt27 formal plan lacks its protocol path")
    if (
        _protocol_calibration_schema(Path(protocol_value))
        != "txnopt-local-runtime-calibration-v2"
    ):
        raise ValueError("Attempt27 source plan does not bind protocol v2")
    samples = payload.get("samples")
    if not isinstance(samples, list) or len(samples) != 96:
        raise ValueError("Attempt27 correction requires all 96 samples")
    peak_rss_bytes = 0
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError("Attempt27 sample must be an object")
        run_peak = sample.get("run_peak_rss_bytes")
        review_peak = sample.get("review_peak_rss_bytes")
        if (
            isinstance(run_peak, bool)
            or not isinstance(run_peak, int)
            or run_peak <= 0
            or isinstance(review_peak, bool)
            or not isinstance(review_peak, int)
            or review_peak <= 0
            or sample.get("prefix_safety") != "PASS"
            or sample.get("aggregate_refinement_replay") != "PASS"
            or sample.get("fallback_count") != 0
        ):
            raise ValueError("Attempt27 sample is incomplete or failed")
        peak_rss_bytes = max(peak_rss_bytes, run_peak, review_peak)
    corrected = dict(payload)
    corrected.update(
        {
            "schema_version": "txnopt-local-runtime-calibration-v2",
            "supersedes_calibration_receipt_path": str(source_path),
            "supersedes_calibration_receipt_sha256": source_sha256,
            "correction_reason": (
                "the completed protocol-v2 sample set was classified as v1 because "
                "the aggregator inspected a nonexistent plan field"
            ),
            "raw_and_review_evidence_rerun": False,
            "peak_rss_bytes": peak_rss_bytes,
            "peak_rss_measurement": "Linux /proc/<pid>/status VmHWM",
            "memory_margin_live_host_verified": False,
        }
    )
    return write_signed_json(output, corrected)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-manifest", type=Path, required=True)
    parser.add_argument("--calibration-plan-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--review-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--orchestration-repository", type=Path, required=True)
    parser.add_argument("--attempt", type=int, default=27)
    parser.add_argument("--resume-materialized-plan", action="store_true")
    parser.add_argument("--case", action="append", required=True)
    parser.add_argument("--seed", action="append", type=int, required=True)
    arguments = parser.parse_args()

    output = arguments.output.resolve()
    if output.exists() or output.with_suffix(output.suffix + ".sha256").exists():
        raise FileExistsError(f"calibration output already exists: {output}")
    review_root = arguments.review_root.resolve()
    if review_root.exists() and (
        not arguments.resume_materialized_plan or any(review_root.iterdir())
    ):
        raise FileExistsError(f"calibration review root already exists: {review_root}")
    calibration_raw_root = arguments.raw_root.resolve()
    if calibration_raw_root.exists() or calibration_raw_root.is_symlink():
        raise FileExistsError(
            f"calibration raw root already exists: {calibration_raw_root}"
        )
    calibration_plan_root = arguments.calibration_plan_root.resolve()
    if arguments.resume_materialized_plan:
        if not calibration_plan_root.is_dir() or calibration_plan_root.is_symlink():
            raise FileNotFoundError(
                f"materialized calibration plan is unavailable: {calibration_plan_root}"
            )
    elif calibration_plan_root.exists() or calibration_plan_root.is_symlink():
        raise FileExistsError(
            f"calibration plan root already exists: {calibration_plan_root}"
        )
    execution_python = _execution_python_path(arguments.python)
    orchestration_identity = _orchestration_identity(
        arguments.orchestration_repository
    )
    plan_path = arguments.plan_manifest.resolve(strict=True)
    plan_sha256 = _verify_sidecar(plan_path)
    plan = load_campaign_plan(plan_path)
    require_prebound_expected_identities(plan)
    calibration_successor = _calibration_successor(
        plan.build_manifest_path,
        arguments.attempt,
    )
    formal_raw_root = plan.raw_output_root.resolve()
    if formal_raw_root.exists() or formal_raw_root.is_symlink():
        raise FileExistsError(
            f"formal raw root must remain absent during calibration: {formal_raw_root}"
        )
    if calibration_raw_root == formal_raw_root:
        raise ValueError("calibration raw root must differ from the formal raw root")
    cases: dict[str, set[str]] = defaultdict(set)
    for raw_case in arguments.case:
        if raw_case.count(":") != 1:
            raise ValueError("cases must use domain:case_id")
        domain, case_id = raw_case.split(":")
        if domain not in {"evrptw", "rcpsp"} or not case_id:
            raise ValueError("calibration case is invalid")
        cases[domain].add(case_id)
    if set(cases) != {"evrptw", "rcpsp"} or any(len(values) < 2 for values in cases.values()):
        raise ValueError("calibration requires at least two cases per domain")
    seeds = tuple(arguments.seed)
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("calibration requires at least three unique seeds")

    selected: list[CalibrationSelection] = []
    for entry in plan.entries:
        load_prebound_expected_identity(entry)
        config_path = entry.config_path
        config = _object(json.loads(config_path.read_bytes()), "run config")
        case = _object(config.get("case"), "case")
        run_config = _object(config.get("run_config"), "run config payload")
        domain = case.get("domain")
        case_id = Path(str(case.get("source_instance_path", ""))).stem
        seed = run_config.get("seed")
        axis = _axis(run_config)
        budget = "fixed_work" if run_config.get("fixed_work") is not None else "fixed_time"
        if (
            isinstance(domain, str)
            and domain in cases
            and case_id in cases[domain]
            and isinstance(seed, int)
            and seed in seeds
        ):
            selected.append(
                CalibrationSelection(
                    source_config_path=config_path,
                    domain=domain,
                    case_id=case_id,
                    axis=axis,
                    seed=seed,
                    budget=budget,
                )
            )
    expected_count = sum(len(values) for values in cases.values()) * len(seeds) * 4 * 2
    if len(selected) != expected_count:
        raise ValueError(
            f"calibration selected {len(selected)} configs, expected {expected_count}"
        )

    if arguments.resume_materialized_plan:
        calibration_plan_path, execution = _load_calibration_plan(
            calibration_plan_root / "manifest.json",
            source_plan_sha256=plan_sha256,
            build_manifest_path=plan.build_manifest_path,
            raw_root=calibration_raw_root,
            attempt=arguments.attempt,
        )
    else:
        calibration_plan_path, execution = _materialize_calibration_plan(
            source_plan_sha256=plan_sha256,
            build_manifest_path=plan.build_manifest_path,
            selections=tuple(selected),
            destination=calibration_plan_root,
            raw_root=calibration_raw_root,
            attempt=arguments.attempt,
        )
    requested_keys = {
        (item.domain, item.case_id, item.axis, item.seed, item.budget)
        for item in selected
    }
    execution_keys = {
        (item.domain, item.case_id, item.axis, item.seed, item.budget)
        for item in execution
    }
    if len(execution) != expected_count or execution_keys != requested_keys:
        raise ValueError("materialized calibration selection differs")
    calibration_plan_sha256 = _verify_sidecar(calibration_plan_path)

    samples: list[dict[str, object]] = []
    groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    maximum_peak_rss_bytes = 0
    review_root.mkdir(parents=True, exist_ok=arguments.resume_materialized_plan)
    for ordinal, item in enumerate(execution, 1):
        run_output, elapsed, run_peak_rss_bytes = _run(
            [
                str(execution_python),
                "-m",
                "txnopt_evidence.cli",
                "run",
                "--config",
                str(item.config_path),
            ]
        )
        manifest = Path(str(run_output.get("manifest_path"))).resolve(strict=True)
        review_dir = review_root / str(run_output.get("run_label"))
        review_output, _review_elapsed, review_peak_rss_bytes = _run(
            [
                str(execution_python),
                "-m",
                "txnopt_evidence.review_cli",
                str(manifest),
                "--output-dir",
                str(review_dir),
                "--expected-identity",
                str(item.expected_identity_path),
            ]
        )
        if (
            review_output.get("status") != "PASS"
            or review_output.get("prefix_safety") != "PASS"
            or review_output.get("aggregate_refinement_replay") != "PASS"
            or review_output.get("fallback_count") != 0
        ):
            raise RuntimeError("independent calibration replay did not pass")
        maximum_peak_rss_bytes = max(
            maximum_peak_rss_bytes,
            run_peak_rss_bytes,
            review_peak_rss_bytes,
        )
        key = (item.domain, item.axis, item.budget)
        groups[key].append(elapsed)
        samples.append(
            {
                "ordinal": ordinal,
                "domain": item.domain,
                "case_id": item.case_id,
                "seed": item.seed,
                "axis": item.axis,
                "budget": item.budget,
                "elapsed_seconds": elapsed,
                "run_peak_rss_bytes": run_peak_rss_bytes,
                "review_peak_rss_bytes": review_peak_rss_bytes,
                "raw_manifest_path": str(manifest),
                "raw_manifest_sha256": str(run_output.get("manifest_sha256")),
                "review_path": str(review_dir / "review.json"),
                "review_sha256": _verify_sidecar(review_dir / "review.json"),
                "semantic_digest": str(review_output.get("semantic_digest")),
                "objective": review_output.get("case_replay"),
                "prefix_safety": review_output.get("prefix_safety"),
                "aggregate_refinement_replay": review_output.get(
                    "aggregate_refinement_replay"
                ),
                "physical_event_count": review_output.get("physical_event_count"),
                "fallback_count": review_output.get("fallback_count"),
                "expected_identity_sha256": _verify_sidecar(
                    item.expected_identity_path
                ),
            }
        )
        print(
            f"[{ordinal}/{len(execution)}] {item.domain} {item.case_id} {item.seed} "
            f"{item.axis} {item.budget}: {elapsed:.3f}s"
        )

    if formal_raw_root.exists() or formal_raw_root.is_symlink():
        raise RuntimeError("local calibration contaminated the formal raw root")

    expected_keys = {
        (domain, axis, budget)
        for domain in ("evrptw", "rcpsp")
        for axis in ("serial_1", "txnopt_1", "txnopt_4", "barrier_4")
        for budget in ("fixed_work", "fixed_time")
    }
    if set(groups) != expected_keys:
        raise RuntimeError("calibration group identity set differs")
    observations = []
    for domain, axis, budget in sorted(groups):
        values = sorted(groups[(domain, axis, budget)])
        index = math.ceil(0.95 * len(values)) - 1
        observations.append(
            {
                "domain": domain,
                "axis": axis,
                "budget": budget,
                "sample_count": len(values),
                "seconds_per_run_p95": values[index],
            }
        )
    schema_version = _protocol_calibration_schema(plan.protocol_path)
    receipt = {
        "schema_version": schema_version,
        "status": "LOCAL_CALIBRATION_COMPLETE_NOT_LEVEL1_EVIDENCE",
        "plan_manifest_path": str(plan_path),
        "plan_manifest_sha256": plan_sha256,
        "calibration_plan_path": str(calibration_plan_path),
        "calibration_plan_sha256": calibration_plan_sha256,
        "raw_output_root": str(calibration_raw_root),
        "review_root": str(review_root),
        "build": calibration_successor.build_name,
        "attempt": arguments.attempt,
        "orchestration_identity": orchestration_identity,
        "representative_cases": {key: sorted(value) for key, value in cases.items()},
        "seeds": seeds,
        "p95_policy": "nearest-rank; six samples per domain-axis-budget group",
        "observations": observations,
        "samples": samples,
        **(
            {
                "peak_rss_bytes": maximum_peak_rss_bytes,
                "peak_rss_measurement": "Linux /proc/<pid>/status VmHWM",
                "memory_margin_live_host_verified": False,
            }
            if schema_version == "txnopt-local-runtime-calibration-v2"
            else {}
        ),
        "cloud_purchase_performed": False,
        "formal_matrix_started": False,
        "holdout_opened": False,
        "formal_raw_root_remained_absent": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
    output.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{digest}  {output.name}\n", encoding="utf-8"
    )
    print(digest)
    return 0


def _axis(config: dict[str, Any]) -> str:
    mode = config.get("execution_mode")
    workers = config.get("workers")
    key = (mode, workers)
    axes = {
        ("serial", 1): "serial_1",
        ("ordered", 1): "txnopt_1",
        ("ordered", 4): "txnopt_4",
        ("barrier", 4): "barrier_4",
    }
    if key not in axes:
        raise ValueError("run config is not a Level 1 formal axis")
    return axes[key]


if __name__ == "__main__":
    raise SystemExit(main())
