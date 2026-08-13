"""Preflight or execute the raw-only TxnOpt Level 1 campaign.

Execution is impossible without a separate signed procurement authorization.
The tool never reviews runs and never emits a readiness decision.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from tools.txnopt_level1_campaign_common import (
    CampaignEntry,
    active_campaign_processes,
    campaign_claim_path,
    executable_path,
    linux_host_identity,
    load_analysis_protocol,
    load_campaign_plan,
    load_prebound_expected_identity,
    memory_gib,
    physical_core_count,
    read_signed_object,
    require_clean_repository,
    require_prebound_expected_identities,
    run_isolated_process,
    sha256_file,
    validate_authorization,
    verify_runtime_installation,
    verify_sidecar,
    write_signed_object,
)


class _WeightedTokens:
    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("core-token capacity must be positive")
        self._capacity = capacity
        self._available = capacity
        self._condition = threading.Condition()

    def acquire(self, weight: int) -> None:
        if weight <= 0 or weight > self._capacity:
            raise ValueError("run worker request exceeds the usable-core budget")
        with self._condition:
            self._condition.wait_for(lambda: self._available >= weight)
            self._available -= weight

    def release(self, weight: int) -> None:
        with self._condition:
            self._available += weight
            if self._available > self._capacity:
                raise RuntimeError("core-token ledger overflow")
            self._condition.notify_all()


def preflight_campaign(
    plan_path: Path,
    analysis_protocol_path: Path,
    *,
    python: Path | None = None,
    wheel: Path | None = None,
) -> dict[str, Any]:
    plan = load_campaign_plan(plan_path)
    require_prebound_expected_identities(plan)
    _analysis, analysis_sha256 = load_analysis_protocol(
        analysis_protocol_path,
        plan=plan,
    )
    if plan.raw_output_root.exists() or plan.raw_output_root.is_symlink():
        raise FileExistsError(
            f"formal raw root already exists; use a new attempt: {plan.raw_output_root}"
        )
    claim_path = campaign_claim_path(plan)
    if claim_path.exists() or claim_path.is_symlink():
        raise FileExistsError(f"formal campaign launch is already claimed: {claim_path}")
    active = active_campaign_processes(plan)
    if active:
        raise RuntimeError(f"formal campaign already has active target processes: {active}")
    runtime_identity: dict[str, Any] | None = None
    if (python is None) != (wheel is None):
        raise ValueError("runtime preflight requires both --python and --wheel")
    if python is not None and wheel is not None:
        runtime_identity = verify_runtime_installation(plan, python=python, wheel=wheel)
    return {
        "schema_version": "txnopt-level1-campaign-preflight-v1",
        "status": "PASS_NOT_AUTHORIZED_TO_EXECUTE",
        "plan_manifest_path": str(plan.manifest_path),
        "plan_manifest_sha256": plan.manifest_sha256,
        "analysis_protocol_path": str(analysis_protocol_path.resolve(strict=True)),
        "analysis_protocol_sha256": analysis_sha256,
        "config_count": len(plan.entries),
        "config_tree_sha256": plan.payload["config_tree_sha256"],
        "expected_identity_tree_sha256": plan.expected_identity_tree_sha256,
        "raw_output_root": str(plan.raw_output_root),
        "raw_output_root_absent": True,
        "atomic_launch_claim_absent": True,
        "active_target_process_count": 0,
        "holdout_opened": False,
        "cloud_purchase_authorized": False,
        "formal_matrix_started": False,
        "runtime_identity": runtime_identity,
    }


def execute_campaign(
    plan_path: Path,
    analysis_protocol_path: Path,
    *,
    authorization_path: Path,
    python: Path,
    wheel: Path,
    execution_receipt_path: Path,
    usable_cores: int,
    per_run_timeout_seconds: float,
) -> dict[str, Any]:
    if usable_cores <= 0 or per_run_timeout_seconds <= 0:
        raise ValueError("usable cores and per-run timeout must be positive")
    plan = load_campaign_plan(plan_path)
    require_prebound_expected_identities(plan)
    analysis, analysis_sha256 = load_analysis_protocol(
        analysis_protocol_path,
        plan=plan,
    )
    authorization = read_signed_object(
        authorization_path,
        schema_version="txnopt-level1-procurement-authorization-v1",
    )
    authorization_sha256 = verify_sidecar(authorization_path)
    validate_authorization(authorization, plan, analysis_sha256)
    if plan.raw_output_root.exists() or plan.raw_output_root.is_symlink():
        raise FileExistsError(
            f"formal raw root already exists; failed attempts are immutable: {plan.raw_output_root}"
        )
    claim_path = campaign_claim_path(plan)
    if claim_path.exists() or claim_path.is_symlink():
        raise FileExistsError(f"formal campaign launch is already claimed: {claim_path}")
    active = active_campaign_processes(plan)
    if active:
        raise RuntimeError(f"formal campaign already has active target processes: {active}")
    resources = _object(analysis.get("resource_environment"), "resource environment")
    physical_cores = _physical_core_count()
    memory_gib = _memory_gib()
    minimum_cores = int(resources["minimum_physical_cores"])
    minimum_memory = int(resources["minimum_memory_gib"])
    if physical_cores < minimum_cores or memory_gib < minimum_memory:
        raise RuntimeError("host does not satisfy the preregistered physical resource floor")
    if usable_cores > physical_cores:
        raise ValueError("usable-core budget exceeds detected physical cores")
    runtime_identity = verify_runtime_installation(plan, python=python, wheel=wheel)
    tool_identity = _tool_identity()
    python_path = executable_path(python)
    host_identity = linux_host_identity(
        physical_cores=physical_cores,
        memory_gib=memory_gib,
        usable_core_tokens=usable_cores,
        exclusive_linux=bool(authorization["exclusive_linux"]),
    )
    started_datetime = datetime.now(UTC)
    started_at = started_datetime.isoformat().replace("+00:00", "Z")
    started_ns = time.monotonic_ns()
    maximum_window_days = int(authorization["maximum_window_days"])
    global_deadline_ns = started_ns + maximum_window_days * 86_400 * 1_000_000_000
    global_deadline_utc = (
        (started_datetime + timedelta(days=maximum_window_days)).isoformat().replace("+00:00", "Z")
    )
    claim_payload = {
        "schema_version": "txnopt-level1-campaign-launch-claim-v2",
        "status": "CLAIMED_BEFORE_RAW_WRITE",
        "plan_manifest_sha256": plan.manifest_sha256,
        "analysis_protocol_sha256": analysis_sha256,
        "authorization_sha256": authorization_sha256,
        "raw_output_root": str(plan.raw_output_root),
        "expected_identity_tree_sha256": plan.expected_identity_tree_sha256,
        "tool_identity": tool_identity,
        "runtime_identity": runtime_identity,
        "host": host_identity,
        "claimed_at_utc": started_at,
        "global_deadline_utc": global_deadline_utc,
        "active_target_process_count_before_claim": 0,
    }
    claim_path.mkdir(parents=True, exist_ok=False)
    claim_receipt_path = claim_path / "claim.json"
    claim_sha256 = write_signed_object(claim_receipt_path, claim_payload)
    stop = threading.Event()
    tokens = _WeightedTokens(usable_cores)
    records: list[dict[str, Any] | None] = [None] * len(plan.entries)

    def execute_one(entry: CampaignEntry) -> dict[str, Any]:
        if stop.is_set():
            return _skipped_record(entry)
        tokens.acquire(entry.workers)
        try:
            if stop.is_set():
                return _skipped_record(entry)
            remaining_seconds = (global_deadline_ns - time.monotonic_ns()) / 1_000_000_000
            if remaining_seconds <= 0:
                stop.set()
                return {
                    **_entry_identity(entry),
                    "status": "FAILED",
                    "error_type": "CampaignWindowExpired",
                    "error": "the signed 14-day campaign window expired",
                }
            return _run_one(
                entry,
                python=python_path,
                timeout_seconds=min(per_run_timeout_seconds, remaining_seconds),
            )
        finally:
            tokens.release(entry.workers)

    with ThreadPoolExecutor(max_workers=min(usable_cores, len(plan.entries))) as executor:
        futures: dict[Future[dict[str, Any]], CampaignEntry] = {
            executor.submit(execute_one, entry): entry for entry in plan.entries
        }
        for future in as_completed(futures):
            entry = futures[future]
            try:
                record = future.result()
            except Exception as error:  # pragma: no cover - defensive scheduler boundary
                record = {
                    **_entry_identity(entry),
                    "status": "FAILED",
                    "error_type": f"{type(error).__module__}.{type(error).__qualname__}",
                    "error": str(error),
                }
            records[entry.ordinal - 1] = record
            if record["status"] == "FAILED":
                stop.set()

    finalized = [record for record in records if record is not None]
    completed_count = sum(record["status"] == "COMPLETE" for record in finalized)
    failed_count = sum(record["status"] == "FAILED" for record in finalized)
    skipped_count = sum(record["status"] == "NOT_STARTED_AFTER_FAILURE" for record in finalized)
    status = "COMPLETE_RAW_ONLY_NOT_REVIEWED" if completed_count == len(plan.entries) else "FAILED"
    ended_at = _utc_now()
    elapsed_seconds = (time.monotonic_ns() - started_ns) / 1_000_000_000
    global_window_satisfied = (
        time.monotonic_ns() <= global_deadline_ns
        and elapsed_seconds <= maximum_window_days * 86_400
    )
    if not global_window_satisfied:
        status = "FAILED"
    receipt = {
        "schema_version": "txnopt-level1-campaign-execution-v2",
        "status": status,
        "plan_manifest_path": str(plan.manifest_path),
        "plan_manifest_sha256": plan.manifest_sha256,
        "analysis_protocol_path": str(analysis_protocol_path.resolve(strict=True)),
        "analysis_protocol_sha256": analysis_sha256,
        "expected_identity_tree_sha256": plan.expected_identity_tree_sha256,
        "authorization_path": str(authorization_path.resolve(strict=True)),
        "authorization_sha256": authorization_sha256,
        "orchestration_identity": tool_identity,
        "producer_runtime_identity": runtime_identity,
        "launch_claim_path": str(claim_receipt_path),
        "launch_claim_sha256": claim_sha256,
        "host": host_identity,
        "started_at_utc": started_at,
        "global_deadline_utc": global_deadline_utc,
        "maximum_window_days": maximum_window_days,
        "ended_at_utc": ended_at,
        "elapsed_seconds": elapsed_seconds,
        "global_window_satisfied": global_window_satisfied,
        "planned_run_count": len(plan.entries),
        "completed_run_count": completed_count,
        "failed_run_count": failed_count,
        "not_started_run_count": skipped_count,
        "runs": finalized,
        "runner_boundary": "RAW_ONLY",
        "independent_review_performed": False,
        "readiness_decision": None,
        "fallback_count": 0,
        "holdout_opened": False,
    }
    write_signed_object(execution_receipt_path.resolve(), receipt)
    return receipt


def _run_one(
    entry: CampaignEntry,
    *,
    python: Path,
    timeout_seconds: float,
) -> dict[str, Any]:
    expected_identity = load_prebound_expected_identity(entry)
    started_at = _utc_now()
    started_ns = time.monotonic_ns()
    completed = run_isolated_process(
        [
            str(python),
            "-I",
            "-m",
            "txnopt_evidence.cli",
            "run",
            "--config",
            str(entry.config_path),
        ],
        cwd=entry.config_path.parent,
        timeout_seconds=timeout_seconds,
    )
    if completed.timed_out:
        return {
            **_entry_identity(entry),
            "status": "FAILED",
            "error_type": "subprocess.TimeoutExpired",
            "error": "run process group exceeded its timeout and was terminated",
            "descendant_cleanup_performed": completed.descendant_cleanup_performed,
            "descendant_processes_remaining": list(completed.descendant_processes_remaining),
            "started_at_utc": started_at,
            "ended_at_utc": _utc_now(),
            "elapsed_seconds": (time.monotonic_ns() - started_ns) / 1_000_000_000,
        }
    if completed.descendant_cleanup_performed:
        return {
            **_entry_identity(entry),
            "status": "FAILED",
            "error_type": "SurvivingDescendantProcess",
            "error": "run parent exited while a descendant remained active",
            "descendant_cleanup_performed": True,
            "descendant_processes_remaining": list(completed.descendant_processes_remaining),
            "started_at_utc": started_at,
            "ended_at_utc": _utc_now(),
            "elapsed_seconds": (time.monotonic_ns() - started_ns) / 1_000_000_000,
        }
    base = {
        **_entry_identity(entry),
        "started_at_utc": started_at,
        "ended_at_utc": _utc_now(),
        "elapsed_seconds": (time.monotonic_ns() - started_ns) / 1_000_000_000,
        "return_code": completed.returncode,
        "descendant_cleanup_performed": completed.descendant_cleanup_performed,
        "descendant_processes_remaining": list(completed.descendant_processes_remaining),
    }
    if completed.returncode != 0:
        return {
            **base,
            "status": "FAILED",
            "stderr_tail": completed.stderr[-4000:],
            "stdout_tail": completed.stdout[-1000:],
        }
    output = _object(json.loads(completed.stdout), "run command output")
    expected_manifest = entry.raw_output_root / entry.run_label / "manifest.json"
    manifest_path = Path(str(output.get("manifest_path"))).resolve(strict=True)
    if (
        output.get("schema_version") != "txnopt-run-command-v1"
        or output.get("status") != "complete"
        or output.get("run_label") != entry.run_label
        or output.get("fallback_count") != 0
        or manifest_path != expected_manifest
    ):
        return {**base, "status": "FAILED", "error": "run command output differs"}
    manifest_sha256 = verify_sidecar(manifest_path)
    if output.get("manifest_sha256") != manifest_sha256:
        return {**base, "status": "FAILED", "error": "raw manifest digest differs"}
    raw_manifest = read_signed_object(manifest_path)
    if raw_manifest.get("schema_version") != "txnopt-raw-artifact-v3":
        return {**base, "status": "FAILED", "error": "raw artifact schema differs"}
    bundle_config = manifest_path.parent / "config.json"
    if (
        raw_manifest.get("run_label") != entry.run_label
        or raw_manifest.get("input_config_sha256") != entry.config_sha256
        or verify_sidecar(bundle_config) != expected_identity.config_artifact_sha256
        or raw_manifest.get("producer_identity")
        != dict(expected_identity.producer_identity)
    ):
        return {
            **base,
            "status": "FAILED",
            "error": "raw bundle config identity differs from the campaign plan",
        }
    return {
        **base,
        "status": "COMPLETE",
        "raw_manifest_path": str(manifest_path),
        "raw_manifest_sha256": manifest_sha256,
    }


def _tool_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
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
    common = root / "tools" / "txnopt_level1_campaign_common.py"
    return {
        "repository_root": str(root),
        "revision": revision,
        "git_tree": tree,
        "source_dirty": False,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "common_sha256": sha256_file(common),
    }


def _physical_core_count() -> int:
    return physical_core_count()


def _memory_gib() -> int:
    return memory_gib()


def _entry_identity(entry: CampaignEntry) -> dict[str, Any]:
    return {
        "ordinal": entry.ordinal,
        "run_label": entry.run_label,
        "config_path": str(entry.config_path),
        "config_sha256": entry.config_sha256,
        "expected_identity_relative_path": entry.expected_identity_relative_path,
        "expected_identity_path": (
            str(entry.expected_identity_path)
            if entry.expected_identity_path is not None
            else None
        ),
        "expected_identity_sha256": entry.expected_identity_sha256,
        "domain": entry.domain,
        "case_id": entry.case_id,
        "seed": entry.seed,
        "axis": entry.axis,
        "budget": entry.budget,
        "workers": entry.workers,
    }


def _skipped_record(entry: CampaignEntry) -> dict[str, Any]:
    return {**_entry_identity(entry), "status": "NOT_STARTED_AFTER_FAILURE"}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be an object")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight")
    preflight.add_argument("--plan-manifest", type=Path, required=True)
    preflight.add_argument("--analysis-protocol", type=Path, required=True)
    preflight.add_argument("--python", type=Path)
    preflight.add_argument("--wheel", type=Path)
    run = commands.add_parser("run")
    run.add_argument("--plan-manifest", type=Path, required=True)
    run.add_argument("--analysis-protocol", type=Path, required=True)
    run.add_argument("--authorization-receipt", type=Path, required=True)
    run.add_argument("--python", type=Path, required=True)
    run.add_argument("--wheel", type=Path, required=True)
    run.add_argument("--execution-receipt", type=Path, required=True)
    run.add_argument("--usable-cores", type=int, required=True)
    run.add_argument("--per-run-timeout-seconds", type=float, default=600.0)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    if arguments.command == "preflight":
        require_clean_repository(Path(__file__).resolve().parents[1])
        result = preflight_campaign(
            arguments.plan_manifest,
            arguments.analysis_protocol,
            python=arguments.python,
            wheel=arguments.wheel,
        )
    else:
        result = execute_campaign(
            arguments.plan_manifest,
            arguments.analysis_protocol,
            authorization_path=arguments.authorization_receipt,
            python=arguments.python,
            wheel=arguments.wheel,
            execution_receipt_path=arguments.execution_receipt,
            usable_cores=arguments.usable_cores,
            per_run_timeout_seconds=arguments.per_run_timeout_seconds,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
