"""Run a bounded local sample and write one signed p95 calibration receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from txnopt_evidence.level1_campaign_common import (
    load_campaign_plan,
    load_prebound_expected_identity,
    require_prebound_expected_identities,
)


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
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-manifest", type=Path, required=True)
    parser.add_argument("--review-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", action="append", required=True)
    parser.add_argument("--seed", action="append", type=int, required=True)
    arguments = parser.parse_args()

    output = arguments.output.resolve()
    if output.exists() or output.with_suffix(output.suffix + ".sha256").exists():
        raise FileExistsError(f"calibration output already exists: {output}")
    review_root = arguments.review_root.resolve()
    if review_root.exists():
        raise FileExistsError(f"calibration review root already exists: {review_root}")
    plan_path = arguments.plan_manifest.resolve(strict=True)
    plan_sha256 = _verify_sidecar(plan_path)
    plan = load_campaign_plan(plan_path)
    require_prebound_expected_identities(plan)
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

    selected: list[tuple[Path, Path, str, str, str, int, str]] = []
    for entry in plan.entries:
        load_prebound_expected_identity(entry)
        if entry.expected_identity_path is None:  # guarded above; keeps typing exact
            raise ValueError("plan entry lacks its expected identity")
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
                (
                    config_path,
                    entry.expected_identity_path,
                    domain,
                    case_id,
                    axis,
                    seed,
                    budget,
                )
            )
    expected_count = sum(len(values) for values in cases.values()) * len(seeds) * 4 * 2
    if len(selected) != expected_count:
        raise ValueError(
            f"calibration selected {len(selected)} configs, expected {expected_count}"
        )

    samples: list[dict[str, object]] = []
    groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    maximum_peak_rss_bytes = 0
    review_root.mkdir(parents=True, exist_ok=False)
    for ordinal, (
        config_path,
        expected_identity_path,
        domain,
        case_id,
        axis,
        seed,
        budget,
    ) in enumerate(selected, 1):
        run_output, elapsed, run_peak_rss_bytes = _run(
            [sys.executable, "-m", "txnopt_evidence.cli", "run", "--config", str(config_path)]
        )
        manifest = Path(str(run_output.get("manifest_path"))).resolve(strict=True)
        review_dir = review_root / str(run_output.get("run_label"))
        review_output, _review_elapsed, review_peak_rss_bytes = _run(
            [
                sys.executable,
                "-m",
                "txnopt_evidence.review_cli",
                str(manifest),
                "--output-dir",
                str(review_dir),
                "--expected-identity",
                str(expected_identity_path),
            ]
        )
        if review_output.get("status") != "PASS":
            raise RuntimeError("independent calibration replay did not pass")
        maximum_peak_rss_bytes = max(
            maximum_peak_rss_bytes,
            run_peak_rss_bytes,
            review_peak_rss_bytes,
        )
        key = (domain, axis, budget)
        groups[key].append(elapsed)
        samples.append(
            {
                "ordinal": ordinal,
                "domain": domain,
                "case_id": case_id,
                "seed": seed,
                "axis": axis,
                "budget": budget,
                "elapsed_seconds": elapsed,
                "run_peak_rss_bytes": run_peak_rss_bytes,
                "review_peak_rss_bytes": review_peak_rss_bytes,
                "raw_manifest_path": str(manifest),
                "raw_manifest_sha256": str(run_output.get("manifest_sha256")),
                "review_path": str(review_dir / "review.json"),
                "review_sha256": _verify_sidecar(review_dir / "review.json"),
                "semantic_digest": str(review_output.get("semantic_digest")),
            }
        )
        print(
            f"[{ordinal}/{len(selected)}] {domain} {case_id} {seed} "
            f"{axis} {budget}: {elapsed:.3f}s"
        )

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
    protocol_schema = plan.payload.get("protocol_schema_version")
    schema_version = (
        "txnopt-local-runtime-calibration-v2"
        if protocol_schema == "txnopt-level1-protocol-v2"
        else "txnopt-local-runtime-calibration-v1"
    )
    receipt = {
        "schema_version": schema_version,
        "status": "LOCAL_CALIBRATION_COMPLETE_NOT_LEVEL1_EVIDENCE",
        "plan_manifest_path": str(plan_path),
        "plan_manifest_sha256": plan_sha256,
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
