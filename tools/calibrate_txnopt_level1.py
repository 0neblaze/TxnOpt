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


def _run(command: list[str]) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"calibration command failed ({completed.returncode}): {completed.stderr.strip()}"
        )
    output: object = json.loads(completed.stdout)
    return _object(output, "command output"), elapsed


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
    plan = _object(json.loads(plan_path.read_bytes()), "plan manifest")
    if plan.get("schema_version") != "txnopt-level1-campaign-plan-v1":
        raise ValueError("unsupported campaign plan schema")
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

    selected: list[tuple[Path, str, str, str, int, str]] = []
    for entry in plan.get("entries", []):
        item = _object(entry, "plan entry")
        relative = item.get("path")
        if not isinstance(relative, str):
            raise ValueError("plan entry path is invalid")
        config_path = plan_path.parent / relative
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
            selected.append((config_path, domain, case_id, axis, seed, budget))
    expected_count = sum(len(values) for values in cases.values()) * len(seeds) * 4 * 2
    if len(selected) != expected_count:
        raise ValueError(
            f"calibration selected {len(selected)} configs, expected {expected_count}"
        )

    samples: list[dict[str, object]] = []
    groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    review_root.mkdir(parents=True, exist_ok=False)
    for ordinal, (config_path, domain, case_id, axis, seed, budget) in enumerate(selected, 1):
        run_output, elapsed = _run(
            [sys.executable, "-m", "txnopt_evidence.cli", "run", "--config", str(config_path)]
        )
        manifest = Path(str(run_output.get("manifest_path"))).resolve(strict=True)
        review_dir = review_root / str(run_output.get("run_label"))
        review_output, _review_elapsed = _run(
            [
                sys.executable,
                "-m",
                "txnopt_evidence.cli",
                "replay",
                str(manifest),
                "--output-dir",
                str(review_dir),
            ]
        )
        if review_output.get("status") != "PASS":
            raise RuntimeError("independent calibration replay did not pass")
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
    receipt = {
        "schema_version": "txnopt-local-runtime-calibration-v1",
        "status": "LOCAL_CALIBRATION_COMPLETE_NOT_LEVEL1_EVIDENCE",
        "plan_manifest_path": str(plan_path),
        "plan_manifest_sha256": plan_sha256,
        "representative_cases": {key: sorted(value) for key, value in cases.items()},
        "seeds": seeds,
        "p95_policy": "nearest-rank; six samples per domain-axis-budget group",
        "observations": observations,
        "samples": samples,
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
