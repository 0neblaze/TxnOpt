"""Independent review CLI for Stage 5.1 best-known values evidence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from evrptw.artifacts import verify_manifest
from evrptw.best_known import (
    BEST_KNOWN_VALUES,
    TOTAL_INSTANCES,
)
from evrptw.experiments.stage051_best_known import (
    BKS_DATA_FIELDS,
    COMPATIBILITY_FIELDS,
    STAGE051_SCHEMA_VERSION,
    canonical_stage051_rows,
    verify_stage04_prerequisite,
)

READY_FOR_STAGE05_2 = "READY_FOR_STAGE05_2"
NOT_READY = "NOT_READY"

_GATE_COVERAGE = "instance_coverage"
_GATE_BKS_VALUES = "bks_values_present"
_GATE_NO_GAP = "no_gap_computation"
_GATE_COMPATIBILITY = "compatibility_assessment_correct"
_GATE_REPLAY = "replay_consistency"

_GATE_NAMES = (
    _GATE_COVERAGE,
    _GATE_BKS_VALUES,
    _GATE_NO_GAP,
    _GATE_COMPATIBILITY,
    _GATE_REPLAY,
)


def review_stage051(
    *,
    run_dir: Path,
    output_dir: Path,
) -> dict[str, Path]:
    """Re-read raw BKS evidence and evaluate gates."""

    output_dir.mkdir(parents=True, exist_ok=True)
    findings: list[dict[str, str]] = []
    gate_results: dict[str, tuple[bool, str]] = {}

    if run_dir.parent.name != "results":
        return _write_failure(output_dir, "run directory is not results/<canonical-run-label>")

    # --- Manifest verification ---
    try:
        verify_manifest(run_dir)
        manifest_ok = True
    except Exception as exc:
        manifest_ok = False
        findings.append({"gate": "manifest", "status": "FAIL", "detail": str(exc)})

    if not manifest_ok:
        return _write_failure(output_dir, "manifest verification failed")

    # --- Read manifest ---
    manifest_path = run_dir / "control" / _find_manifest_name(run_dir)
    if not manifest_path.exists():
        # Try alternate location
        for child in run_dir.iterdir():
            if child.name.endswith("_manifest.json") and child.parent.name == "control":
                manifest_path = child
                break

    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        return _write_failure(output_dir, "manifest not found")

    stage_id = manifest.get("stage_id", "")
    component = manifest.get("component", "")
    if stage_id != "stage05.1" or component != "best_known":
        return _write_failure(
            output_dir,
            f"manifest stage/component mismatch: {stage_id}/{component}",
        )

    run_label = manifest.get("run_label", run_dir.name)
    if run_label != run_dir.name:
        return _write_failure(output_dir, "manifest run label does not match run directory")
    metadata_refs = [
        item
        for item in manifest.get("artifacts", [])
        if isinstance(item, dict) and item.get("artifact_type") == "manifest_metadata"
    ]
    if len(metadata_refs) != 1:
        return _write_failure(output_dir, "exactly one manifest metadata artifact is required")
    metadata_path = run_dir / str(metadata_refs[0]["relative_path"])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != STAGE051_SCHEMA_VERSION:
        return _write_failure(output_dir, "Stage 5.1 v2 metadata is required")
    if metadata.get("git_dirty") is not False:
        return _write_failure(output_dir, "Stage 5.1 evidence must come from a clean revision")
    try:
        prerequisite = verify_stage04_prerequisite(
            run_dir.parents[1],
            Path("experiments/manifests/stage04_adaptive_weights_artifact_manifest.json"),
        )
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as error:
        return _write_failure(output_dir, f"Stage 4 prerequisite failed: {error}")
    if metadata.get("stage04_prerequisite") != prerequisite:
        return _write_failure(output_dir, "Stage 4 prerequisite provenance mismatch")

    # --- Find and read BKS data CSV ---
    bks_csv_name = f"{run_label}_bks_data.csv"
    bks_path = run_dir / bks_csv_name
    if not bks_path.exists():
        return _write_failure(output_dir, f"BKS data CSV not found: {bks_path}")

    bks_rows: list[dict[str, str]] = []
    with bks_path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        bks_rows = list(reader)

    # --- Find and read compatibility CSV ---
    compat_csv_name = f"{run_label}_compatibility_assessment.csv"
    compat_path = run_dir / compat_csv_name
    compat_rows: list[dict[str, str]] = []
    if compat_path.exists():
        with compat_path.open(encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            compat_rows = list(reader)

    # --- Gate 1: Instance coverage ---
    csv_instances = {row.get("instance", "") for row in bks_rows}
    canonical_instances = {rec.instance for rec in BEST_KNOWN_VALUES}
    missing = canonical_instances - csv_instances
    extra = csv_instances - canonical_instances
    coverage_ok = (
        len(bks_rows) == TOTAL_INSTANCES
        and len(csv_instances) == TOTAL_INSTANCES
        and len(missing) == 0
        and len(extra) == 0
    )
    detail = f"csv={len(csv_instances)} canonical={len(canonical_instances)}"
    if missing:
        detail += f" missing={sorted(missing)[:5]}"
    if extra:
        detail += f" extra={sorted(extra)[:5]}"
    gate_results[_GATE_COVERAGE] = (coverage_ok, detail)

    # --- Gate 2: BKS values present ---
    missing_values = [
        row["instance"]
        for row in bks_rows
        if row.get("bks_vehicles") in ("unknown", "", None)
        or row.get("bks_distance") in ("unknown", "", None)
    ]
    bks_ok = len(missing_values) == 0
    detail = (
        f"all {len(bks_rows)} instances have bks_vehicles and bks_distance"
        if bks_ok
        else f"missing BKS values: {missing_values[:5]}"
    )
    gate_results[_GATE_BKS_VALUES] = (bks_ok, detail)

    # --- Gate 3: No gap computation ---
    gap_columns = [k for k in bks_rows[0] if "gap" in k.lower()] if bks_rows else []
    no_gap_ok = len(gap_columns) == 0
    gate_results[_GATE_NO_GAP] = (
        no_gap_ok,
        f"gap columns found: {gap_columns}" if gap_columns else "no gap columns",
    )

    # --- Gate 4: Compatibility assessment ---
    strict_rows_ok, strict_detail = validate_stage051_rows(bks_rows, compat_rows)
    compat_ok = _check_compatibility(compat_rows)
    gate_results[_GATE_COMPATIBILITY] = (
        compat_ok,
        strict_detail,
    )

    # --- Gate 5: Replay consistency ---
    replay_ok = _check_replay(bks_rows) and strict_rows_ok
    detail = "all BKS values match canonical data"
    if not replay_ok:
        mismatches = _find_replay_mismatches(bks_rows)
        detail = f"{len(mismatches)} mismatches: {mismatches[:3]}"
    gate_results[_GATE_REPLAY] = (replay_ok, detail)

    # --- Overall status ---
    all_passed = all(passed for passed, _ in gate_results.values())
    status = READY_FOR_STAGE05_2 if all_passed else NOT_READY

    # --- Write outputs ---
    _write_review_outputs(
        output_dir,
        run_label=run_label,
        status=status,
        gate_results=gate_results,
        findings=findings,
        bks_row_count=len(bks_rows),
        compat_row_count=len(compat_rows),
    )

    return {
        "review_report": output_dir / "review_report.md",
        "review_findings": output_dir / "review_findings.csv",
        "gate_evaluation": output_dir / "gate_evaluation.csv",
        "review_manifest": output_dir / "review_manifest.json",
    }


# ---------------------------------------------------------------------------
# Gate evaluation helpers
# ---------------------------------------------------------------------------


def _check_compatibility(rows: list[dict[str, str]]) -> bool:
    """Check that compatibility assessment matches canonical data."""

    _, canonical = canonical_stage051_rows()
    return rows == canonical


def _check_replay(rows: list[dict[str, str]]) -> bool:
    """Check that CSV BKS values match canonical data."""

    canonical, _ = canonical_stage051_rows()
    return rows == canonical


def validate_stage051_rows(
    bks_rows: list[dict[str, str]],
    compatibility_rows: list[dict[str, str]],
) -> tuple[bool, str]:
    """Strictly compare every published Stage 5.1 field with canonical data."""

    canonical_bks, canonical_compatibility = canonical_stage051_rows()
    if len(bks_rows) != TOTAL_INSTANCES:
        return False, f"expected {TOTAL_INSTANCES} BKS rows, got {len(bks_rows)}"
    if len({row.get("instance", "") for row in bks_rows}) != TOTAL_INSTANCES:
        return False, "BKS instances must be unique"
    if any(tuple(row) != BKS_DATA_FIELDS for row in bks_rows):
        return False, "BKS CSV columns or column order do not match the canonical schema"
    if any(tuple(row) != COMPATIBILITY_FIELDS for row in compatibility_rows):
        return False, "compatibility CSV columns or column order do not match the canonical schema"
    if bks_rows != canonical_bks:
        return False, "one or more BKS fields differ from canonical data"
    if compatibility_rows != canonical_compatibility:
        return False, "compatibility assessment differs from the complete canonical assessment"
    if any("gap" in key.lower() for row in bks_rows for key in row):
        return False, "gap columns are not permitted"
    return True, "all BKS and compatibility fields match canonical data"


def _find_replay_mismatches(rows: list[dict[str, str]]) -> list[str]:
    """Return instance names with mismatched BKS values."""

    canonical_rows, _ = canonical_stage051_rows()
    canonical = {row["instance"]: row for row in canonical_rows}
    mismatches: list[str] = []
    for row in rows:
        inst = row.get("instance", "")
        expected = canonical.get(inst)
        if expected is None:
            mismatches.append(f"{inst}:not_found")
            continue
        differing = [field for field in BKS_DATA_FIELDS if row.get(field) != expected[field]]
        if differing:
            mismatches.append(f"{inst}:{','.join(differing)}")
    return mismatches


def _find_manifest_name(run_dir: Path) -> str:
    """Find the manifest filename from the run label."""

    run_label = run_dir.name
    return f"{run_label}_manifest.json"


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def _write_review_outputs(
    output_dir: Path,
    *,
    run_label: str,
    status: str,
    gate_results: dict[str, tuple[bool, str]],
    findings: list[dict[str, str]],
    bks_row_count: int,
    compat_row_count: int,
) -> None:
    """Write review report, findings, gate evaluation, and manifest."""

    # Review report
    report_lines: list[str] = [
        f"# Stage 5.1 Review Report — {run_label}",
        "",
        f"**Status: {status}**",
        "",
        "## Gate Evaluation",
        "",
        "| Gate | Passed | Detail |",
        "|------|--------|--------|",
    ]
    for gate_name in _GATE_NAMES:
        passed, detail = gate_results.get(gate_name, (False, "not evaluated"))
        report_lines.append(
            f"| {gate_name} | {'yes' if passed else 'no'} | {detail} |"
        )
    report_lines.append("")
    report_lines.append("## Summary")
    report_lines.append("")
    report_lines.append(f"- BKS data rows: {bks_row_count}")
    report_lines.append(f"- Compatibility dimensions: {compat_row_count}")
    report_lines.append(f"- Total canonical instances: {TOTAL_INSTANCES}")
    report_lines.append("")
    (output_dir / "review_report.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )

    # Findings CSV
    with (output_dir / "review_findings.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=("gate", "status", "detail"))
        writer.writeheader()
        for gate_name in _GATE_NAMES:
            passed, detail = gate_results.get(gate_name, (False, "not evaluated"))
            findings.append({
                "gate": gate_name,
                "status": "PASS" if passed else "FAIL",
                "detail": detail,
            })
        writer.writerows(findings)

    # Gate evaluation CSV
    with (output_dir / "gate_evaluation.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=("gate", "passed", "detail"))
        writer.writeheader()
        for gate_name in _GATE_NAMES:
            passed, detail = gate_results.get(gate_name, (False, "not evaluated"))
            writer.writerow({
                "gate": gate_name,
                "passed": passed,
                "detail": detail,
            })

    # Review manifest
    manifest = {
        "run_label": run_label,
        "stage_id": "stage05.1",
        "component": "best_known",
        "status": status,
        "schema_version": "stage05.1-review-v2",
        "gates": {
            name: {"passed": passed, "detail": detail}
            for name, (passed, detail) in gate_results.items()
        },
        "total_instances": TOTAL_INSTANCES,
        "bks_row_count": bks_row_count,
        "compat_row_count": compat_row_count,
    }
    (output_dir / "review_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def _write_failure(output_dir: Path, reason: str) -> dict[str, Path]:
    """Write a minimal NOT_READY report."""

    output_dir.mkdir(parents=True, exist_ok=True)
    report = f"# Stage 5.1 Review Report\n\n**Status: {NOT_READY}**\n\nReason: {reason}\n"
    (output_dir / "review_report.md").write_text(report, encoding="utf-8")
    with (output_dir / "review_findings.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=("gate", "status", "detail"))
        writer.writeheader()
        writer.writerow({"gate": "manifest", "status": "FAIL", "detail": reason})
    with (output_dir / "gate_evaluation.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=("gate", "passed", "detail"))
        writer.writeheader()
        for gate in _GATE_NAMES:
            writer.writerow({"gate": gate, "passed": False, "detail": reason})
    manifest = {
        "status": NOT_READY,
        "reason": reason,
        "stage_id": "stage05.1",
    }
    (output_dir / "review_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "review_report": output_dir / "review_report.md",
        "review_findings": output_dir / "review_findings.csv",
        "gate_evaluation": output_dir / "gate_evaluation.csv",
        "review_manifest": output_dir / "review_manifest.json",
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Review Stage 5.1 best-known values evidence"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    outputs = review_stage051(
        run_dir=arguments.run_dir.resolve(),
        output_dir=arguments.output_dir.resolve(),
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
