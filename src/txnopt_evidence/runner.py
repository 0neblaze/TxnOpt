"""Raw-only Level 1 producer for TxnOpt case configurations."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from txnopt_evidence.case_codec import execute_case, parse_run_config
from txnopt_evidence.codec import (
    canonical_json_bytes,
    sha256_bytes,
    write_event_stream,
    write_exclusive,
    write_sidecar,
    write_signed_json,
)
from txnopt_evidence.contracts import RawArtifactRef

_RUN_LABEL = re.compile(r"[a-z0-9][a-z0-9._-]{2,127}")


def run_config_file(config_path: Path) -> RawArtifactRef:
    input_bytes = config_path.resolve(strict=True).read_bytes()
    payload: object = json.loads(input_bytes)
    if not isinstance(payload, dict):
        raise ValueError("TxnOpt run config must contain an object")
    return run_config(payload, input_sha256=sha256_bytes(input_bytes))


def run_config(
    payload: Mapping[str, Any],
    *,
    input_sha256: str,
) -> RawArtifactRef:
    if payload.get("schema_version") != "txnopt-run-config-v1":
        raise ValueError("unsupported TxnOpt run config schema")
    run_label = payload.get("run_label")
    output_root = payload.get("output_root")
    run_config_payload = payload.get("run_config")
    case_payload = payload.get("case")
    if not isinstance(run_label, str) or _RUN_LABEL.fullmatch(run_label) is None:
        raise ValueError("run_label is not canonical")
    if not isinstance(output_root, str) or not output_root:
        raise ValueError("output_root must be a non-empty path string")
    if not isinstance(run_config_payload, dict) or not isinstance(case_payload, dict):
        raise ValueError("run_config and case must be objects")

    output_dir = Path(output_root).expanduser().resolve() / run_label
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"raw run directory already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    captured: list[tuple[Mapping[str, object], ...]] = []
    physical: list[tuple[Mapping[str, object], ...]] = []
    config = parse_run_config(run_config_payload)
    result = execute_case(
        case_payload,
        config=config,
        semantic_sink=captured.append,
        physical_sink=physical.append,
    )
    if len(captured) != 1:
        raise RuntimeError("runtime did not emit exactly one semantic stream")

    normalized_config = canonical_json_bytes(dict(payload), pretty=True)
    config_path = output_dir / "config.json"
    config_digest = write_exclusive(config_path, normalized_config)
    write_sidecar(config_path, config_digest)
    events_path = output_dir / "events.jsonl"
    events_digest, terminal_event_digest = write_event_stream(events_path, captured[0])
    result_path = output_dir / "result.json"
    result_digest = write_signed_json(result_path, result)
    artifacts: list[dict[str, object]] = [
        {
            "path": "config.json",
            "sha256": config_digest,
            "bytes": config_path.stat().st_size,
        },
        {
            "path": "events.jsonl",
            "sha256": events_digest,
            "bytes": events_path.stat().st_size,
            "event_count": len(captured[0]),
            "terminal_event_sha256": terminal_event_digest,
        },
        {
            "path": "result.json",
            "sha256": result_digest,
            "bytes": result_path.stat().st_size,
        },
    ]
    if physical:
        if len(physical) != 1:
            raise RuntimeError("runtime emitted more than one physical stream")
        physical_path = output_dir / "physical.jsonl"
        physical_digest, terminal_physical_digest = write_event_stream(
            physical_path,
            physical[0],
        )
        artifacts.append(
            {
                "path": "physical.jsonl",
                "sha256": physical_digest,
                "bytes": physical_path.stat().st_size,
                "event_count": len(physical[0]),
                "terminal_event_sha256": terminal_physical_digest,
            }
        )
    manifest = {
        "schema_version": "txnopt-raw-artifact-v1",
        "run_label": run_label,
        "input_config_sha256": input_sha256,
        "contract": "txnopt-contract-v1",
        "semantic_trace": "txnopt-semantic-trace-v1",
        "physical_trace": "txnopt-physical-trace-v1",
        "domain": case_payload.get("domain"),
        "artifacts": artifacts,
        "runner_decision": None,
        "fallback_count": 0,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_digest = write_signed_json(manifest_path, manifest)
    return RawArtifactRef(run_label, manifest_path, manifest_digest)


__all__ = ["run_config", "run_config_file"]
