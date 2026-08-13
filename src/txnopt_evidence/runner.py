"""Raw-only Level 1 producer for TxnOpt case configurations."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from txnopt import _native
from txnopt_evidence.case_codec import execute_case, parse_run_config
from txnopt_evidence.codec import (
    canonical_json_bytes,
    read_signed_json,
    sha256_bytes,
    sha256_file,
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
    producer_identity = _producer_identity(payload.get("build_manifest"))

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
        "producer_identity": producer_identity,
        "artifacts": artifacts,
        "runner_decision": None,
        "fallback_count": 0,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_digest = write_signed_json(manifest_path, manifest)
    return RawArtifactRef(run_label, manifest_path, manifest_digest)


def _producer_identity(raw_binding: object) -> dict[str, object]:
    attestation = dict(_native.BUILD_ATTESTATION)
    native_path = Path(_native.__file__).resolve(strict=True)
    native_sha256 = sha256_file(native_path)
    if raw_binding is None:
        return {
            "binding_status": "UNBOUND_TEST_ONLY",
            "native_build_attestation": attestation,
            "installed_native_sha256": native_sha256,
        }
    if not isinstance(raw_binding, dict):
        raise ValueError("build_manifest binding must be an object")
    path = raw_binding.get("path")
    expected_sha256 = raw_binding.get("sha256")
    if not isinstance(path, str) or not isinstance(expected_sha256, str):
        raise ValueError("build_manifest binding path and SHA-256 are required")
    manifest_path = Path(path).resolve(strict=True)
    manifest = read_signed_json(manifest_path)
    actual_sha256 = sha256_file(manifest_path)
    if actual_sha256 != expected_sha256:
        raise ValueError("build_manifest binding SHA-256 differs")
    producer = manifest.get("producer")
    artifacts = manifest.get("artifacts")
    if not isinstance(producer, dict) or not isinstance(artifacts, dict):
        raise ValueError("build manifest producer or artifacts are missing")
    native = artifacts.get("native_extension")
    wheel = artifacts.get("wheel")
    if not isinstance(native, dict) or not isinstance(wheel, dict):
        raise ValueError("build manifest native or wheel artifact is missing")
    for key in (
        "revision",
        "git_tree",
        "source_manifest_sha256",
        "tracked_file_count",
        "source_dirty",
        "development_override",
    ):
        if producer.get(key) != attestation.get(
            {"revision": "source_revision", "git_tree": "source_tree"}.get(key, key)
        ):
            raise ValueError(f"installed native build differs from build manifest: {key}")
    if (
        producer.get("source_dirty") is not False
        or producer.get("development_override") is not False
    ):
        raise ValueError("formal run requires a clean non-development native build")
    if (
        native.get("protocol") != "txnopt-native-round-v1"
        or native.get("sha256") != native_sha256
    ):
        raise ValueError("installed native extension differs from build manifest")
    if not isinstance(wheel.get("sha256"), str):
        raise ValueError("build manifest wheel SHA-256 is missing")
    return {
        "binding_status": "BOUND_CLEAN_BUILD",
        "build_manifest_path": str(manifest_path),
        "build_manifest_sha256": actual_sha256,
        "wheel_sha256": wheel["sha256"],
        "installed_native_sha256": native_sha256,
        "native_build_attestation": attestation,
    }


__all__ = ["run_config", "run_config_file"]
