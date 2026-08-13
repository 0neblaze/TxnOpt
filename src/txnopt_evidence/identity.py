"""Pre-run evidence identity used as the trust root for independent replay."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Self

from txnopt_evidence.codec import (
    canonical_json_bytes,
    read_signed_json,
    sha256_bytes,
    sha256_file,
)


def _frozen_mapping(payload: Mapping[str, object]) -> Mapping[str, object]:
    copied: object = json.loads(canonical_json_bytes(dict(payload)))
    if not isinstance(copied, dict):  # pragma: no cover - canonical object invariant
        raise TypeError("identity payload must be an object")
    return MappingProxyType(copied)


def _digest(value: object, label: str, *, length: int = 64) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase hexadecimal digest")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


@dataclass(frozen=True, slots=True)
class ExpectedEvidenceIdentity:
    """Immutable identity fixed before a v3 producer run starts.

    The raw bundle is observed evidence.  This object is the separate trust root
    supplied by a plan or a test harness; it must never be reconstructed from a
    bundle that is already under review.
    """

    SCHEMA_VERSION: ClassVar[str] = "txnopt-expected-evidence-identity-v1"
    ARTIFACT_SCHEMA: ClassVar[str] = "txnopt-raw-artifact-v3"

    run_label: str
    input_config_sha256: str
    config_artifact_sha256: str
    domain: str
    execution_mode: str
    expected_oracle: str
    producer_identity: Mapping[str, object]

    def __post_init__(self) -> None:
        _string(self.run_label, "run_label")
        _digest(self.input_config_sha256, "input_config_sha256")
        _digest(self.config_artifact_sha256, "config_artifact_sha256")
        if self.config_artifact_sha256 != self.input_config_sha256:
            raise ValueError("retained config digest must equal the input config digest")
        if self.domain not in {"evrptw", "rcpsp"}:
            raise ValueError("domain must be evrptw or rcpsp")
        _string(self.execution_mode, "execution_mode")
        _string(self.expected_oracle, "expected_oracle")
        object.__setattr__(self, "producer_identity", _frozen_mapping(self.producer_identity))
        _validate_producer_payload(self.producer_identity)

    @classmethod
    def from_plan_inputs(
        cls,
        config_path: Path,
        *,
        build_manifest_path: Path,
    ) -> Self:
        config_bytes, config = _config(config_path)
        build_path = build_manifest_path.resolve(strict=True)
        build = read_signed_json(build_path)
        build_digest = sha256_file(build_path)
        binding = config.get("build_manifest")
        if not isinstance(binding, dict):
            raise ValueError("formal config lacks its build manifest binding")
        if (
            binding.get("sha256") != build_digest
            or Path(str(binding.get("path"))).resolve(strict=True) != build_path
        ):
            raise ValueError("formal config build manifest binding differs")
        producer = _bound_producer_identity(build, build_digest)
        return cls._from_config(config_bytes, config, producer)

    @classmethod
    def for_unbound_test(
        cls,
        config_path: Path,
        *,
        producer_identity: object,
    ) -> Self:
        """Create an explicit local-test anchor; formal tools never call this."""

        if not isinstance(producer_identity, dict):
            raise ValueError("unbound test producer identity must be an object")
        if producer_identity.get("binding_status") != "UNBOUND_TEST_ONLY":
            raise ValueError("unbound test identity requires UNBOUND_TEST_ONLY")
        config_bytes, config = _config(config_path)
        return cls._from_config(config_bytes, config, producer_identity)

    @classmethod
    def _from_config(
        cls,
        config_bytes: bytes,
        config: dict[str, Any],
        producer_identity: Mapping[str, object],
    ) -> Self:
        run_config = config.get("run_config")
        case = config.get("case")
        if not isinstance(run_config, dict) or not isinstance(case, dict):
            raise ValueError("run config and case must be objects")
        domain = _string(case.get("domain"), "case domain")
        execution_mode = _string(run_config.get("execution_mode"), "execution mode")
        return cls(
            run_label=_string(config.get("run_label"), "run_label"),
            input_config_sha256=sha256_bytes(config_bytes),
            config_artifact_sha256=sha256_bytes(config_bytes),
            domain=domain,
            execution_mode=execution_mode,
            expected_oracle=_expected_oracle(config),
            producer_identity=producer_identity,
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "artifact_schema": self.ARTIFACT_SCHEMA,
            "run_label": self.run_label,
            "input_config_sha256": self.input_config_sha256,
            "config_artifact_sha256": self.config_artifact_sha256,
            "domain": self.domain,
            "execution_mode": self.execution_mode,
            "expected_oracle": self.expected_oracle,
            "producer_identity": dict(self.producer_identity),
        }

    @classmethod
    def from_payload(cls, payload: object) -> Self:
        expected = {
            "schema_version",
            "artifact_schema",
            "run_label",
            "input_config_sha256",
            "config_artifact_sha256",
            "domain",
            "execution_mode",
            "expected_oracle",
            "producer_identity",
        }
        if not isinstance(payload, dict) or set(payload) != expected:
            raise ValueError("expected evidence identity field set differs")
        if (
            payload.get("schema_version") != cls.SCHEMA_VERSION
            or payload.get("artifact_schema") != cls.ARTIFACT_SCHEMA
        ):
            raise ValueError("expected evidence identity schema differs")
        producer = payload.get("producer_identity")
        if not isinstance(producer, dict):
            raise ValueError("expected producer identity must be an object")
        return cls(
            run_label=_string(payload.get("run_label"), "run_label"),
            input_config_sha256=_digest(
                payload.get("input_config_sha256"), "input_config_sha256"
            ),
            config_artifact_sha256=_digest(
                payload.get("config_artifact_sha256"), "config_artifact_sha256"
            ),
            domain=_string(payload.get("domain"), "domain"),
            execution_mode=_string(payload.get("execution_mode"), "execution_mode"),
            expected_oracle=_string(payload.get("expected_oracle"), "expected_oracle"),
            producer_identity=producer,
        )


def _config(path: Path) -> tuple[bytes, dict[str, Any]]:
    candidate = path.expanduser().absolute()
    if candidate.is_symlink():
        raise ValueError("expected config cannot be a symlink")
    config_bytes = candidate.resolve(strict=True).read_bytes()
    raw: object = json.loads(config_bytes)
    if not isinstance(raw, dict) or raw.get("schema_version") != "txnopt-run-config-v1":
        raise ValueError("expected config schema differs")
    return config_bytes, raw


def _bound_producer_identity(
    build: Mapping[str, Any],
    build_manifest_sha256: str,
) -> dict[str, object]:
    if build.get("schema_version") != "txnopt-level1-build-manifest-v1":
        raise ValueError("build manifest schema differs")
    producer = build.get("producer")
    artifacts = build.get("artifacts")
    if not isinstance(producer, dict) or not isinstance(artifacts, dict):
        raise ValueError("build manifest producer or artifacts are missing")
    wheel = artifacts.get("wheel")
    native = artifacts.get("native_extension")
    if not isinstance(wheel, dict) or not isinstance(native, dict):
        raise ValueError("build manifest wheel or native artifact is missing")
    if (
        producer.get("source_dirty") is not False
        or producer.get("development_override") is not False
    ):
        raise ValueError("expected producer must be a clean non-development build")
    return {
        "binding_status": "BOUND_CLEAN_BUILD",
        "build_manifest_sha256": _digest(
            build_manifest_sha256, "build_manifest_sha256"
        ),
        "source_revision": _digest(producer.get("revision"), "source_revision", length=40),
        "source_tree": _digest(producer.get("git_tree"), "source_tree", length=40),
        "source_manifest_sha256": _digest(
            producer.get("source_manifest_sha256"), "source_manifest_sha256"
        ),
        "tracked_file_count": _positive_integer(
            producer.get("tracked_file_count"), "tracked_file_count"
        ),
        "wheel_sha256": _digest(wheel.get("sha256"), "wheel_sha256"),
        "installed_native_sha256": _digest(native.get("sha256"), "native_sha256"),
        "native_protocol": _string(native.get("protocol"), "native_protocol"),
    }


def _positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _validate_producer_payload(payload: Mapping[str, object]) -> None:
    status = payload.get("binding_status")
    if status == "UNBOUND_TEST_ONLY":
        if not {"binding_status", "installed_native_sha256"}.issubset(payload):
            raise ValueError("unbound producer identity is incomplete")
        _digest(payload.get("installed_native_sha256"), "installed_native_sha256")
        return
    expected = {
        "binding_status",
        "build_manifest_sha256",
        "source_revision",
        "source_tree",
        "source_manifest_sha256",
        "tracked_file_count",
        "wheel_sha256",
        "installed_native_sha256",
        "native_protocol",
    }
    if status != "BOUND_CLEAN_BUILD" or set(payload) != expected:
        raise ValueError("bound producer identity field set differs")
    _digest(payload.get("build_manifest_sha256"), "build_manifest_sha256")
    _digest(payload.get("source_revision"), "source_revision", length=40)
    _digest(payload.get("source_tree"), "source_tree", length=40)
    _digest(payload.get("source_manifest_sha256"), "source_manifest_sha256")
    _positive_integer(payload.get("tracked_file_count"), "tracked_file_count")
    _digest(payload.get("wheel_sha256"), "wheel_sha256")
    _digest(payload.get("installed_native_sha256"), "installed_native_sha256")
    if payload.get("native_protocol") != "txnopt-native-round-v1":
        raise ValueError("native protocol differs")


def _expected_oracle(config: Mapping[str, Any]) -> str:
    case = config.get("case")
    if not isinstance(case, dict):
        raise ValueError("case must be an object")
    domain = case.get("domain")
    if domain == "rcpsp":
        return "txnopt_cases.rcpsp.oracle.RCPSPOracle"
    if domain == "evrptw" and case.get("backend", "python") == "python":
        return "txnopt_cases.evrptw.oracle.EVRPTWOracle"
    if domain == "evrptw" and case.get("backend") == "native":
        return "txnopt_cases.evrptw.native_oracle.NativeEVRPTWOracle"
    raise ValueError("case domain or backend differs")


__all__ = ["ExpectedEvidenceIdentity"]
