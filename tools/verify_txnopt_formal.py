"""Re-run the bounded TxnOpt TLA+/PlusCal verification receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

_STATE_COUNTS = re.compile(
    r"(?P<generated>\d+) states generated, "
    r"(?P<distinct>\d+) distinct states found"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: list[str], *, cwd: Path) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    output = completed.stdout + completed.stderr
    if completed.returncode != 0:
        raise RuntimeError(
            f"formal command failed with exit code {completed.returncode}:\n{output}"
        )
    return output


def _counts(output: str) -> dict[str, int]:
    match = _STATE_COUNTS.search(output)
    if match is None or "No error has been found" not in output:
        raise RuntimeError("TLC output did not contain a successful state count")
    return {
        "generated_states": int(match.group("generated")),
        "distinct_states": int(match.group("distinct")),
    }


def _resolve_java(argument: str | None) -> Path:
    candidate = argument or os.environ.get("TXNOPT_JAVA") or shutil.which("java")
    if not candidate:
        raise RuntimeError("Java was not found; pass --java or set TXNOPT_JAVA")
    path = Path(candidate).resolve()
    if not path.is_file():
        raise RuntimeError(f"Java executable does not exist: {path}")
    return path


def _resolve_jar(argument: str | None) -> Path:
    candidate = argument or os.environ.get("TLA2TOOLS_JAR")
    if not candidate:
        raise RuntimeError("pass --tla2tools or set TLA2TOOLS_JAR")
    path = Path(candidate).resolve()
    if not path.is_file():
        raise RuntimeError(f"tla2tools.jar does not exist: {path}")
    return path


def verify(
    root: Path,
    *,
    java: Path,
    tla2tools: Path,
) -> dict[str, Any]:
    formal = root / "formal"
    receipt_path = formal / "model-check-receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected_hashes: dict[str, str] = receipt["input_sha256"]
    for relative_path, expected in expected_hashes.items():
        actual = _sha256(root / relative_path)
        if actual != expected:
            raise RuntimeError(
                f"formal input hash mismatch for {relative_path}: {actual} != {expected}"
            )
    expected_jar = str(receipt["toolchain"]["tla2tools_jar_sha256"])
    actual_jar = _sha256(tla2tools)
    if actual_jar != expected_jar:
        raise RuntimeError(f"tla2tools.jar hash mismatch: {actual_jar} != {expected_jar}")

    with tempfile.TemporaryDirectory(prefix="txnopt-formal-") as raw_temp:
        temp = Path(raw_temp)
        for name in ("TxnOptOrderedPlusCal.tla", "TxnOptOrderedPlusCal.cfg"):
            shutil.copy2(formal / name, temp / name)
        translation = _run(
            [str(java), "-cp", str(tla2tools), "pcal.trans", "TxnOptOrderedPlusCal.tla"],
            cwd=temp,
        )
        if "Translation completed" not in translation:
            raise RuntimeError("PlusCal translation did not report completion")
        if (
            _sha256(temp / "TxnOptOrderedPlusCal.tla")
            != expected_hashes["formal/TxnOptOrderedPlusCal.tla"]
        ):
            raise RuntimeError("checked-in PlusCal translation is not reproducible")

        base = [str(java), "-XX:+UseParallelGC", "-jar", str(tla2tools), "-workers", "4"]
        primary_output = _run(
            base
            + [
                "-metadir",
                str(temp / "primary-states"),
                "-config",
                "TxnOpt.cfg",
                "TxnOpt.tla",
            ],
            cwd=formal,
        )
        pluscal_output = _run(
            base
            + [
                "-metadir",
                str(temp / "pluscal-states"),
                "-config",
                "TxnOptOrderedPlusCal.cfg",
                "TxnOptOrderedPlusCal.tla",
            ],
            cwd=temp,
        )
        t3_output = _run(
            base
            + [
                "-metadir",
                str(temp / "t3-states"),
                "-config",
                "TxnOptT3.cfg",
                "TxnOptT3.tla",
            ],
            cwd=formal,
        )

    observed = {
        "primary": _counts(primary_output),
        "pluscal": _counts(pluscal_output),
        "t3_witness": _counts(t3_output),
    }
    if observed != receipt["model_check"]:
        raise RuntimeError(
            f"model-check state counts changed: {observed} != {receipt['model_check']}"
        )
    return {
        "schema_version": "txnopt-formal-verification-v1",
        "status": "PASS",
        "receipt": str(receipt_path),
        "model_check": observed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--java")
    parser.add_argument("--tla2tools")
    args = parser.parse_args()
    result = verify(
        args.root.resolve(),
        java=_resolve_java(args.java),
        tla2tools=_resolve_jar(args.tla2tools),
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
