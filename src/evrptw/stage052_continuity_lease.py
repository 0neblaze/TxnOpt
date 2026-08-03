"""Atomic single-writer lease for long Stage 5.2 Codex work."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import secrets
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "stage05.2-continuity-lease-v1"
DEFAULT_TTL_SECONDS = 45 * 60


def _paths(root: Path) -> tuple[Path, Path, Path]:
    directory = root / "results" / ".stage052-continuity"
    return directory / "writer.lock", directory / "lease.json", directory / "renew.json"


def _boot_id() -> str:
    return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()


def _process_start_time(pid: int) -> str:
    raw = (Path("/proc") / str(pid) / "stat").read_text(encoding="ascii")
    fields = raw[raw.rfind(")") + 2 :].split()
    if len(fields) <= 19:
        raise RuntimeError("process identity record is truncated")
    return fields[19]


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def _lock_is_held(path: Path) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        return False
    finally:
        os.close(descriptor)


def inspect(root: Path) -> dict[str, Any] | None:
    lock_path, lease_path, _ = _paths(root)
    record = _read_json(lease_path)
    if record is None or record.get("schema_version") != SCHEMA_VERSION:
        return None
    try:
        pid = int(record["pid"])
        active = (
            record.get("boot_id") == _boot_id()
            and record.get("process_start_time") == _process_start_time(pid)
            and float(record["expires_at_epoch"]) > time.time()
            and _lock_is_held(lock_path)
        )
    except (KeyError, TypeError, ValueError, OSError, RuntimeError):
        return None
    return record if active else None


def _holder(
    root: Path,
    *,
    token: str,
    owner: str,
    phase: str,
    label: str | None,
    revision: str,
    ttl_seconds: int,
) -> int:
    lock_path, lease_path, renew_path = _paths(root)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 3
        stopping = False

        def stop(_signum: int, _frame: object) -> None:
            nonlocal stopping
            stopping = True

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        now = time.time()
        record: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "owner": owner,
            "token": token,
            "pid": os.getpid(),
            "process_start_time": _process_start_time(os.getpid()),
            "boot_id": _boot_id(),
            "phase": phase,
            "label": label,
            "repository_revision": revision,
            "acquired_at_epoch": now,
            "renewed_at_epoch": now,
            "expires_at_epoch": now + ttl_seconds,
        }
        _atomic_json(lease_path, record)
        last_renewed = now
        while not stopping and time.time() < float(record["expires_at_epoch"]):
            renewal = _read_json(renew_path)
            if (
                renewal is not None
                and renewal.get("token") == token
                and isinstance(renewal.get("renewed_at_epoch"), (int, float))
                and float(renewal["renewed_at_epoch"]) > last_renewed
            ):
                last_renewed = float(renewal["renewed_at_epoch"])
                record["renewed_at_epoch"] = last_renewed
                record["expires_at_epoch"] = last_renewed + ttl_seconds
                record["phase"] = str(renewal.get("phase", record["phase"]))
                record["label"] = renewal.get("label", record["label"])
                _atomic_json(lease_path, record)
            time.sleep(1.0)
        current = _read_json(lease_path)
        if current is not None and current.get("token") == token:
            lease_path.unlink(missing_ok=True)
            renew_path.unlink(missing_ok=True)
        return 0
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def acquire(
    root: Path,
    *,
    owner: str,
    phase: str,
    label: str | None,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
) -> dict[str, Any]:
    active = inspect(root)
    if active is not None:
        raise RuntimeError(f"continuity lease is already held by {active['owner']}")
    token = secrets.token_hex(16)
    revision = subprocess.run(
        ("git", "-C", str(root), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    command = (
        sys.executable,
        "-m",
        "evrptw.stage052_continuity_lease",
        "_hold",
        "--root",
        str(root),
        "--token",
        token,
        "--owner",
        owner,
        "--phase",
        phase,
        "--label",
        label or "",
        "--revision",
        revision,
        "--ttl-seconds",
        str(ttl_seconds),
    )
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        record = inspect(root)
        if record is not None and record.get("token") == token:
            return record
        if process.poll() is not None:
            break
        time.sleep(0.05)
    raise RuntimeError("continuity lease could not be acquired atomically")


def renew(root: Path, *, token: str, phase: str, label: str | None) -> dict[str, Any]:
    active = inspect(root)
    if active is None or active.get("token") != token:
        raise RuntimeError("continuity lease identity is not active")
    _, _, renew_path = _paths(root)
    timestamp = time.time()
    _atomic_json(
        renew_path,
        {"token": token, "renewed_at_epoch": timestamp, "phase": phase, "label": label},
    )
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        updated = inspect(root)
        if updated is not None and float(updated["renewed_at_epoch"]) >= timestamp:
            return updated
        time.sleep(0.05)
    raise RuntimeError("continuity lease renewal was not acknowledged")


def release(root: Path, *, token: str) -> None:
    active = inspect(root)
    if active is None or active.get("token") != token:
        raise RuntimeError("continuity lease identity is not active")
    os.kill(int(active["pid"]), signal.SIGTERM)
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if inspect(root) is None:
            return
        time.sleep(0.05)
    raise RuntimeError("continuity lease holder did not release")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("acquire", "renew", "release", "inspect", "_hold"):
        item = subparsers.add_parser(name)
        item.add_argument("--root", type=Path, required=True)
        item.add_argument("--token", default="")
        item.add_argument("--owner", default="")
        item.add_argument("--phase", default="")
        item.add_argument("--label", default="")
        item.add_argument("--revision", default="")
        item.add_argument("--ttl-seconds", type=int, default=DEFAULT_TTL_SECONDS)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    root = arguments.root.resolve()
    if arguments.command == "_hold":
        return _holder(
            root,
            token=arguments.token,
            owner=arguments.owner,
            phase=arguments.phase,
            label=arguments.label or None,
            revision=arguments.revision,
            ttl_seconds=arguments.ttl_seconds,
        )
    if arguments.command == "acquire":
        payload = acquire(
            root,
            owner=arguments.owner,
            phase=arguments.phase,
            label=arguments.label or None,
            ttl_seconds=arguments.ttl_seconds,
        )
    elif arguments.command == "renew":
        payload = renew(
            root,
            token=arguments.token,
            phase=arguments.phase,
            label=arguments.label or None,
        )
    elif arguments.command == "release":
        release(root, token=arguments.token)
        payload = {"released": True}
    else:
        payload = inspect(root) or {"active": False}
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
