from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from evrptw.repository import repository_root


def test_repository_root_uses_active_checkout_when_package_is_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    (checkout / "pyproject.toml").write_text("[project]\nname='evrptw-reproduction'\n")
    configs = checkout / "configs"
    configs.mkdir()
    (configs / "stage052_performance.toml").write_text("schema_version='test'\n")
    nested = checkout / "nested" / "working-directory"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)

    assert repository_root() == checkout.resolve()


def test_repository_root_honours_explicit_checkout_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    (checkout / "pyproject.toml").write_text("[project]\nname='evrptw-reproduction'\n")
    configs = checkout / "configs"
    configs.mkdir()
    (configs / "stage052_performance.toml").write_text("schema_version='test'\n")
    monkeypatch.setenv("EVRPTW_REPOSITORY_ROOT", str(checkout))
    monkeypatch.chdir(tmp_path)

    assert repository_root() == checkout.resolve()
