from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load_tool() -> ModuleType:
    tools = Path(__file__).resolve().parents[1] / "tools"
    sys.path.insert(0, str(tools))
    spec = importlib.util.spec_from_file_location(
        "delete_stage052_verified_sources",
        tools / "delete_stage052_verified_sources.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_signed_candidate_manifest_is_verified_before_exact_deletion(
    tmp_path: Path,
) -> None:
    tool = _load_tool()
    d_root = tmp_path / "d"
    wsl_root = tmp_path / "wsl"
    d_source = d_root / "history" / "attempt01"
    wsl_source = wsl_root / "attempt02"
    d_source.mkdir(parents=True)
    wsl_source.mkdir(parents=True)
    (d_source / "manifest.json").write_text("d", encoding="utf-8")
    (wsl_source / "raw.bin").write_bytes(b"wsl")
    verifier = sys.modules["hash_retention_tree_windows"]
    d_identity = verifier.verify_mappings(
        d_root,
        workers=2,
        mappings=(("attempt01:d", tool.PurePosixPath("history/attempt01")),),
    )["mappings"]["attempt01:d"]
    wsl_identity = verifier.verify_mappings(
        wsl_root,
        workers=2,
        mappings=(("attempt02:wsl", tool.PurePosixPath("attempt02")),),
    )["mappings"]["attempt02:wsl"]
    manifest = tmp_path / "source-deletion-candidates.json"
    payload = {
        "schema_version": "stage052-source-deletion-candidates-v1",
        "migration_id": "test",
        "source_deletion_authorized": False,
        "requires_literal_confirmation": "确认",
        "rollback_after_source_deletion": "none_single_media_archive",
        "candidate_count": 2,
        "candidate_bytes": (
            d_identity["byte_count"] + wsl_identity["byte_count"]
        ),
        "candidates": [
            {
                "logical_id": "attempt01:d",
                "source_root_alias": "d_archive",
                "source_path": str(d_source),
                "file_count": d_identity["file_count"],
                "byte_count": d_identity["byte_count"],
                "tree_sha256": d_identity["tree_sha256"],
                "deletion_authorized": False,
            },
            {
                "logical_id": "attempt02:wsl",
                "source_root_alias": "wsl_staging",
                "source_path": str(wsl_source),
                "file_count": wsl_identity["file_count"],
                "byte_count": wsl_identity["byte_count"],
                "tree_sha256": wsl_identity["tree_sha256"],
                "deletion_authorized": False,
            },
        ],
    }
    digest = tool._write_signed_json(manifest, payload)

    _loaded, candidates = tool.load_candidates(
        manifest,
        expected_sha256=digest,
        roots={"d_archive": d_root, "wsl_staging": wsl_root},
    )
    observed = tool._posix_identities(
        root=d_root,
        candidates=(candidates[0],),
        workers=2,
    )
    assert observed["attempt01:d"] == (
        d_identity["file_count"],
        d_identity["byte_count"],
        d_identity["tree_sha256"],
    )
    deleted, absent = tool.delete_candidates(candidates)

    assert set(deleted) == {str(d_source), str(wsl_source)}
    assert absent == ()
    assert not d_source.exists()
    assert not wsl_source.exists()


def test_tampered_deletion_manifest_fails_closed(tmp_path: Path) -> None:
    tool = _load_tool()
    manifest = tmp_path / "source-deletion-candidates.json"
    digest = tool._write_signed_json(manifest, {"schema_version": "wrong"})
    manifest.write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError, match="SHA-256"):
        tool.load_candidates(
            manifest,
            expected_sha256=digest,
            roots={
                "d_archive": tmp_path / "d",
                "wsl_staging": tmp_path / "wsl",
            },
        )
