"""Tests for Stage 5.1 best-known values and model compatibility."""

from __future__ import annotations

import pytest

from evrptw.best_known import (
    BEST_KNOWN_VALUES,
    SOURCE_REFERENCES,
    TOTAL_INSTANCES,
    assess_compatibility,
    get_all_bks,
    get_bks,
)


def test_stage051_replay_rejects_tampered_compatibility_text() -> None:
    from evrptw.experiments.stage051_best_known import canonical_stage051_rows
    from evrptw.experiments.stage051_best_known_review import validate_stage051_rows

    bks_rows, compatibility_rows = canonical_stage051_rows()
    tampered = [dict(row) for row in compatibility_rows]
    tampered[0]["published_model"] = "tampered"
    passed, _ = validate_stage051_rows(bks_rows, tampered)
    assert not passed


@pytest.mark.parametrize(
    ("field", "value"),
    [("source_doi", "wrong"), ("bks_charging_time", "0"), ("model_compatible", "True")],
)
def test_stage051_replay_rejects_tampered_bks_fields(field: str, value: str) -> None:
    from evrptw.experiments.stage051_best_known import canonical_stage051_rows
    from evrptw.experiments.stage051_best_known_review import validate_stage051_rows

    bks_rows, compatibility_rows = canonical_stage051_rows()
    tampered = [dict(row) for row in bks_rows]
    tampered[0][field] = value
    passed, _ = validate_stage051_rows(tampered, compatibility_rows)
    assert not passed


def test_stage051_output_layout_rejects_double_run_label(tmp_path: object) -> None:
    from pathlib import Path

    from evrptw.experiments.stage051_best_known import validate_stage051_output_dir

    root = Path(str(tmp_path))
    label = "stage05.1_best_known_attempt02"
    with pytest.raises(ValueError, match="results/<canonical-run-label>"):
        validate_stage051_output_dir(root, root / "results" / label / label, label)


def test_stage051_prerequisite_rejects_missing_publication(tmp_path: object) -> None:
    from pathlib import Path

    from evrptw.experiments.stage051_best_known import verify_stage04_prerequisite

    with pytest.raises(FileNotFoundError):
        verify_stage04_prerequisite(Path(str(tmp_path)), Path("missing.json"))


class TestBKSDataCompleteness:
    """Verify that all 92 Schneider benchmark instances have BKS records."""

    def test_total_count(self) -> None:
        assert TOTAL_INSTANCES == 92
        assert len(BEST_KNOWN_VALUES) == 92

    def test_small_instance_count(self) -> None:
        small = [r for r in BEST_KNOWN_VALUES if r.customer_count < 100]
        assert len(small) == 36

    def test_large_instance_count(self) -> None:
        large = [r for r in BEST_KNOWN_VALUES if r.customer_count == 100]
        assert len(large) == 56

    @pytest.mark.parametrize("count", [5, 10, 15])
    def test_small_instance_groups(self, count: int) -> None:
        instances = [r for r in BEST_KNOWN_VALUES if r.customer_count == count]
        assert len(instances) == 12

    def test_class_counts(self) -> None:
        from collections import Counter

        classes = Counter(r.class_name for r in BEST_KNOWN_VALUES)
        # Small instances: 2 per class per size, 3 sizes = 6 per class
        # Large: C1=9, C2=8, R1=12, R2=11, RC1=8, RC2=8
        assert classes["C1"] == 6 + 9
        assert classes["C2"] == 6 + 8
        assert classes["R1"] == 6 + 12
        assert classes["R2"] == 6 + 11
        assert classes["RC1"] == 6 + 8
        assert classes["RC2"] == 6 + 8

    def test_unique_instances(self) -> None:
        names = [r.instance for r in BEST_KNOWN_VALUES]
        assert len(names) == len(set(names))

    def test_no_unknown_bks(self) -> None:
        """All instances must have actual BKS values, not 'unknown'."""
        for rec in BEST_KNOWN_VALUES:
            assert rec.bks_vehicles > 0, f"{rec.instance}: vehicles must be positive"
            assert rec.bks_distance > 0, f"{rec.instance}: distance must be positive"

    def test_charging_time_and_count_unknown(self) -> None:
        """Published BKS does not report charging time or count."""
        for rec in BEST_KNOWN_VALUES:
            assert rec.bks_charging_time is None
            assert rec.bks_charging_count is None


class TestBKSLookup:
    """Test the get_bks lookup function."""

    def test_get_existing_small(self) -> None:
        rec = get_bks("c101C5")
        assert rec is not None
        assert rec.bks_vehicles == 2
        assert rec.bks_distance == 257.75

    def test_get_existing_large(self) -> None:
        rec = get_bks("c101_21")
        assert rec is not None
        assert rec.bks_vehicles == 12
        assert rec.bks_distance == 1053.83

    def test_get_nonexistent(self) -> None:
        assert get_bks("nonexistent") is None

    def test_get_all_returns_complete(self) -> None:
        assert len(get_all_bks()) == 92


class TestBKSValueCorrectness:
    """Spot-check BKS values against published tables."""

    @pytest.mark.parametrize(
        "instance, vehicles, distance",
        [
            ("c101C5", 2, 257.75),
            ("r104C5", 2, 136.69),
            ("rc108C5", 1, 253.93),
            ("c101C10", 3, 393.76),
            ("rc201C10", 1, 412.86),
            ("c103C15", 3, 384.29),
            ("rc204C15", 1, 384.86),  # VNS/TS improved over CPLEX
        ],
    )
    def test_small_instance_values(
        self, instance: str, vehicles: int, distance: float
    ) -> None:
        rec = get_bks(instance)
        assert rec is not None
        assert rec.bks_vehicles == vehicles
        assert abs(rec.bks_distance - distance) < 1e-6

    @pytest.mark.parametrize(
        "instance, vehicles, distance, source_ref",
        [
            ("c101_21", 12, 1053.83, "SSG"),
            ("c102_21", 11, 1051.38, "GS"),
            ("c106_21", 11, 1057.65, "HPH"),
            ("r101_21", 18, 1663.04, "HPH"),
            ("r104_21", 11, 1088.43, "SSG"),
            ("rc102_21", 14, 1552.08, "HPH"),
            ("rc208_21", 3, 836.29, "GS"),
        ],
    )
    def test_large_instance_values(
        self,
        instance: str,
        vehicles: int,
        distance: float,
        source_ref: str,
    ) -> None:
        rec = get_bks(instance)
        assert rec is not None
        assert rec.bks_vehicles == vehicles
        assert abs(rec.bks_distance - distance) < 1e-6
        assert rec.source_ref == source_ref

    def test_rc204_15_uses_vns_value(self) -> None:
        """RC204-15 BKS should be the VNS/TS value, not CPLEX upper bound."""
        rec = get_bks("rc204C15")
        assert rec is not None
        assert rec.bks_distance == 384.86  # VNS/TS, not 407.45
        assert "VNS" in rec.source_table


class TestInstanceNameMapping:
    """Verify instance name mapping between paper and file conventions."""

    def test_small_instance_names(self) -> None:
        for rec in BEST_KNOWN_VALUES:
            if rec.customer_count < 100:
                # C101-5 -> c101C5
                expected = rec.paper_name.lower().replace("-", "C")
                assert rec.instance == expected

    def test_large_instance_names(self) -> None:
        for rec in BEST_KNOWN_VALUES:
            if rec.customer_count == 100:
                # c101 -> c101_21
                assert rec.instance == f"{rec.paper_name}_21"


class TestSourceReferences:
    """Verify source reference data."""

    def test_all_sources_defined(self) -> None:
        for rec in BEST_KNOWN_VALUES:
            assert rec.source_ref in SOURCE_REFERENCES
            assert rec.compilation_ref in SOURCE_REFERENCES

    def test_ssg_doi(self) -> None:
        assert SOURCE_REFERENCES["SSG"].doi == "10.1287/trsc.2013.0490"
        assert SOURCE_REFERENCES["SSG"].year == 2014
        assert SOURCE_REFERENCES["SSG"].in_vor_collection is True

    def test_hph_doi(self) -> None:
        assert SOURCE_REFERENCES["HPH"].doi == "10.1016/j.ejor.2016.01.038"
        assert SOURCE_REFERENCES["HPH"].in_vor_collection is True

    def test_kc_doi(self) -> None:
        assert SOURCE_REFERENCES["KC"].doi == "10.1016/j.trc.2016.01.013"
        assert SOURCE_REFERENCES["KC"].in_vor_collection is True

    def test_gs_not_in_vor_collection(self) -> None:
        assert SOURCE_REFERENCES["GS"].in_vor_collection is False


class TestCompatibilityAssessment:
    """Verify model compatibility assessment."""

    def test_overall_incompatible(self) -> None:
        assessment = assess_compatibility()
        assert assessment.overall_compatible is False

    def test_five_dimensions(self) -> None:
        assessment = assess_compatibility()
        assert len(assessment.dimensions) == 5

    def test_charging_compatible(self) -> None:
        assessment = assess_compatibility()
        dim = next(d for d in assessment.dimensions if d.dimension == "charging_model")
        assert dim.compatible is True

    def test_objective_incompatible(self) -> None:
        assessment = assess_compatibility()
        dim = next(d for d in assessment.dimensions if d.dimension == "objective_function")
        assert dim.compatible is False

    def test_distance_incompatible(self) -> None:
        assessment = assess_compatibility()
        dim = next(d for d in assessment.dimensions if d.dimension == "distance_metric")
        assert dim.compatible is False

    def test_vehicle_params_compatible(self) -> None:
        assessment = assess_compatibility()
        dim = next(d for d in assessment.dimensions if d.dimension == "vehicle_parameters")
        assert dim.compatible is True

    def test_time_windows_compatible(self) -> None:
        assessment = assess_compatibility()
        dim = next(d for d in assessment.dimensions if d.dimension == "time_windows")
        assert dim.compatible is True

    def test_summary_mentions_no_gap(self) -> None:
        assessment = assess_compatibility()
        assert "no gap" in assessment.summary.lower()


class TestRunLabelValidation:
    """Test Stage 5.1 run label validation."""

    def test_valid_attempt_label(self) -> None:
        from evrptw.experiments.stage051_best_known import validate_stage051_run_label

        validate_stage051_run_label("stage05.1_best_known_attempt01")

    def test_valid_rerun_label(self) -> None:
        from evrptw.experiments.stage051_best_known import validate_stage051_run_label

        validate_stage051_run_label("stage05.1_best_known_rerun05")

    def test_invalid_label_missing_dot(self) -> None:
        from evrptw.experiments.stage051_best_known import validate_stage051_run_label

        with pytest.raises(ValueError, match="non-canonical"):
            validate_stage051_run_label("stage051_best_known_attempt01")

    def test_invalid_label_wrong_component(self) -> None:
        from evrptw.experiments.stage051_best_known import validate_stage051_run_label

        with pytest.raises(ValueError, match="non-canonical"):
            validate_stage051_run_label("stage05.1_benchmark_attempt01")
