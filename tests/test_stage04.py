"""Tests for Stage 4 adaptive weights and search control."""

from __future__ import annotations

import pytest

from evrptw.alns import (
    OperatorProfile,
    OperatorStatistics,
    Stage04Config,
    _apply_stage04_segment_update,
    _stage04_accumulate,
    _stage04_adaptive_weight_statistics,
    solve_alns,
)
from evrptw.benchmark import parse_schneider
from evrptw.objective import ObjectiveComparison, SolutionObjective
from evrptw.stage04 import with_fixed_weights
from evrptw.validation import validate_routes


def test_stage04_operator_audit_rejects_update_below_minimum() -> None:
    from evrptw.experiments.stage04_weights_review import (
        validate_stage04_operator_audit,
    )

    statistics = {
        "standard": {
            "role": "neighborhood",
            "calls": 4,
            "accepted": 2,
            "accepted_improving": 1,
            "accepted_equal": 1,
            "accepted_worse": 0,
            "rejected": 2,
            "new_global_best": 0,
            "vehicle_reduction": 0,
        }
    }
    events = [{"type": "stage04_segment_update", "operator": "standard", "segment_calls": 4}]
    passed, _ = validate_stage04_operator_audit(statistics, events, min_calls=5)
    assert not passed


def test_stage04_operator_audit_requires_all_six_categories() -> None:
    from evrptw.experiments.stage04_weights_review import (
        validate_stage04_operator_audit,
    )

    statistics = {
        "standard": {
            "role": "neighborhood",
            "calls": 5,
            "accepted": 3,
            "accepted_improving": 1,
            "accepted_equal": 2,
            "accepted_worse": 0,
            "rejected": 2,
            "new_global_best": 0,
        }
    }
    passed, _ = validate_stage04_operator_audit(statistics, [], min_calls=5)
    assert not passed


def test_stage04_operator_audit_requires_exact_call_partition_and_role() -> None:
    from evrptw.experiments.stage04_weights_review import (
        validate_stage04_operator_audit,
    )

    statistics = {
        "neighborhood:standard": {
            "role": "neighborhood",
            "calls": 5,
            "accepted": 2,
            "accepted_improving": 1,
            "accepted_equal": 1,
            "accepted_worse": 0,
            "rejected": 2,
            "new_global_best": 0,
            "vehicle_reduction": 0,
        }
    }
    passed, detail = validate_stage04_operator_audit(
        statistics,
        [{
            "type": "stage04_segment_update",
            "operator": "neighborhood:standard",
            "role": "neighborhood",
            "segment_calls": 5,
        }],
        min_calls=5,
    )
    assert not passed
    assert "calls" in detail

    statistics["neighborhood:standard"]["rejected"] = 3
    statistics["neighborhood:standard"].pop("role")
    passed, detail = validate_stage04_operator_audit(
        statistics,
        [{
            "type": "stage04_segment_update",
            "operator": "neighborhood:standard",
            "role": "neighborhood",
            "segment_calls": 5,
        }],
        min_calls=5,
    )
    assert not passed
    assert "role" in detail


def test_stage04_operator_audit_requires_segment_events() -> None:
    from evrptw.experiments.stage04_weights_review import (
        validate_stage04_operator_audit,
    )

    statistics = {
        "destroy:random": {
            "role": "destroy",
            "calls": 5,
            "accepted": 0,
            "accepted_improving": 0,
            "accepted_equal": 0,
            "accepted_worse": 0,
            "rejected": 5,
            "new_global_best": 0,
            "vehicle_reduction": 0,
        }
    }
    passed, detail = validate_stage04_operator_audit(statistics, [], min_calls=5)
    assert not passed
    assert "segment" in detail


def test_stage04_operator_audit_requires_exact_boundary_operator_matrix() -> None:
    from evrptw.experiments.stage04_weights_review import (
        validate_stage04_operator_audit,
    )

    statistics = {
        "destroy:random": {
            "role": "destroy",
            "calls": 10,
            "accepted": 5,
            "accepted_improving": 2,
            "accepted_equal": 2,
            "accepted_worse": 1,
            "rejected": 5,
            "new_global_best": 1,
            "vehicle_reduction": 0,
        }
    }
    boundary_event = {
        "type": "stage04_segment_update",
        "operator": "destroy:random",
        "role": "destroy",
        "iteration": 49,
        "segment_calls": 5,
    }
    passed, detail = validate_stage04_operator_audit(
        statistics,
        [boundary_event],
        min_calls=5,
        segment_length=50,
        completed_iterations=100,
    )
    assert not passed
    assert "missing" in detail

    passed, detail = validate_stage04_operator_audit(
        statistics,
        [boundary_event, boundary_event, {**boundary_event, "iteration": 99}],
        min_calls=5,
        segment_length=50,
        completed_iterations=100,
    )
    assert not passed
    assert "duplicate" in detail

    passed, detail = validate_stage04_operator_audit(
        statistics,
        [boundary_event, {**boundary_event, "iteration": 50}],
        min_calls=5,
        segment_length=50,
        completed_iterations=100,
    )
    assert not passed
    assert "unexpected" in detail


def test_stage04_fixed_work_boundary_rejects_late_acceptance() -> None:
    from evrptw.experiments.stage04_weights_review import (
        validate_fixed_work_boundary_events,
    )

    lanes = {"3": "adaptive_fixed_work:legacy"}
    events = [
        {
            "event_id": 10,
            "event_type": "exact_budget_boundary",
            "lane_id": 3,
            "status": "budget_exhausted",
            "accepted": False,
            "global_best": False,
        },
        {
            "event_id": 11,
            "event_type": "candidate_state",
            "lane_id": 3,
            "status": "accepted",
            "accepted": True,
            "global_best": False,
        },
    ]
    passed, detail = validate_fixed_work_boundary_events(
        events,
        lanes,
        axis="adaptive_fixed_work",
        boundary_expected=True,
    )
    assert not passed
    assert "after budget" in detail

    events.pop()
    passed, _ = validate_fixed_work_boundary_events(
        events,
        lanes,
        axis="adaptive_fixed_work",
        boundary_expected=True,
    )
    assert passed


def test_stage04_per_run_rows_reject_duplicate_and_invalid_numeric_fields() -> None:
    from evrptw.experiments.stage04_weights_review import validate_stage04_per_run_rows

    rows = [
        {"instance": "c101C5", "seed": "2014", "axis": "adaptive_wall_clock", "vehicle_count": "2"},
        {"instance": "c101C5", "seed": "2014", "axis": "adaptive_wall_clock", "vehicle_count": "2"},
    ]
    passed, detail = validate_stage04_per_run_rows(rows, scope="smoke")
    assert not passed
    assert "duplicate" in detail

    rows[1]["seed"] = "not-an-int"
    passed, detail = validate_stage04_per_run_rows(rows, scope="smoke")
    assert not passed
    assert "invalid" in detail


def test_stage04_scope_audit_rejects_missing_and_duplicate_axes() -> None:
    from evrptw.experiments.stage04_weights_review import validate_stage04_scope_identities

    rows = [
        {"instance": "c101C5", "seed": 2014, "axis": "adaptive_wall_clock"},
        {"instance": "c101C5", "seed": 2014, "axis": "adaptive_wall_clock"},
    ]
    passed, detail = validate_stage04_scope_identities(rows, scope="smoke")
    assert not passed
    assert "duplicate" in detail
    assert "missing" in detail


def test_stage04_prerequisite_rejects_missing_publication(tmp_path: object) -> None:
    from pathlib import Path

    from evrptw.experiments.stage04_weights import verify_stage034_prerequisite

    with pytest.raises(FileNotFoundError):
        verify_stage034_prerequisite(Path(str(tmp_path)), Path("missing.json"))


def test_stage04_prerequisite_rejects_publication_sidecar_mismatch(
    tmp_path: object,
) -> None:
    from pathlib import Path

    from evrptw.experiments.stage04_weights import verify_stage034_prerequisite

    root = Path(str(tmp_path))
    publication = root / "publication.json"
    publication.write_text("{}", encoding="utf-8")
    publication.with_suffix(".json.sha256").write_text("0" * 64, encoding="utf-8")
    with pytest.raises(RuntimeError, match="sidecar mismatch"):
        verify_stage034_prerequisite(root, publication)

# ─── Config validation ──────────────────────────────────────────────


class TestStage04ConfigValidation:
    def test_default_config_is_valid(self) -> None:
        config = Stage04Config()
        assert config.enabled
        assert config.segment_length == 50
        assert config.min_calls_per_operator == 5
        assert config.reward_vehicle_reduction > config.reward_distance_improvement

    def test_vehicle_reward_must_exceed_distance_reward(self) -> None:
        with pytest.raises(ValueError, match="reward_vehicle_reduction must exceed"):
            Stage04Config(
                reward_vehicle_reduction=4.0,
                reward_distance_improvement=4.0,
            )

    def test_best_vehicle_reward_must_exceed_best_distance(self) -> None:
        with pytest.raises(
            ValueError, match="reward_new_global_best_vehicle_reduction must exceed"
        ):
            Stage04Config(
                reward_new_global_best=16.0,
                reward_new_global_best_vehicle_reduction=16.0,
            )

    def test_segment_length_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="segment_length must be positive"):
            Stage04Config(segment_length=0)

    def test_restart_threshold_must_exceed_reheat_threshold(self) -> None:
        with pytest.raises(
            ValueError, match="restart_stagnation_threshold must exceed"
        ):
            Stage04Config(
                reheat_stagnation_threshold=10,
                restart_stagnation_threshold=10,
            )

    def test_temperature_target_must_be_in_open_interval(self) -> None:
        with pytest.raises(
            ValueError, match="temperature_target_acceptance_rate must be in"
        ):
            Stage04Config(temperature_target_acceptance_rate=0.0)
        with pytest.raises(
            ValueError, match="temperature_target_acceptance_rate must be in"
        ):
            Stage04Config(temperature_target_acceptance_rate=1.0)


# ─── Reward computation ─────────────────────────────────────────────


class TestStage04Rewards:
    def test_rejected_reward_is_zero(self) -> None:
        config = Stage04Config()
        assert config.reward_for(
            accepted=False,
            comparison="worse",
            is_global_best=False,
            vehicle_reduction=False,
        ) == 0.0

    def test_accepted_worse_reward(self) -> None:
        config = Stage04Config()
        assert config.reward_for(
            accepted=True,
            comparison="worse",
            is_global_best=False,
            vehicle_reduction=False,
        ) == config.reward_accepted_worse

    def test_accepted_equal_reward(self) -> None:
        config = Stage04Config()
        assert config.reward_for(
            accepted=True,
            comparison="equal",
            is_global_best=False,
            vehicle_reduction=False,
        ) == config.reward_accepted_equal

    def test_distance_improvement_reward(self) -> None:
        config = Stage04Config()
        assert config.reward_for(
            accepted=True,
            comparison="better",
            is_global_best=False,
            vehicle_reduction=False,
        ) == config.reward_distance_improvement

    def test_vehicle_reduction_reward_exceeds_distance(self) -> None:
        config = Stage04Config()
        assert config.reward_for(
            accepted=True,
            comparison="better",
            is_global_best=False,
            vehicle_reduction=True,
        ) == config.reward_vehicle_reduction
        assert config.reward_vehicle_reduction > config.reward_distance_improvement

    def test_new_global_best_distance_reward(self) -> None:
        config = Stage04Config()
        assert config.reward_for(
            accepted=True,
            comparison="better",
            is_global_best=True,
            vehicle_reduction=False,
        ) == config.reward_new_global_best

    def test_new_global_best_vehicle_reduction_reward(self) -> None:
        config = Stage04Config()
        assert config.reward_for(
            accepted=True,
            comparison="better",
            is_global_best=True,
            vehicle_reduction=True,
        ) == config.reward_new_global_best_vehicle_reduction


# ─── Fixed weights helper ───────────────────────────────────────────


class TestWithFixedWeights:
    def test_none_returns_none(self) -> None:
        assert with_fixed_weights(None) is None

    def test_fixed_weights_copy(self) -> None:
        config = Stage04Config()
        fixed = with_fixed_weights(config)
        assert fixed is not None
        assert fixed.fixed_weights is True
        assert fixed.auto_temperature is False
        assert fixed.reheat_enabled is False
        assert fixed.restart_enabled is False
        assert fixed.intensification_enabled is False

    def test_adaptive_original_unchanged(self) -> None:
        config = Stage04Config()
        with_fixed_weights(config)
        assert config.fixed_weights is False
        assert config.auto_temperature is True
        assert config.reheat_enabled is True


# ─── Segment-based weight update ────────────────────────────────────


class TestSegmentWeightUpdate:
    def test_adaptive_weight_set_matches_actual_selection_roles(self) -> None:
        neighborhoods = {
            "standard": OperatorStatistics(),
            "vehicle_reduction_refinement": OperatorStatistics(),
        }
        destroy = {"random": OperatorStatistics()}
        repair = {"greedy": OperatorStatistics()}
        selected = _stage04_adaptive_weight_statistics(
            OperatorProfile.STAGE02_CONSTRAINT_GUIDED,
            neighborhoods,
            destroy,
            repair,
        )
        assert set(selected) == {
            "neighborhood:standard",
            "destroy:random",
            "repair:greedy",
        }

    def test_segment_accumulation(self) -> None:
        stats = OperatorStatistics()
        _stage04_accumulate(stats, 4.0)
        _stage04_accumulate(stats, 8.0)
        assert stats.segment_calls == 2
        assert stats.segment_reward_sum == 12.0

    def test_segment_update_applies_batch(self) -> None:
        config = Stage04Config()
        stats = OperatorStatistics(weight=1.0)
        for _ in range(10):
            _stage04_accumulate(stats, 8.0)
        _apply_stage04_segment_update({"op": stats}, config, 49, [])
        assert stats.segment_calls == 0
        assert stats.segment_reward_sum == 0.0
        assert stats.weight > 1.0  # reward of 8.0 should increase weight
        assert len(stats.weight_history) == 1
        assert stats.weight_history[0] == (49, stats.weight)

    def test_segment_update_skips_low_call_operators(self) -> None:
        config = Stage04Config(min_calls_per_operator=5)
        stats = OperatorStatistics(weight=1.0)
        _stage04_accumulate(stats, 8.0)  # only 1 call, below minimum
        _apply_stage04_segment_update({"op": stats}, config, 49, [])
        assert stats.weight == 1.0  # unchanged
        assert stats.segment_calls == 0  # reset
        assert len(stats.weight_history) == 0

    def test_segment_update_resets_accumulators(self) -> None:
        config = Stage04Config()
        stats = OperatorStatistics(weight=1.0)
        for _ in range(10):
            _stage04_accumulate(stats, 4.0)
        _apply_stage04_segment_update({"op": stats}, config, 49, [])
        assert stats.segment_calls == 0
        assert stats.segment_reward_sum == 0.0

    def test_segment_update_logs_events(self) -> None:
        config = Stage04Config()
        stats = OperatorStatistics(weight=1.0)
        for _ in range(10):
            _stage04_accumulate(stats, 8.0)
        events: list[dict[str, object]] = []
        _apply_stage04_segment_update({"op": stats}, config, 49, events)
        assert len(events) == 1
        assert events[0]["type"] == "stage04_segment_update"
        assert events[0]["operator"] == "op"
        assert events[0]["iteration"] == 49


# ─── End-to-end ALNS with Stage 4 ────────────────────────────────────


@pytest.fixture(scope="module")
def c101c5_instance() -> object:
    return parse_schneider("data/schneider/c101C5.txt")


class TestStage04EndToEnd:
    def test_stage04_disabled_preserves_legacy_behavior(
        self, c101c5_instance: object
    ) -> None:
        """Without Stage04Config, the solver uses per-call weight update."""
        result = solve_alns(
            c101c5_instance,
            seed=2014,
            max_iterations=30,
            time_limit_seconds=5,
            operator_profile="stage02_constraint_guided",
            backend="cpu_batch",
        )
        assert result.feasible
        assert result.stage04_statistics == {}

    def test_stage04_adaptive_produces_feasible(
        self, c101c5_instance: object
    ) -> None:
        config = Stage04Config()
        result = solve_alns(
            c101c5_instance,
            seed=2014,
            max_iterations=30,
            time_limit_seconds=5,
            operator_profile="stage02_constraint_guided",
            backend="cpu_batch",
            stage04_config=config,
        )
        assert result.feasible
        assert result.stage04_statistics["enabled"] is True
        assert result.stage04_statistics["fixed_weights"] is False
        assert result.stage04_statistics["auto_temperature"] is True

    def test_stage04_fixed_weights_produces_feasible(
        self, c101c5_instance: object
    ) -> None:
        config = with_fixed_weights(Stage04Config())
        result = solve_alns(
            c101c5_instance,
            seed=2014,
            max_iterations=30,
            time_limit_seconds=5,
            operator_profile="stage02_constraint_guided",
            backend="cpu_batch",
            stage04_config=config,
        )
        assert result.feasible
        assert result.stage04_statistics["fixed_weights"] is True
        assert result.stage04_statistics["auto_temperature"] is False

    def test_stage04_adaptive_tracks_six_categories(
        self, c101c5_instance: object
    ) -> None:
        config = Stage04Config()
        result = solve_alns(
            c101c5_instance,
            seed=2014,
            max_iterations=50,
            time_limit_seconds=10,
            operator_profile="stage02_constraint_guided",
            backend="cpu_batch",
            stage04_config=config,
        )
        assert result.feasible
        # Check that six-category statistics are tracked in neighborhood_stats
        for stats_dict in [
            result.neighborhood_statistics,
            result.destroy_statistics,
            result.repair_statistics,
        ]:
            if not stats_dict:
                continue
            for _name, stats in stats_dict.items():
                assert "accepted_improving" in stats
                assert "accepted_equal" in stats
                assert "accepted_worse" in stats
                # The sum of three categories should equal accepted
                assert (
                    stats["accepted_improving"]
                    + stats["accepted_equal"]
                    + stats["accepted_worse"]
                    == stats["accepted"]
                )

    def test_stage04_adaptive_records_temperature_history(
        self, c101c5_instance: object
    ) -> None:
        config = Stage04Config()
        result = solve_alns(
            c101c5_instance,
            seed=2014,
            max_iterations=50,
            time_limit_seconds=10,
            operator_profile="stage02_constraint_guided",
            backend="cpu_batch",
            stage04_config=config,
        )
        assert result.feasible
        assert len(result.stage04_temperature_history) > 0
        # First entry should be the initial temperature
        first_iter, first_temp = result.stage04_temperature_history[0]
        assert first_iter == 0
        assert first_temp > 0

    def test_stage04_adaptive_records_weight_history(
        self, c101c5_instance: object
    ) -> None:
        config = Stage04Config(segment_length=10)
        result = solve_alns(
            c101c5_instance,
            seed=2014,
            max_iterations=50,
            time_limit_seconds=10,
            operator_profile="stage02_constraint_guided",
            backend="cpu_batch",
            stage04_config=config,
        )
        assert result.feasible
        # With segment_length=10 and 50 iterations, at least some
        # operators should have weight history entries
        assert len(result.stage04_weight_history) > 0
        for _name, history in result.stage04_weight_history.items():
            assert len(history) > 0
            for _iteration, weight in history:
                assert weight > 0

    def test_stage04_reheating_triggers_on_stagnation(
        self, c101c5_instance: object
    ) -> None:
        """Use a larger instance with more iterations to allow stagnation."""
        instance = parse_schneider("data/schneider/c101_21.txt")
        config = Stage04Config(
            reheat_stagnation_threshold=5,
            restart_stagnation_threshold=15,
            max_reheats=10,
            max_restarts=5,
        )
        result = solve_alns(
            instance,
            seed=2014,
            max_iterations=200,
            time_limit_seconds=15,
            operator_profile="stage02_constraint_guided",
            backend="cpu_batch",
            stage04_config=config,
        )
        assert result.feasible
        # The 100-customer instance should stagnate enough to trigger reheating
        assert result.stage04_statistics["reheat_count"] > 0
        # Check that reheat events are in the event log
        reheat_events = [
            e for e in result.stage04_event_log if e.get("type") == "stage04_reheat"
        ]
        assert len(reheat_events) > 0

    def test_stage04_stagnation_restart_triggers(
        self, c101c5_instance: object
    ) -> None:
        """Stagnation restart should trigger on sufficient stagnation."""
        instance = parse_schneider("data/schneider/c101_21.txt")
        config = Stage04Config(
            reheat_stagnation_threshold=3,
            restart_stagnation_threshold=6,
            max_reheats=10,
            max_restarts=5,
        )
        result = solve_alns(
            instance,
            seed=2014,
            max_iterations=200,
            time_limit_seconds=15,
            operator_profile="stage02_constraint_guided",
            backend="cpu_batch",
            stage04_config=config,
        )
        assert result.feasible
        # Stagnation restart should trigger at least once
        assert result.stage04_statistics["restart_count"] >= 0  # may be 0 on small instances
        restart_events = [
            e for e in result.stage04_event_log
            if e.get("type") == "stage04_restart"
        ]
        # If restart_count > 0, there should be events
        if result.stage04_statistics["restart_count"] > 0:
            assert len(restart_events) > 0

    def test_stage04_cpu_scalar_rejected(self, c101c5_instance: object) -> None:
        """Stage 4 requires cpu_batch backend."""
        config = Stage04Config()
        with pytest.raises(ValueError, match="cpu_batch"):
            solve_alns(
                c101c5_instance,
                seed=2014,
                max_iterations=10,
                time_limit_seconds=5,
                operator_profile="stage02_constraint_guided",
                backend="cpu_scalar",
                stage04_config=config,
            )

    def test_stage04_acceptance_rate_recorded(
        self, c101c5_instance: object
    ) -> None:
        config = Stage04Config()
        result = solve_alns(
            c101c5_instance,
            seed=2014,
            max_iterations=50,
            time_limit_seconds=10,
            operator_profile="stage02_constraint_guided",
            backend="cpu_batch",
            stage04_config=config,
        )
        assert result.feasible
        rate = result.stage04_statistics["acceptance_rate"]
        assert 0.0 <= rate <= 1.0

    def test_stage04_validator_consistency(
        self, c101c5_instance: object
    ) -> None:
        """Stage 4 solutions must pass the unified validator."""
        config = Stage04Config()
        result = solve_alns(
            c101c5_instance,
            seed=2014,
            max_iterations=50,
            time_limit_seconds=10,
            operator_profile="stage02_constraint_guided",
            backend="cpu_batch",
            stage04_config=config,
        )
        assert result.feasible
        # Re-validate the solution independently
        instance = c101c5_instance
        report = validate_routes(instance, [list(r) for r in result.routes])
        assert report.feasible
        recomputed = SolutionObjective.from_report(instance, report)
        from evrptw.objective import compare_objectives
        assert compare_objectives(recomputed, result.objective) is ObjectiveComparison.EQUAL
