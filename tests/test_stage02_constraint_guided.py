from __future__ import annotations

from pathlib import Path

from evrptw.experiments.stage02_constraint_guided_review import (
    _accepted_feasible_constraint_candidate,
    _blocking_review_findings,
    _comparison_metric_integrity,
    _dynamic_event_counts_ok,
    _load_comparison_metric_rows,
    _load_comparison_rows,
    _review_provenance,
    _run_manifest_paths,
)
from evrptw.experiments.stage02_route_reduction import (
    CONSTRAINT_GUIDED_ALGORITHM,
    load_config,
)


def test_stage02_constraint_guided_config_keeps_formal_scope_and_quality_baseline() -> None:
    config = load_config(Path("configs/stage02_constraint_guided.toml"))

    assert config.algorithm == CONSTRAINT_GUIDED_ALGORITHM
    assert config.operator_profile.value == "stage02_constraint_guided"
    assert config.comparison_per_run.as_posix().endswith(
        "experiments/summaries/stage02_quality_attempt02_per_run_results.csv"
    )
    assert config.comparison_label == "stage02_2"
    assert config.comparison_gate == "stage02_2_best_objective"
    assert len(config.instances) == 12
    assert config.seeds == (2014, 2015, 2016)
    assert config.time_limit_seconds == 30.0
    assert config.max_iterations == 1000
    assert config.threads == 1
    operators = config.vehicle_operator_config
    assert operators.small_removal_min_fraction == 0.05
    assert operators.small_removal_max_fraction == 0.10
    assert operators.medium_removal_min_fraction == 0.10
    assert operators.medium_removal_max_fraction == 0.20
    assert operators.large_removal_min_fraction == 0.20
    assert operators.large_removal_max_fraction == 0.35
    assert operators.medium_stagnation_threshold == 4
    assert operators.large_stagnation_threshold == 8
    assert operators.exploration_period == 3
    assert operators.quality_probe_exact_evaluation_budget == 2
    assert operators.quality_route_segment_probe_exact_evaluation_budget == 4
    assert operators.vehicle_reduction_refinement_exact_evaluation_budget == 512
    assert operators.constraint_lane_time_budget_seconds == 0.1


def test_review_manifest_is_not_counted_as_run_manifest_on_repeat(tmp_path: Path) -> None:
    (tmp_path / "stage02_constraint_guided_attempt_manifest.json").write_text(
        "{}\n", encoding="utf-8"
    )
    (tmp_path / "review_manifest.json").write_text("{}\n", encoding="utf-8")

    assert _run_manifest_paths(tmp_path) == [
        tmp_path / "stage02_constraint_guided_attempt_manifest.json"
    ]


def test_pending_rerun_does_not_mask_other_review_failures() -> None:
    findings = [
        {"finding": "raw_solution_validator", "status": "fail"},
        {"finding": "independent_complete_rerun_replay", "status": "pending"},
        {"finding": "runner_hard_gates", "status": "pending"},
    ]

    assert _blocking_review_findings(findings, True) == ["raw_solution_validator"]


def test_constraint_acceptance_requires_a_feasible_candidate_event() -> None:
    ranking_event = {
        "status": "candidate_proposed",
        "accepted": "True",
        "candidate_feasible": "False",
    }
    candidate_event = {
        "status": "candidate_proposed",
        "accepted": "True",
        "candidate_feasible": "True",
    }

    assert not _accepted_feasible_constraint_candidate(ranking_event)
    assert _accepted_feasible_constraint_candidate(candidate_event)


def test_dynamic_count_checker_rejects_zero_actual_for_selected_tier(
    tmp_path: Path,
) -> None:
    (tmp_path / "stage02_constraint_guided_parameters.toml").write_text(
        """
[vehicle_operators]
small_removal_min_fraction = 0.05
small_removal_max_fraction = 0.10
medium_removal_min_fraction = 0.10
medium_removal_max_fraction = 0.20
large_removal_min_fraction = 0.20
large_removal_max_fraction = 0.35
""",
        encoding="utf-8",
    )

    valid, invalid = _dynamic_event_counts_ok(
        Path.cwd(),
        tmp_path,
        [
            {
                "instance": "c101C5",
                "status": "time_limit",
                "removal_tier": "small",
                "removal_size_requested": "1",
                "removal_size_actual": "0",
            }
        ],
    )

    assert valid is False
    assert invalid == 1


def test_review_provenance_captures_dependency_and_runtime_hashes() -> None:
    environment = {
        "captured_environment": {
            "native_extension": str(
                Path.cwd()
                / ".venv/lib/python3.13/site-packages/evrptw/_core.cpython-313-darwin.so"
            ),
            "packages": {"pytest": "9.1.0"},
            "python": {"version": "3.13.13", "executable": "/tmp/python"},
        }
    }

    provenance = _review_provenance(Path.cwd(), environment)

    assert provenance["complete"] is True
    assert set(provenance["files"]) == {
        "pyproject.toml",
        "uv.lock",
        "native_extension",
        "review_cli",
    }


def test_stage02_2_readiness_metric_supplement_is_complete() -> None:
    comparison_rows = _load_comparison_rows(Path("results/stage02-quality_attempt02"))
    rows = _load_comparison_metric_rows(
        Path.cwd(), Path("results/stage02-quality_attempt02")
    )

    assert len(rows) == 36
    integrity_ok, _ = _comparison_metric_integrity(
        Path.cwd(), comparison_rows, rows
    )
    assert integrity_ok
    assert all(
        row[metric] not in {"", "not_recorded_in_stage02_2_summary"}
        for row in rows
        for metric in (
            "effective_iterations",
            "cache_hits",
            "cache_misses",
            "unique_route_evaluations",
            "removal_tier_counts",
            "constraint_operator_statistics_json",
        )
    )


def test_stage02_2_metric_supplement_rejects_protocol_drift() -> None:
    comparison_rows = _load_comparison_rows(Path("results/stage02-quality_attempt02"))
    rows = _load_comparison_metric_rows(
        Path.cwd(), Path("results/stage02-quality_attempt02")
    )
    rows[0] = dict(rows[0], operator_profile="baseline")

    integrity_ok, observed = _comparison_metric_integrity(
        Path.cwd(), comparison_rows, rows
    )

    assert not integrity_ok
    assert observed["protocol_ok"] is False
