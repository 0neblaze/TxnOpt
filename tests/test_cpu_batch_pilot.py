from __future__ import annotations

from evrptw.experiments.cpu_batch_pilot_review import evaluate_paired_gate


def test_cpu_paired_gate_requires_correctness_and_two_improved_families() -> None:
    rows = [
        {
            "instance": instance,
            "seed": seed,
            "valid": True,
            "saving_fraction": saving,
            "fixed_iterations": 40,
        }
        for instance, saving in (
            ("c101C5", -0.05),
            ("c101_21", 0.12),
            ("r101_21", 0.11),
            ("rc101_21", -0.01),
        )
        for seed in (2014, 2015, 2016)
    ]

    decision = evaluate_paired_gate(rows)

    assert decision["status"] == "ADOPT_CPU_BATCH_DEFAULT"
    assert decision["positive_families"] == 2
    assert decision["median_family_saving_fraction"] == 0.11

    rows[3]["valid"] = False
    assert evaluate_paired_gate(rows)["status"] == "KEEP_CPU_SCALAR_DEFAULT"

    rows[3]["valid"] = True
    rows[3]["fixed_iterations"] = 1
    smoke = evaluate_paired_gate(rows)
    assert smoke["protocol_complete"] is False
    assert smoke["status"] == "KEEP_CPU_SCALAR_DEFAULT"

    rows[3]["fixed_iterations"] = 40
    rows[0]["valid"] = False
    c5_failure = evaluate_paired_gate(rows)
    assert c5_failure["all_pairs_valid"] is False
    assert c5_failure["status"] == "KEEP_CPU_SCALAR_DEFAULT"
