"""Tests for L7 - Observability, Evaluation and Feedback Improvement.

All offline and deterministic: every score, split, attribution and
before/after direction asserted here comes from real graph runs of L2's
loop over the fixed dataset (no mocking of any framework internals; the
only scripted part is the FakeModel, which IS the system under test).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lessons.l7_observability_and_evaluation.main import (  # noqa: E402
    DATASET,
    HUMAN_ANNOTATIONS,
    JUDGE_PASS_THRESHOLD,
    PER_CASE_TOKEN_BUDGET,
    aggregate_score,
    attribution_table,
    attribute_failure,
    budget_check,
    calibrate,
    combined_score,
    group_scores,
    make_offline_judge,
    result_score,
    revise_script,
    run_case,
    run_dataset,
    run_reflected,
    split_by_group,
    trajectory_breakdown,
    trajectory_score,
)
from lessons.l2_agent_loop.main import FakeModel  # noqa: E402


@pytest.fixture(scope="module")
def baseline():
    """One full baseline pass over the dataset, reused by all tests."""
    return run_dataset()


@pytest.fixture(scope="module")
def reflected():
    """One full pass with the post-reflection scripts."""
    return run_reflected()


def by_id(records, case_id):
    return next(r for r in records if r.case.case_id == case_id)


# -- dataset -----------------------------------------------------------------


def test_dataset_shape():
    """12 cases over 4 source groups, unique ids, every case solvable but one designed."""
    assert len(DATASET) == 12
    groups = {case.source_group for case in DATASET}
    assert groups == {"specbook-a", "specbook-b", "vendor-datasheet", "legacy-notes"}
    assert len({case.case_id for case in DATASET}) == 12


def test_run_record_counts_directly(baseline):
    """RunRecord numbers are counted from the run, not inferred: steps/journal."""
    record = by_id(baseline, "l7-c01")
    assert record.stop_reason == "success"
    assert record.steps == 2  # one tool turn + one final turn
    assert record.tokens_used == 100  # L2 fake usage: 50 tokens x 2 turns
    assert len(record.tool_calls) == 1
    assert "118.11" in record.final_answer


def test_semantic_loop_survives_runtime_detector():
    """c02's duplicate (3 vs 3.0) evades L2's exact-match no_progress detector."""
    record = run_case(next(c for c in DATASET if c.case_id == "l7-c02"))
    assert record.stop_reason == "success"  # the runtime did NOT flag it
    assert trajectory_breakdown(record)["loops"] == 1  # our evaluator does


# -- result vs trajectory ----------------------------------------------------


def test_result_correct_trajectory_bad(baseline):
    """c02: result 1.0 but trajectory 0.75 -- only the trajectory evaluator sees it."""
    record = by_id(baseline, "l7-c02")
    assert result_score(record) == 1.0
    assert trajectory_score(record) == 0.75
    assert trajectory_breakdown(record) == {"loops": 1, "wasted_steps": 1, "wrong_termination": 0}


def test_result_wrong_trajectory_clean(baseline):
    """c07 (and c06): result 0.0 but trajectory 1.0 -- only the result evaluator sees it."""
    for case_id in ("l7-c06", "l7-c07"):
        record = by_id(baseline, case_id)
        assert result_score(record) == 0.0
        assert trajectory_score(record) == 1.0


def test_wrong_termination_on_solvable_case(baseline):
    """A budget stop on a solvable case is a trajectory failure, not just a 0 score."""
    record = by_id(baseline, "l7-c04")
    assert record.stop_reason == "budget"
    assert record.case.solvable
    assert trajectory_breakdown(record)["wrong_termination"] == 1


# -- group split -------------------------------------------------------------


def test_group_split_is_disjoint_complete_and_deterministic():
    """Same source group never crosses sets; split is seed-deterministic."""
    split = split_by_group(DATASET)
    train, dev, test = split["train"], split["dev"], split["test"]
    assert not (train & dev) and not (train & test) and not (dev & test)
    assert train | dev | test == {"specbook-a", "specbook-b", "vendor-datasheet", "legacy-notes"}
    assert (len(train), len(dev), len(test)) == (2, 1, 1)  # ratios 0.5/0.25/0.25
    assert split == split_by_group(DATASET)  # same seed -> same split
    # every case of a group lands in exactly the set its group landed in
    for case in DATASET:
        sets = [s for s in (train, dev, test) if case.source_group in s]
        assert len(sets) == 1


# -- attribution -------------------------------------------------------------


def test_attribution_labels_constructed_cases(baseline):
    """Each constructed failure maps to its expected layer, incl. `unknown`."""
    table = {cause: set(ids) for cause, ids in attribution_table(baseline).items()}
    assert table["budget_stop"] == {"l7-c04"}
    assert table["tool_error"] == {"l7-c05"}
    assert table["routing_error"] == {"l7-c02", "l7-c08"}
    assert table["model_error"] == {"l7-c06", "l7-c07"}
    assert table["unknown"] == {"l7-c11"}
    assert table["ok"] == {"l7-c01", "l7-c03", "l7-c09", "l7-c10", "l7-c12"}


def test_unknown_means_faithful_to_tool_but_label_disagrees(baseline):
    """c11 is `unknown` BECAUSE the answer matches the tool output, not the label."""
    record = by_id(baseline, "l7-c11")
    from lessons.l7_observability_and_evaluation.main import faithful_to_tool

    assert faithful_to_tool(record) is True
    assert attribute_failure(record) == "unknown"
    # c06's answer (200) contradicts its own tool (196.85) -> model_error, not unknown
    assert faithful_to_tool(by_id(baseline, "l7-c06")) is False


# -- judges / calibration / reward hacking -----------------------------------


def test_human_annotations_join_results(baseline):
    """Recorded human verdicts agree with rules where they judged, except unknown."""
    for case_id, annotation in HUMAN_ANNOTATIONS.items():
        rule = "pass" if result_score(by_id(baseline, case_id)) == 1.0 else "fail"
        if annotation["verdict"] == "unknown":
            continue  # escalated to review, not counted as (dis)agreement
        assert annotation["verdict"] == rule, case_id


def test_fake_judge_calibration_numbers(baseline):
    """v1 agrees 9/12 (fooled 3x); v2 agrees 11/12 (over-strict 1x)."""
    v1 = calibrate(make_offline_judge("v1"), baseline)
    v2 = calibrate(make_offline_judge("v2"), baseline)
    assert (v1["agree"], v1["total"]) == (9, 12)
    assert (v2["agree"], v2["total"]) == (11, 12)
    assert v1["matrix"][("pass", "fail")] == 3  # judge pass / truth fail: the hacks
    assert v2["matrix"][("fail", "pass")] == 1  # judge fail / truth pass: c10 paraphrase


def test_reward_hacking_v1_high_v2_low(baseline):
    """The mandated counterexample: SAME case, v1 >= threshold AND v2 < threshold."""
    case = by_id(baseline, "l7-c07").case
    answer = by_id(baseline, "l7-c07").final_answer
    judged_v1 = make_offline_judge("v1")(case, answer)
    judged_v2 = make_offline_judge("v2")(case, answer)
    assert judged_v1["score"] >= JUDGE_PASS_THRESHOLD  # fluent wrong answer scores HIGH
    assert judged_v2["score"] < JUDGE_PASS_THRESHOLD  # evidence check catches it
    assert result_score(by_id(baseline, "l7-c07")) == 0.0  # and the task truly failed


# -- budget ------------------------------------------------------------------


def test_budget_check_flags_overruns(baseline):
    """Per-case and total budget checks flag exactly the designed violations."""
    check = budget_check(baseline)
    assert check["per_case_violations"] == ["l7-c04"]
    assert check["total_tokens"] == 1500
    assert check["total_over_budget"] is True  # 1500 > declared 1400


def test_group_cost_aggregation(baseline):
    """Simulated token cost aggregates per source group exactly."""
    from lessons.l7_observability_and_evaluation.main import group_cost

    costs = group_cost(baseline)
    assert {g: stats["tokens"] for g, stats in costs.items()} == {
        "specbook-a": 650,
        "specbook-b": 450,
        "vendor-datasheet": 300,
        "legacy-notes": 100,
    }
    assert sum(stats["tokens"] for stats in costs.values()) == 1500


# -- reflection (evaluator-optimizer) -----------------------------------------


def test_reflection_fixes_failing_cases(baseline, reflected):
    """Every targeted failing case passes after reflection, deterministically."""
    for case_id in ("l7-c02", "l7-c04", "l7-c05", "l7-c06", "l7-c07", "l7-c08"):
        before, after = by_id(baseline, case_id), by_id(reflected, case_id)
        assert result_score(before) < 1.0 or trajectory_score(before) < 1.0
        assert result_score(after) == 1.0, case_id
        assert trajectory_score(after) == 1.0, case_id


def test_reflection_no_result_regression_on_passing_cases(baseline, reflected):
    """Cases that passed before still pass after (c09 dips in trajectory only)."""
    for case_id in ("l7-c01", "l7-c03", "l7-c09", "l7-c10", "l7-c12"):
        assert result_score(by_id(baseline, case_id)) == 1.0
        assert result_score(by_id(reflected, case_id)) == 1.0


def test_unknown_case_stays_failing_and_reported(reflected):
    """c11 does not respond to the reflection and is reported as unknown."""
    still = [r.case.case_id for r in reflected if result_score(r) < 1.0]
    assert still == ["l7-c11"]
    assert attribute_failure(by_id(reflected, "l7-c11")) == "unknown"


def test_minor_group_regression_is_scripted_and_visible(baseline, reflected):
    """vendor-datasheet regresses slightly while the aggregate improves."""
    before, after = group_scores(baseline), group_scores(reflected)
    assert after["vendor-datasheet"] < before["vendor-datasheet"]  # the dip
    # and it is exactly the added verification call on c09:
    assert len(by_id(reflected, "l7-c09").tool_calls) == 2
    assert trajectory_score(by_id(reflected, "l7-c09")) == 0.95
    # every OTHER group improved or held, and the aggregate went up:
    for group in before:
        if group != "vendor-datasheet":
            assert after[group] >= before[group]
    assert aggregate_score(reflected) > aggregate_score(baseline)


def test_reflected_budget_back_within_limit(reflected):
    """The reflection also brought the total token spend under the declared budget."""
    check = budget_check(reflected)
    assert check["total_tokens"] == 1300
    assert check["total_over_budget"] is False
    assert check["per_case_violations"] == []


def test_revise_script_leaves_unlisted_cases_untouched():
    """Only listed cases get a revised script; everything else keeps its baseline."""
    for case in DATASET:
        revised = revise_script(case)
        if case.case_id not in {"l7-c02", "l7-c04", "l7-c05", "l7-c06", "l7-c07", "l7-c08", "l7-c09"}:
            assert revised is None
