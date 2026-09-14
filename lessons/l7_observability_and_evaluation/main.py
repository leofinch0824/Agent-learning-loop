"""L7 - Observability, Evaluation and Feedback Improvement.

The system under test is L2's agent loop (imported, not reimplemented):
``build_loop_graph`` driven by scripted FakeModels, run over a small fixed
PCB dataset of 12 cases with 4 source groups. This file is fully OFFLINE and
deterministic -- no MLflow, no server, no real LLM. The server-dependent
twin lives in ``mlflow_demo.py``; the real-LLM judge lives in
``live_demo.py``.

    dataset (12 cases, 4 source groups)
        │
        v
    L2 loop graph + FakeModel scripts  ──>  RunRecord per case
        │                                          │
        │         ┌────────────────┬───────────────┼───────────────┐
        │         v                v               v               v
        │   result evaluator   trajectory     failure        rule/human/
        │   (params match)     evaluator      attribution     judge eval
        │                                          │
        └── group-aware split (no source crosses sets)
                                                   v
                            reflection (evaluator-optimizer): failure summary
                            -> revised scripts -> re-run SAME dataset -> compare

Run it:

    poetry run python lessons/l7_observability_and_evaluation/main.py

Design promise: every printed number comes from a real (offline) graph run
counted directly -- model turns from the final state, tool executions from
the journal -- and test_main.py pins the exact scores, splits, attributions
and before/after directions asserted here.
"""

from __future__ import annotations

import json
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lib import banner  # noqa: E402

from lessons.l2_agent_loop.main import (  # noqa: E402
    USAGE,
    FakeModel,
    build_loop_graph,
    clear_tool_journal,
    final_msg,
    initial_state,
    tool_call_msg,
    tool_journal,
)


# ---------------------------------------------------------------------------
# 1. The evaluation dataset: 12 PCB cases across 4 source groups
# ---------------------------------------------------------------------------
# `source_group` is the DOCUMENT ORIGIN of the case. Real evaluation data is
# grouped by origin: near-duplicate rows from one document must never land in
# different splits (section 3 / demo 3).
#
# Failure modes are DELIBERATELY distributed so each evaluator misses what the
# other catches:
#   c02  result CORRECT, trajectory BAD (one semantic loop then succeeds)
#   c06/c07  result WRONG, trajectory clean (confident mental math / hack)
#   c11  result wrong, model faithful to the tool -> cause `unknown` (label
#        suspect), stays failing after reflection (报告未知)


@dataclass(frozen=True)
class Case:
    case_id: str
    source_group: str
    question: str
    expected_params: tuple[tuple[float, str], ...]  # (value, unit) ALL must match
    evidences: tuple[str, ...]  # exact substrings a grounded answer must quote
    min_tool_calls: int  # minimum executions a good trajectory needs
    solvable: bool = True
    script: list[dict] = field(default_factory=list)  # scripted FakeModel turns


def turn_tool(call_id: str, name: str, args: dict) -> dict:
    """One scripted model turn that requests a tool call (L2 wire format)."""
    return {"message": tool_call_msg(call_id, name, args), "usage": USAGE}


def turn_final(text: str) -> dict:
    """One scripted model turn that closes the loop with a final answer."""
    return {"message": final_msg(text), "usage": USAGE}


DATASET: list[Case] = [
    # -- specbook-a (工艺规范 A): happy path, loop-then-succeed, spec, runaway
    Case(
        "l7-c01", "specbook-a", "3 mm 是多少 mil？",
        ((118.11, "mil"),), ("118.11 mil",), min_tool_calls=1,
        script=[
            turn_tool("c1", "unit_convert", {"value": 3, "from_unit": "mm", "to_unit": "mil"}),
            turn_final("3 mm 约为 118.11 mil。"),
        ],
    ),
    Case(
        # One SEMANTIC loop then success: value 3 vs 3.0 differ as JSON, so
        # L2's no-progress detector does NOT fire, but the call is a duplicate
        # in substance -- only the trajectory evaluator catches it.
        "l7-c02", "specbook-a", "3 mm 是多少 mil？",
        ((118.11, "mil"),), ("118.11 mil",), min_tool_calls=1,
        script=[
            turn_tool("c1", "unit_convert", {"value": 3, "from_unit": "mm", "to_unit": "mil"}),
            turn_tool("c2", "unit_convert", {"value": 3.0, "from_unit": "mm", "to_unit": "mil"}),
            turn_final("3 mm 约为 118.11 mil。"),
        ],
    ),
    Case(
        "l7-c03", "specbook-a", "默认过孔直径是多少 mm？",
        ((0.3, "mm"),), ("0.3 mm",), min_tool_calls=1,
        script=[
            turn_tool("c1", "lookup_spec", {"item": "default_via_diameter_mm"}),
            turn_final("默认过孔直径为 0.3 mm。"),
        ],
    ),
    Case(
        # Solvable, but the scripted model never converges: distinct converts
        # forever -> the runtime stops it at MAX_STEPS (budget fingerprint).
        "l7-c04", "specbook-a", "3 mm 是多少 mil？内层最小线宽是多少 mm？",
        ((118.11, "mil"), (0.127, "mm")), ("118.11 mil", "0.127 mm"), min_tool_calls=2,
        script=[
            turn_tool(f"b{i}", "unit_convert", {"value": i, "from_unit": "mm", "to_unit": "mil"})
            for i in range(1, 11)
        ],
    ),
    # -- specbook-b (工艺规范 B): tool errors, mental math, hack, no-progress
    Case(
        # Three consecutive DISTINCT-args errors (unknown unit each time) ->
        # L2 declares tool_error; the trailing final answer is never reached.
        "l7-c05", "specbook-b", "把 3、4、5 mm 换算成 mil",
        ((118.11, "mil"),), ("118.11 mil",), min_tool_calls=1,
        script=[
            turn_tool("c1", "unit_convert", {"value": 3, "from_unit": "mm", "to_unit": "milimeter"}),
            turn_tool("c2", "unit_convert", {"value": 4, "from_unit": "mm", "to_unit": "milimeter"}),
            turn_tool("c3", "unit_convert", {"value": 5, "from_unit": "mm", "to_unit": "milimetres"}),
            turn_final("never reached"),
        ],
    ),
    Case(
        # The unit-conversion HABIT the reflection later fixes: the tool is
        # called correctly (clean trajectory) but the final answer ignores the
        # observation and uses confident mental math (200 instead of 196.85).
        "l7-c06", "specbook-b", "5 mm 是多少 mil？",
        ((196.85, "mil"),), ("196.85 mil",), min_tool_calls=1,
        script=[
            turn_tool("c1", "unit_convert", {"value": 5, "from_unit": "mm", "to_unit": "mil"}),
            turn_final("按经验 5 mm 约为 200 mil，这个换算很常用。"),
        ],
    ),
    Case(
        # The REWARD-HACKING case: fluent, well-formed, confident -- and wrong
        # (0.15 vs the tool's 0.1). A fluency-only judge scores it HIGH.
        "l7-c07", "specbook-b", "外层最小线宽是多少 mm？",
        ((0.1, "mm"),), ("0.1 mm",), min_tool_calls=1,
        script=[
            turn_tool("c1", "lookup_spec", {"item": "min_trace_width_outer_mm"}),
            turn_final("根据工艺规范，外层最小线宽为 0.15 mm。该结论基于长期生产经验，可信度高。"),
        ],
    ),
    Case(
        # Identical request twice -> the tools node refuses, runtime stops with
        # no_progress and no final answer ever exists.
        "l7-c08", "specbook-b", "4 mm 是多少 mil？",
        ((157.48, "mil"),), ("157.48 mil",), min_tool_calls=1,
        script=[
            turn_tool("c1", "unit_convert", {"value": 4, "from_unit": "mm", "to_unit": "mil"}),
            turn_tool("c2", "unit_convert", {"value": 4, "from_unit": "mm", "to_unit": "mil"}),
        ],
    ),
    # -- vendor-datasheet (供应商资料): the MINOR group watched for regression
    Case(
        "l7-c09", "vendor-datasheet", "2.54 mm 是多少 mil？",
        ((100.0, "mil"),), ("100.0 mil",), min_tool_calls=1,
        script=[
            turn_tool("c1", "unit_convert", {"value": 2.54, "from_unit": "mm", "to_unit": "mil"}),
            turn_final("2.54 mm 正好是 100.0 mil。"),
        ],
    ),
    Case(
        # Paraphrased correct answer ("59 mil" instead of "59.06 mil"): passes
        # the tolerance result evaluator, but a strict evidence-quoting judge
        # v2 fails it -> v2's false negative (calibration section).
        "l7-c10", "vendor-datasheet", "1.5 mm 是多少 mil？",
        ((59.06, "mil"),), ("59.06 mil",), min_tool_calls=1,
        script=[
            turn_tool("c1", "unit_convert", {"value": 1.5, "from_unit": "mm", "to_unit": "mil"}),
            turn_final("1.5 mm 约 59 mil 左右。"),
        ],
    ),
    Case(
        # Label/source conflict: the ground truth says 0.2 mm, the tool table
        # says 0.3 mm. The model faithfully quotes the tool -> result fails,
        # attribution cannot blame the model -> `unknown`, needs human review.
        "l7-c11", "vendor-datasheet", "供应商推荐的默认过孔直径是多少 mm？",
        ((0.2, "mm"),), ("0.2 mm",), min_tool_calls=1,
        script=[
            turn_tool("c1", "lookup_spec", {"item": "default_via_diameter_mm"}),
            turn_final("供应商推荐的默认过孔直径为 0.3 mm。"),
        ],
    ),
    # -- legacy-notes (历史笔记): the single-case group
    Case(
        "l7-c12", "legacy-notes", "内层最小线宽是多少 mm？",
        ((0.127, "mm"),), ("0.127 mm",), min_tool_calls=1,
        script=[
            turn_tool("c1", "lookup_spec", {"item": "min_trace_width_inner_mm"}),
            turn_final("内层最小线宽为 0.127 mm。"),
        ],
    ),
]

CASES_BY_ID = {case.case_id: case for case in DATASET}


# ---------------------------------------------------------------------------
# 2. Run harness: drive L2's loop graph, capture a RunRecord per case
# ---------------------------------------------------------------------------


@dataclass
class RunRecord:
    """One executed run: what the loop did, and how it ended.

    This is the local stand-in for telemetry: `steps`/`tool_calls` are counted
    directly from the final state and the execution journal, not inferred.
    """

    case: Case
    stop_reason: str
    steps: int
    tokens_used: int
    tool_calls: list[tuple[str, str]]  # executed (tool_name, arguments_json)
    final_answer: str
    messages: list[dict]  # final-state message history (for span-tree replay)
    variant: str = "baseline"


def run_case(case: Case, script: list[dict] | None = None, variant: str = "baseline") -> RunRecord:
    """Run ONE dataset case through L2's graph with a scripted model.

    The global tool journal is cleared around the run and copied out, so each
    RunRecord counts exactly its own executions.
    """
    clear_tool_journal()
    fake = FakeModel(script if script is not None else case.script)
    state = build_loop_graph(fake.respond).invoke(initial_state(case.question))
    executed = list(tool_journal)
    clear_tool_journal()
    return RunRecord(
        case=case,
        stop_reason=state["stop_reason"],
        steps=state["steps"],
        tokens_used=state["tokens_used"],
        tool_calls=executed,
        final_answer=state.get("final_answer") or "",
        messages=list(state["messages"]),
        variant=variant,
    )


def run_dataset(dataset: list[Case] = DATASET, variant: str = "baseline") -> list[RunRecord]:
    return [run_case(case, variant=variant) for case in dataset]


# ---------------------------------------------------------------------------
# 3. Evaluators: result vs trajectory (they catch DIFFERENT failures)
# ---------------------------------------------------------------------------

NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
UNIT_TOKENS = ("mm", "mil", "um", "cm", "inch")
TOLERANCE = 0.01  # relative tolerance on expected parameter values


def params_match(answer: str, expected_params: tuple[tuple[float, str], ...]) -> bool:
    """Rule-based RESULT check: every expected (value, unit) appears, in tolerance."""
    if not answer:
        return False
    numbers = [float(n) for n in NUM_RE.findall(answer)]
    for value, unit in expected_params:
        if unit not in answer:
            return False
        if not any(abs(n - value) / max(abs(value), 1e-9) <= TOLERANCE for n in numbers):
            return False
    return True


def result_score(record: RunRecord) -> float:
    """1.0 only when the run succeeded AND every expected param matches."""
    if record.stop_reason != "success":
        return 0.0
    return 1.0 if params_match(record.final_answer, record.case.expected_params) else 0.0


def semantic_tool_key(name: str, arguments: str) -> tuple:
    """Normalize a tool call so 3 and 3.0, key order etc. collapse to one key.

    L2's runtime only catches EXACT duplicates (identical JSON); this
    evaluator cares about SEMANTIC duplicates, so normalization lives here.
    """
    try:
        args = json.loads(arguments)
    except json.JSONDecodeError:
        return (name, arguments)
    normalized = {
        k: float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v
        for k, v in args.items()
    }
    return (name, tuple(sorted(normalized.items())))


def count_loops(tool_calls: list[tuple[str, str]]) -> int:
    """Executed calls whose semantic key was already executed -> loops."""
    seen: set[tuple] = set()
    loops = 0
    for name, args in tool_calls:
        key = semantic_tool_key(name, args)
        if key in seen:
            loops += 1
        seen.add(key)
    return loops


def trajectory_breakdown(record: RunRecord) -> dict[str, int]:
    """The three trajectory smells, each counted independently."""
    return {
        "loops": count_loops(record.tool_calls),
        "wasted_steps": max(0, len(record.tool_calls) - record.case.min_tool_calls),
        "wrong_termination": int(
            record.stop_reason in {"budget", "no_progress"} and record.case.solvable
        ),
    }


def trajectory_score_from_facts(breakdown: dict[str, int]) -> float:
    """Score a trajectory from its breakdown (kept pure for mlflow_demo reuse)."""
    score = (
        1.0
        - 0.2 * breakdown["loops"]
        - 0.05 * breakdown["wasted_steps"]
        - 0.5 * breakdown["wrong_termination"]
    )
    return round(max(0.0, score), 3)


def trajectory_score(record: RunRecord) -> float:
    return trajectory_score_from_facts(trajectory_breakdown(record))


# ---------------------------------------------------------------------------
# 4. Failure attribution: which layer broke, per run
# ---------------------------------------------------------------------------


def _tool_observation_values(record: RunRecord) -> list[float]:
    """Numeric values successfully returned by tools during the run."""
    values: list[float] = []
    for message in record.messages:
        if message.get("role") != "tool":
            continue
        try:
            envelope = json.loads(message["content"])
        except json.JSONDecodeError:
            continue
        value = envelope.get("value") if envelope.get("ok") else None
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def faithful_to_tool(record: RunRecord) -> bool:
    """True when the final answer's number MATCHES a tool-returned value.

    This is the tell for `unknown`: the model did exactly what the tool said
    and the result still disagrees with the label -> the label/source is
    suspect, not the model. Distinguishing this from model_error is why the
    attribution is not just "answer wrong -> model's fault".
    """
    if not record.final_answer:
        return False
    numbers = [float(n) for n in NUM_RE.findall(record.final_answer)]
    return any(
        abs(n - v) / max(abs(v), 1e-9) <= TOLERANCE
        for n in numbers
        for v in _tool_observation_values(record)
    )


def attribute_failure(record: RunRecord) -> str:
    """Classify one run: ok / tool_error / budget_stop / routing_error / model_error / unknown.

    Priority order mirrors where the EVIDENCE lives: the stop_reason first
    (the runtime knows), then the trajectory smells (routing), then the
    answer-vs-tool relation (model vs label).
    """
    if result_score(record) == 1.0 and trajectory_score(record) == 1.0:
        return "ok"
    if record.stop_reason == "tool_error":
        return "tool_error"
    if record.stop_reason == "budget":
        return "budget_stop"
    if record.stop_reason == "no_progress":
        return "routing_error"
    # stop_reason == "success" from here
    breakdown = trajectory_breakdown(record)
    if result_score(record) == 1.0:
        # correct result, dirty path: looped or wasted -> routing
        return "routing_error" if (breakdown["loops"] or breakdown["wasted_steps"]) else "ok"
    # wrong result: was the model faithful to its tools? then we cannot blame it
    return "unknown" if faithful_to_tool(record) else "model_error"


def attribution_table(records: list[RunRecord]) -> dict[str, list[str]]:
    table: dict[str, list[str]] = {}
    for record in records:
        table.setdefault(attribute_failure(record), []).append(record.case.case_id)
    return table


# ---------------------------------------------------------------------------
# 5. Three kinds of judgment: rules, recorded human annotations, LLM judge
# ---------------------------------------------------------------------------
# The judge interface is identical offline and live (see live_demo.py):
#   judge(case, answer) -> {"judge", "verdict", "score", "reason"}

JUDGE_PASS_THRESHOLD = 0.8


def make_offline_judge(version: str) -> Callable[[Case, str], dict]:
    """A scripted LLM-judge fake with the SAME interface as the live judge.

    v1 checks fluency and format only -- exactly the failure mode the
    reward-hacking counterexample exploits. v2 adds the grounding fix: every
    evidence substring must appear verbatim in the answer.
    """

    def judge(case: Case, answer: str) -> dict:
        fluent = (
            bool(answer)
            and bool(NUM_RE.search(answer))
            and any(unit in answer for unit in UNIT_TOKENS)
            and len(answer) >= 8
        )
        if version == "v1":
            return {
                "judge": "v1",
                "verdict": "pass" if fluent else "fail",
                "score": 0.9 if fluent else 0.1,
                "reason": "fluency + format only (no grounding check)",
            }
        quoted = bool(answer) and all(evidence in answer for evidence in case.evidences)
        passed = fluent and quoted
        missing = [e for e in case.evidences if e not in (answer or "")]
        return {
            "judge": "v2",
            "verdict": "pass" if passed else "fail",
            "score": 0.9 if passed else 0.1,
            "reason": (
                "fluency + format + evidence quoted verbatim"
                if passed
                else f"answer does not quote the evidence verbatim: {missing}"
            ),
        }

    return judge


# Recorded human judgment over a slice of the dataset (a fixture, not a live
# queue): case_id -> verdict + note. "unknown" verdicts are escalated, never
# counted as agreement in either direction.
HUMAN_ANNOTATIONS = {
    "l7-c04": {"verdict": "fail", "note": "跑到预算上限，没有给出任何答案"},
    "l7-c07": {"verdict": "fail", "note": "0.15 mm 与规范表 0.1 mm 不一致（人工核对表 3）"},
    "l7-c10": {"verdict": "pass", "note": "59 mil 在容差内，近似说法可以接受"},
    "l7-c11": {"verdict": "unknown", "note": "供应商资料写 0.2 mm，工具表返回 0.3 mm，标注待复核"},
}


def calibrate(judge: Callable[[Case, str], dict], records: list[RunRecord]) -> dict:
    """Judge-vs-ground-truth agreement: a 2x2 matrix plus the agreement rate.

    Ground truth here is the rule-based result verdict (score == 1.0); in a
    real setup it would be human-verified labels. A judge that agrees with
    truth on this dataset is CALIBRATED; one that scores hacks high is not.
    """
    matrix = {(j, t): 0 for j in ("pass", "fail") for t in ("pass", "fail")}
    for record in records:
        judged = judge(record.case, record.final_answer)
        truth = "pass" if result_score(record) == 1.0 else "fail"
        matrix[(judged["verdict"], truth)] += 1
    total = sum(matrix.values())
    agree = matrix[("pass", "pass")] + matrix[("fail", "fail")]
    return {"matrix": matrix, "agree": agree, "total": total, "rate": agree / total}


# ---------------------------------------------------------------------------
# 6. Resource budget: simulated token accounting and declared limits
# ---------------------------------------------------------------------------

PER_CASE_TOKEN_BUDGET = 250  # per single case run
TOTAL_TOKEN_BUDGET = 1400  # one full pass over the 12-case dataset (baseline: 1500)
SIM_TOKEN_PRICE_USD = 0.001  # per simulated token, fake pricing for the report


def case_cost_tokens(record: RunRecord) -> int:
    return record.tokens_used


def group_cost(records: list[RunRecord]) -> dict[str, dict[str, float]]:
    """Aggregate simulated tokens and cost per source group."""
    groups: dict[str, dict[str, float]] = {}
    for record in records:
        g = record.case.source_group
        groups.setdefault(g, {"cases": 0, "tokens": 0, "cost_usd": 0.0})
        groups[g]["cases"] += 1
        groups[g]["tokens"] += case_cost_tokens(record)
        groups[g]["cost_usd"] = round(groups[g]["tokens"] * SIM_TOKEN_PRICE_USD, 3)
    return groups


def budget_check(records: list[RunRecord]) -> dict:
    per_case = [
        r.case.case_id
        for r in records
        if case_cost_tokens(r) > PER_CASE_TOKEN_BUDGET
    ]
    total = sum(case_cost_tokens(r) for r in records)
    return {
        "per_case_violations": per_case,
        "total_tokens": total,
        "total_budget": TOTAL_TOKEN_BUDGET,
        "total_over_budget": total > TOTAL_TOKEN_BUDGET,
        "total_cost_usd": round(total * SIM_TOKEN_PRICE_USD, 3),
    }


# ---------------------------------------------------------------------------
# 7. Reflection (evaluator-optimizer): failure summary -> revised approach
# ---------------------------------------------------------------------------


def failure_summary(records: list[RunRecord]) -> dict:
    """Compress evaluation output into the reflection's input."""
    summary: dict[str, list[str]] = {}
    for record in records:
        cause = attribute_failure(record)
        summary.setdefault(cause, []).append(record.case.case_id)
    return summary


def reflect(summary: dict[str, list[str]]) -> str:
    """The scripted 'thinking' step: consume the failure summary, emit a plan.

    In a real evaluator-optimizer the plan would be generated; here it is
    fixed text encoding exactly three habits plus one honest side effect
    (an extra verification call on vendor-datasheet cases) that section 9
    watches for a regression.
    """
    return (
        "revised approach:\n"
        "  1) unit conversions ONLY via unit_convert; quote the tool's value\n"
        "     verbatim in the answer (fixes model_error cases c06/c07);\n"
        "  2) never repeat a semantically identical tool call (fixes c02);\n"
        "  3) plan finitely: convert -> lookup -> answer, no runaway loops\n"
        "     (fixes budget c04, tool_error c05, no_progress c08);\n"
        "  4) when the tool value contradicts the expected label, keep the\n"
        "     tool value and flag the case for review (c11 stays unknown);\n"
        f"  side effect: one extra verification call on vendor-datasheet cases\n"
        f"  (defensive habit -> watch the per-group cost table).\n"
        f"  input summary: { {k: v for k, v in sorted(summary.items())} }"
    )


# The scripted OUTCOME of applying the revised approach to each failing case.
# Cases not listed keep their baseline script (no regression by construction
# -- except c09, listed on purpose: the reflection adds a verification call
# that costs one wasted step in the minor group).
REVISED_SCRIPTS: dict[str, list[dict]] = {
    # habit 2: drop the semantically duplicate convert
    "l7-c02": [
        turn_tool("r1", "unit_convert", {"value": 3, "from_unit": "mm", "to_unit": "mil"}),
        turn_final("3 mm 约为 118.11 mil。"),
    ],
    # habit 3: the finite plan this solvable case needed all along
    "l7-c04": [
        turn_tool("r1", "unit_convert", {"value": 3, "from_unit": "mm", "to_unit": "mil"}),
        turn_tool("r2", "lookup_spec", {"item": "min_trace_width_inner_mm"}),
        turn_final("3 mm 约为 118.11 mil；内层最小线宽为 0.127 mm。"),
    ],
    # habit 1 (+ correct units): no more error envelopes
    "l7-c05": [
        turn_tool("r1", "unit_convert", {"value": 3, "from_unit": "mm", "to_unit": "mil"}),
        turn_final("3 mm 约为 118.11 mil。"),
    ],
    # habit 1: quote the tool value, no mental math
    "l7-c06": [
        turn_tool("r1", "unit_convert", {"value": 5, "from_unit": "mm", "to_unit": "mil"}),
        turn_final("5 mm 约为 196.85 mil。"),
    ],
    "l7-c07": [
        turn_tool("r1", "lookup_spec", {"item": "min_trace_width_outer_mm"}),
        turn_final("外层最小线宽为 0.1 mm。"),
    ],
    # habit 2: distinct calls, then answer
    "l7-c08": [
        turn_tool("r1", "unit_convert", {"value": 4, "from_unit": "mm", "to_unit": "mil"}),
        turn_final("4 mm 约为 157.48 mil。"),
    ],
    # THE SCRIPTED REGRESSION: defensive verification call added in the minor
    # group -- result stays correct, trajectory loses one step to waste.
    "l7-c09": [
        turn_tool("r1", "unit_convert", {"value": 2.54, "from_unit": "mm", "to_unit": "mil"}),
        turn_tool("r2", "unit_convert", {"value": 100.0, "from_unit": "mil", "to_unit": "mm"}),
        turn_final("2.54 mm 正好是 100.0 mil。"),
    ],
    # c11 intentionally NOT revised: the label/source conflict is not the
    # model's to fix; it must stay failing with cause `unknown` (报告未知).
}


def revise_script(case: Case) -> list[dict] | None:
    """The revised FakeModel script for a case (None = keep baseline)."""
    return REVISED_SCRIPTS.get(case.case_id)


def run_reflected(dataset: list[Case] = DATASET) -> list[RunRecord]:
    records = []
    for case in dataset:
        revised = revise_script(case)
        records.append(
            run_case(case, script=revised if revised is not None else case.script, variant="reflected")
        )
    return records


def combined_score(record: RunRecord) -> float:
    """One number per case for before/after tables: result-weighted."""
    return round(0.75 * result_score(record) + 0.25 * trajectory_score(record), 4)


def group_scores(records: list[RunRecord]) -> dict[str, float]:
    """Mean combined score per source group -- where regressions hide."""
    groups: dict[str, list[float]] = {}
    for record in records:
        groups.setdefault(record.case.source_group, []).append(combined_score(record))
    return {g: round(sum(v) / len(v), 4) for g, v in sorted(groups.items())}


def aggregate_score(records: list[RunRecord]) -> float:
    return round(sum(combined_score(r) for r in records) / len(records), 4)


# ---------------------------------------------------------------------------
# 8. Demos
# ---------------------------------------------------------------------------


def demo_run_trace_span_vocabulary() -> RunRecord:
    banner("DEMO 1 - run / trace / span / step：本地重放建立词汇")
    record = run_case(CASES_BY_ID["l7-c01"])
    print("  本课离线部分没有遥测后端；这里从 RunRecord 重放一棵 span 树建立词汇。")
    print("  真实 trace（服务器上的 span 树）在 mlflow_demo.py 里读取。\n")
    print(f"  run l7-c01  question={record.case.question!r}  stop={record.stop_reason}")
    step_no = 0
    for message in record.messages:
        if message.get("role") == "user":
            print(f"  └─ input {message['content']!r}")
        elif message.get("tool_calls"):
            step_no += 1
            for call in message["tool_calls"]:
                fn = call["function"]
                print(f"     ├─ step {step_no} span model -> tool {fn['name']}({fn['arguments']})")
        elif message.get("role") == "tool":
            print(f"     │    └─ observation {message['content']}")
        else:
            step_no += 1
            print(f"     └─ step {step_no} span model (final): {message['content']}")
    print(
        f"\n  词汇表：run=一次带记录的执行（对应一次 mlflow run）；trace=这一次执行\n"
        f"  的完整遥测树；span=树中一个节点（LangGraph 下一个节点一个 span）；\n"
        f"  step=模型轮/super-step。本例 steps={record.steps}，工具执行 {len(record.tool_calls)} 次。"
    )
    return record


def demo_result_vs_trajectory(records: list[RunRecord]) -> None:
    banner("DEMO 2 - 结果评价 vs 轨迹评价：各自漏掉什么")
    print(f"  {'case':<8} {'stop':<11} {'result':<7} {'traj':<6} loops/wasted/wrongTerm")
    for record in records:
        b = trajectory_breakdown(record)
        print(
            f"  {record.case.case_id:<8} {record.stop_reason:<11} "
            f"{result_score(record):<7.1f} {trajectory_score(record):<6.3f} "
            f"{b['loops']}/{b['wasted_steps']}/{b['wrong_termination']}"
        )
    c02 = next(r for r in records if r.case.case_id == "l7-c02")
    c07 = next(r for r in records if r.case.case_id == "l7-c07")
    print(f"\n  c02: result={result_score(c02):.1f} 但 trajectory={trajectory_score(c02)} "
          f"(语义重复调用 1 次) -> 只有轨迹评价抓得到")
    print(f"  c07: result={result_score(c07):.1f} 但 trajectory={trajectory_score(c07):.1f} "
          f"(路径干净、答案自信且错误) -> 只有结果评价抓得到")
    print("  **两种评价互补；只看一边都会把系统评价错。**")


def demo_group_split() -> None:
    banner("DEMO 3 - 数据分组与留出：同一来源不跨集合")
    split = split_by_group(DATASET)
    for name, groups in (("train", split["train"]), ("dev", split["dev"]), ("test", split["test"])):
        cases = [c.case_id for c in DATASET if c.source_group in groups]
        print(f"  {name:<6} groups={sorted(groups)} -> cases={cases}")
    leaked = split["train"] & split["dev"] | split["train"] & split["test"] | split["dev"] & split["test"]
    print(f"\n  组间交集（必须为空）: {leaked or '{}'}")
    print("  行级随机切分会让同一文档的近重复样本同时进 train 和 test，指标虚高；")
    print("  按 source_group 整组留出才衡量对新来源的泛化（种子固定，可复现）。")


def split_by_group(dataset: list[Case], ratios=(0.5, 0.25, 0.25), seed: int = 7) -> dict[str, set[str]]:
    """Group-aware split: SHUFFLE GROUPS, then allocate whole groups to sets.

    Rows of one source never cross sets. Deterministic via a fixed seed, so
    the same dataset always yields the same split.
    """
    groups = sorted({case.source_group for case in dataset})
    rng = random.Random(seed)
    rng.shuffle(groups)
    n_train = round(len(groups) * ratios[0])
    n_dev = round(len(groups) * ratios[1])
    return {
        "train": set(groups[:n_train]),
        "dev": set(groups[n_train : n_train + n_dev]),
        "test": set(groups[n_train + n_dev :]),
    }


def demo_attribution(records: list[RunRecord]) -> None:
    banner("DEMO 4 - 失败归因：把失败分到层")
    table = attribution_table(records)
    print(f"  {'cause':<14} cases")
    for cause in sorted(table):
        print(f"  {cause:<14} {table[cause]}")
    by_cause = {cause: set(ids) for cause, ids in table.items()}
    checks = {
        "budget_stop": "l7-c04",
        "tool_error": "l7-c05",
        "routing_error": {"l7-c02", "l7-c08"},
        "model_error": {"l7-c06", "l7-c07"},
        "unknown": {"l7-c11"},
    }
    print()
    for cause, expected in checks.items():
        got = by_cause.get(cause, set())
        expected_set = {expected} if isinstance(expected, str) else expected
        mark = "OK " if got == expected_set else "?? "
        print(f"  {mark}{cause:<14} {sorted(got)} (预期 {sorted(expected_set)})")
    print("  unknown = 模型忠实引用了工具值，但与标注冲突 -> 归因给\"模型\"是错的，")
    print("  应送人工复核（这正是归因层存在的意义）。")


def demo_judges(records: list[RunRecord]) -> None:
    banner("DEMO 5 - 规则 / 人工 / LLM-judge 三种评价与校准")
    print("  [规则] result+trajectory 已在 DEMO 2 打出；规则可复现但覆盖有限。\n")
    print("  [人工] 已记录的人工判定（fixture）join 到结果上:")
    for case_id, annotation in HUMAN_ANNOTATIONS.items():
        record = next(r for r in records if r.case.case_id == case_id)
        rule = "pass" if result_score(record) == 1.0 else "fail"
        print(
            f"    {case_id}: human={annotation['verdict']:<8} rule={rule:<5} "
            f"note={annotation['note']}"
        )
    print("    -> l7-c11 人工给了 unknown：与规则 fail 不计入一致率，走复核队列。\n")
    print("  [LLM-judge] 离线用脚本化 fake（接口与 live_demo 的真实 judge 相同）：")
    for version in ("v1", "v2"):
        calibration = calibrate(make_offline_judge(version), records)
        m = calibration["matrix"]
        print(
            f"    {version}: agreement={calibration['agree']}/{calibration['total']}"
            f" ({calibration['rate']:.0%})  matrix judge\\truth "
            f"pass/pass={m[('pass', 'pass')]} pass/fail={m[('pass', 'fail')]} "
            f"fail/pass={m[('fail', 'pass')]} fail/fail={m[('fail', 'fail')]}"
        )
    print("    v1 的 pass/fail 格子 = 高分但任务失败（下一节）；v2 的 fail/pass 格子 =")
    print("    过严导致的误杀（c10 近似说法没逐字引用证据）。校准就是数这些格子。")


def demo_reward_hacking(records: list[RunRecord]) -> None:
    banner("DEMO 6 - 关键反例：高分但任务失败（reward hacking）")
    case = CASES_BY_ID["l7-c07"]
    record = next(r for r in records if r.case.case_id == "l7-c07")
    print(f"  case {case.case_id}: {case.question}")
    print(f"  expected params: {case.expected_params}  evidence: {case.evidences}")
    print(f"  answer: {record.final_answer}\n")
    for version in ("v1", "v2"):
        judged = make_offline_judge(version)(case, record.final_answer)
        verdict = (
            "HIGH (>= 0.8) -- 被骗"
            if judged["score"] >= JUDGE_PASS_THRESHOLD
            else "LOW (< 0.8) -- 抓住"
        )
        print(f"  judge {version}: score={judged['score']} verdict={judged['verdict']} -> {verdict}")
        print(f"            reason: {judged['reason']}")
    print(f"  ground truth: result_score={result_score(record):.1f}（任务失败）")
    print("  **v1 只查流畅与格式，给流畅的错误答案打高分；v2 要求逐字引用证据，")
    print("  同一案例同一答案 v1>=阈值 且 v2<阈值。修复不是换模型，是换判据。**")


def demo_budget(records: list[RunRecord]) -> None:
    banner("DEMO 7 - 资源预算：按案例与按来源组的模拟成本")
    print(f"  {'case':<8} {'tokens':<7} {'limit':<6} status")
    for record in records:
        tokens = case_cost_tokens(record)
        status = "OVER" if tokens > PER_CASE_TOKEN_BUDGET else "ok"
        print(f"  {record.case.case_id:<8} {tokens:<7} {PER_CASE_TOKEN_BUDGET:<6} {status}")
    print("\n  per source group:")
    for group, stats in group_cost(records).items():
        print(
            f"    {group:<17} cases={stats['cases']} "
            f"tokens={stats['tokens']:<4} cost_usd={stats['cost_usd']}"
        )
    check = budget_check(records)
    print(
        f"\n  dataset total: {check['total_tokens']} tokens "
        f"(budget {check['total_budget']}, cost ~${check['total_cost_usd']}) -> "
        f"{'OVER BUDGET' if check['total_over_budget'] else 'within budget'}"
    )
    print(f"  per-case violations: {check['per_case_violations']}")
    print("  真实 usage 字段来自 provider 遥测，见 mlflow_demo / live_demo 的 tokens 指标。")


def demo_reflection(records: list[RunRecord]) -> list[RunRecord]:
    banner("DEMO 8 - 一次简单反思（evaluator-optimizer）前后对照")
    summary = failure_summary(records)
    print("  失败摘要（反思的输入）:")
    for cause in sorted(summary):
        print(f"    {cause:<14} {summary[cause]}")
    plan = reflect(summary)
    print(f"\n  反思输出:\n  {plan}\n")
    reflected = run_reflected()
    print(f"  {'case':<8} {'before(res/traj)':<17} {'after(res/traj)':<17} direction")
    for before, after in zip(records, reflected):
        b, a = (result_score(before), trajectory_score(before)), (result_score(after), trajectory_score(after))
        if a[0] > b[0] or a[1] > b[1]:
            direction = "improved"
        elif combined_score(after) < combined_score(before):
            direction = "REGRESSED"
        elif a == b:
            direction = "unchanged"
        else:
            direction = "changed"
        print(f"  {before.case.case_id:<8} {b[0]:.1f}/{b[1]:<12} {a[0]:.1f}/{a[1]:<12} {direction}")
    print(f"\n  aggregate: {aggregate_score(records)} -> {aggregate_score(reflected)}")
    after_budget = budget_check(reflected)
    print(
        f"  budget    : {budget_check(records)['total_tokens']} -> {after_budget['total_tokens']} tokens "
        f"(limit {TOTAL_TOKEN_BUDGET}, now {'within' if not after_budget['total_over_budget'] else 'STILL OVER'} budget)"
    )
    return reflected


def demo_regression_report(records: list[RunRecord], reflected: list[RunRecord]) -> None:
    banner("DEMO 9 - 退化检查：分组建模前后的同时对照")
    before, after = group_scores(records), group_scores(reflected)
    print(f"  {'group':<17} {'before':<8} {'after':<8} delta")
    for group in before:
        delta = round(after[group] - before[group], 4)
        flag = "  <-- REGRESSION" if delta < 0 else ""
        print(f"  {group:<17} {before[group]:<8} {after[group]:<8} {delta:+.4f}{flag}")
    print(f"  {'AGGREGATE':<17} {aggregate_score(records):<8} {aggregate_score(reflected):<8} "
          f"{round(aggregate_score(reflected) - aggregate_score(records), 4):+.4f}")
    print("\n  **聚合指标大涨的同时 vendor-datasheet 小幅退化（反思加的验证调用）。")
    print("  报告必须暴露它，而不是让平均分淹没它。**")
    still = [r.case.case_id for r in reflected if result_score(r) < 1.0]
    unknown = [cid for cid in still if attribute_failure(next(r for r in reflected if r.case.case_id == cid)) == "unknown"]
    print(f"\n  仍未通过: {still}")
    print(f"  其中归因 unknown: {unknown} -> 报告为未知（疑似标注/来源冲突，送人工复核），")
    print("  不臆造成\"已解释的失败\"。")


if __name__ == "__main__":
    baseline = run_dataset()
    demo_run_trace_span_vocabulary()
    demo_result_vs_trajectory(baseline)
    demo_group_split()
    demo_attribution(baseline)
    demo_judges(baseline)
    demo_reward_hacking(baseline)
    demo_budget(baseline)
    reflected = demo_reflection(baseline)
    demo_regression_report(baseline, reflected)
    banner("done - 继续跑 mlflow_demo.py 看同一数据集在服务器上的 run/trace", width=72)
