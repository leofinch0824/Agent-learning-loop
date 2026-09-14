"""L7 x MLflow -- the observability mainline: runs, traces, and evaluation.

Prerequisites (repo root):

    docker compose up -d      # MLflow server on http://localhost:5000

Run:

    poetry run python lessons/l7_observability_and_evaluation/mlflow_demo.py

The SAME 12-case dataset and the SAME L2 loop graph as main.py, but every
case becomes a real server-side run + linked trace, and the rule evaluators
run through ``mlflow.genai.evaluate``:

  0. Connectivity: the shared "agent-loop" experiment.
  1. Per-case runs: params (case_id, source_group, stop_reason), metrics
     (steps, tool_calls, tokens_sim, result_score, trajectory_score), one
     linked trace per run; read back with flush=True.
  2. FAILURE LOCALIZATION: for the known failing cases, fetch the linked
     trace and attribute the failure from the span tree ALONE (which node
     ran how often, which tool span repeated its arguments).
  3. Cost/budget read-back: aggregate the logged tokens per source group
     from search_runs and compare against the declared budget.
  4. mlflow.genai.evaluate: the rule evaluators of main.py wrapped as
     scorers over a real evaluation run.

Probed mlflow 3.16 facts encoded here (learned by probing, not from blogs):

1. Classic ``mlflow.evaluate`` is deprecated since 3.0 AND, with a plain
   python-function model against this server, it HANGS (probed: >100s,
   predict_fn never invoked) -> verdict 未实测/不可用 for this shape; we do
   NOT force a broken API and use ``mlflow.genai.evaluate`` instead.
2. ``mlflow.genai.make_genai_metric`` does NOT exist in 3.16. The surface
   is the ``mlflow.genai.scorer`` decorator plus
   ``mlflow.genai.evaluate(data, scorers, predict_fn=...)``.
3. genai.evaluate validates that the ``inputs`` dict keys equal the
   predict_fn parameter names, reuses the ACTIVE run (so our "l7-" run
   name prefix survives), auto-traces every predict_fn call, and logs each
   scorer as ``<name>/mean`` on the run.
4. L2's loop under the langchain autolog yields one span per node PER
   iteration (model / tools, plus the router functions); the budget-stop
   fingerprint is 7 model spans for MAX_STEPS=6 ending on a bare model
   span, and a semantic duplicate tool call is readable from the tools
   span's ``inputs`` snapshot.
5. ``genai.evaluate`` runs scorers and predict_fn on worker threads whose
   telemetry hooks lazily import ``mlflow.server.jobs.utils``; racing on
   the interpreter import lock DEADLOCKS (silent hang, stack-dumped while
   probing). Importing those modules on the main thread first removes the
   race -- see ``_warm_up_lazy_imports``.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import mlflow
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lib import (  # noqa: E402
    DEFAULT_EXPERIMENT,
    DEFAULT_TRACKING_URI,
    banner,
    enable_graph_tracing,
    latest_traces,
    print_span_tree,
    setup_mlflow,
)
from lessons.l7_observability_and_evaluation.main import (  # noqa: E402
    CASES_BY_ID,
    DATASET,
    PER_CASE_TOKEN_BUDGET,
    TOTAL_TOKEN_BUDGET,
    params_match,
    run_case,
    result_score,
    semantic_tool_key,
    trajectory_breakdown,
    trajectory_score_from_facts,
)

# genai.evaluate may call predict_fn from worker threads; our SUT touches a
# module-global execution journal (L2's tool_journal), so runs are serialized.
_RUN_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# section 0: connectivity
# ---------------------------------------------------------------------------


def section_0_connectivity() -> tuple[mlflow.MlflowClient, str]:
    banner("MLFLOW 0 - connectivity & the 'agent-loop' experiment")
    print(f"  tracking uri : {DEFAULT_TRACKING_URI}")
    client, experiment_id = setup_mlflow(DEFAULT_EXPERIMENT)
    experiment = client.get_experiment_by_name(DEFAULT_EXPERIMENT)
    print(f"  experiment   : {experiment.name} (id={experiment.experiment_id})")
    print(f"  stage        : {experiment.lifecycle_stage}")
    existing = client.search_runs(
        [experiment_id],
        filter_string="tags.`mlflow.runName` LIKE 'l7-%'",
        max_results=1000,
    )
    print(f"  existing l7- runs: {len(existing)} (this demo adds more; prefix = hygiene)")
    return client, experiment_id


# ---------------------------------------------------------------------------
# section 1: one run + one linked trace per dataset case
# ---------------------------------------------------------------------------


def evaluate_case(record) -> dict:
    """The two rule scores of main.py, as metrics dicts for the server."""
    return {
        "result_score": result_score(record),
        "trajectory_score": trajectory_score_from_facts(trajectory_breakdown(record)),
    }


def section_1_case_runs(client: mlflow.MlflowClient, experiment_id: str) -> dict[str, str]:
    banner("MLFLOW 1 - per-case runs: params + metrics + one linked trace each")
    enable_graph_tracing()
    run_ids: dict[str, str] = {}
    for case in DATASET:
        # The case runs OUTSIDE... no: inside the run on purpose -- the invoke
        # inside start_run() is what LINKS the trace to the run.
        with mlflow.start_run(run_name=f"l7-case-{case.case_id}") as run:
            record = run_case(case)
            mlflow.log_params(
                {
                    "case_id": case.case_id,
                    "source_group": case.source_group,
                    "stop_reason": record.stop_reason,
                    "question": case.question[:60],
                }
            )
            mlflow.log_metrics(
                {
                    "steps": record.steps,
                    "tool_calls": len(record.tool_calls),
                    "tokens_sim": record.tokens_used,
                    **evaluate_case(record),
                }
            )
            run_ids[case.case_id] = run.info.run_id
        print(
            f"  {case.case_id} [{case.source_group:<17}] stop={record.stop_reason:<11} "
            f"result={evaluate_case(record)['result_score']:.1f} -> run logged"
        )

    # Read-back uses flush=True: traces export asynchronously (probed fact).
    traces = latest_traces(client, experiment_id, n=1)
    print(f"\n  newest trace after flush: {traces[0].info.trace_id[:16]}...")
    print("  the loop case's span tree (c02 -- watch the two tools spans):")
    loop_trace = client.search_traces(
        locations=[experiment_id], run_id=run_ids["l7-c02"], flush=True
    )[0]
    print_span_tree(loop_trace)
    return run_ids


# ---------------------------------------------------------------------------
# section 2: failure localization from the span tree ALONE
# ---------------------------------------------------------------------------


def fingerprint_from_spans(trace) -> dict:
    """Facts readable from a trace's span tree without any local state.

    Shape facts (trustworthy per L1): node-span counts and ordering. The
    tools spans' `inputs` snapshot carries the requested tool_calls, so a
    SEMANTIC duplicate (3 vs 3.0) is detectable from telemetry alone.
    """
    node_spans = [
        s for s in trace.data.spans if s.name in {"model", "tools"}
    ]
    node_spans.sort(key=lambda s: s.start_time_ns)
    requested_keys: list[tuple] = []
    for span in node_spans:
        if span.name != "tools":
            continue
        messages = (span.inputs or {}).get("messages") or []
        for call in (messages[-1].get("tool_calls") or []) if messages else []:
            fn = call["function"]
            requested_keys.append(semantic_tool_key(fn["name"], fn["arguments"]))
    duplicates = len(requested_keys) - len(set(requested_keys))
    return {
        "model_spans": sum(1 for s in node_spans if s.name == "model"),
        "tools_spans": sum(1 for s in node_spans if s.name == "tools"),
        "ends_on_model": bool(node_spans) and node_spans[-1].name == "model",
        "duplicate_tool_args": duplicates,
    }


def attribute_from_fingerprint(fingerprint: dict) -> str:
    """Turn the span-tree fingerprint into a localization verdict."""
    if fingerprint["duplicate_tool_args"]:
        return "loop: a tool span repeats a semantically identical call"
    if fingerprint["ends_on_model"] and fingerprint["tools_spans"] == 6:
        return "budget stop: 7 model spans (MAX_STEPS+1), ends on a bare model span"
    return "clean shape: the tree alone cannot explain this failure"


def section_2_failure_localization(
    client: mlflow.MlflowClient, experiment_id: str, run_ids: dict[str, str]
) -> None:
    banner("MLFLOW 2 - failure localization: search runs, then read the linked trace")
    wanted = set(run_ids.values())
    runs = [
        r
        for r in client.search_runs(
            [experiment_id],
            filter_string="tags.`mlflow.runName` LIKE 'l7-case-%'",
            max_results=1000,
            order_by=["attributes.start_time DESC"],
        )
        if r.info.run_id in wanted
    ]
    failing = [r for r in runs if r.data.metrics.get("result_score", 1.0) < 1.0]
    print(f"  {len(runs)} case runs searched back, {len(failing)} with result_score < 1:")
    for run in failing:
        print(
            f"    {run.data.params['case_id']} ({run.data.params['source_group']}) "
            f"stop={run.data.params['stop_reason']} result={run.data.metrics['result_score']:.1f}"
        )
    for case_id in ("l7-c04", "l7-c02", "l7-c07"):
        run_id = run_ids[case_id]
        linked = client.search_traces(locations=[experiment_id], run_id=run_id, flush=True)
        trace = linked[0]
        fingerprint = fingerprint_from_spans(trace)
        print(f"\n  -- {case_id}: span tree of its linked trace --")
        print_span_tree(trace)
        print(f"     fingerprint: {fingerprint}")
        print(f"     verdict    : {attribute_from_fingerprint(fingerprint)}")
    print(
        "\n  NOTE the honest limit: c07's tree has the SAME shape as a passing\n"
        "  run's tree (model -> tools -> model). WHERE fails (which node) and\n"
        "  WHETHER the task failed are different questions -- the second needs\n"
        "  the result metric plus the span payloads, i.e. evaluation, not just\n"
        "  localization. Trace-first finds runaway/loop failures fast; it does\n"
        "  not replace scoring."
    )


# ---------------------------------------------------------------------------
# section 3: cost / budget read-back from server data
# ---------------------------------------------------------------------------


def group_cost_from_server(
    client: mlflow.MlflowClient, experiment_id: str, run_ids: dict[str, str]
) -> dict[str, dict[str, float]]:
    """Aggregate logged tokens_sim per source_group, read back from the server."""
    wanted = set(run_ids.values())
    runs = [
        r
        for r in client.search_runs(
            [experiment_id],
            filter_string="tags.`mlflow.runName` LIKE 'l7-case-%'",
            max_results=1000,
        )
        if r.info.run_id in wanted
    ]
    groups: dict[str, dict[str, float]] = {}
    for run in runs:
        group = run.data.params["source_group"]
        entry = groups.setdefault(group, {"cases": 0, "tokens": 0.0, "over_per_case": 0})
        entry["cases"] += 1
        entry["tokens"] += run.data.metrics.get("tokens_sim", 0.0)
        if run.data.metrics.get("tokens_sim", 0.0) > PER_CASE_TOKEN_BUDGET:
            entry["over_per_case"] += 1
    return dict(sorted(groups.items()))


def section_3_cost_readback(
    client: mlflow.MlflowClient, experiment_id: str, run_ids: dict[str, str]
) -> dict[str, dict[str, float]]:
    banner("MLFLOW 3 - cost & budget read-back (aggregated from server metrics)")
    groups = group_cost_from_server(client, experiment_id, run_ids)
    total = 0.0
    print(f"  {'group':<17} {'cases':<6} {'tokens':<7} over-limit-cases")
    for group, stats in groups.items():
        total += stats["tokens"]
        print(
            f"  {group:<17} {stats['cases']:<6} {stats['tokens']:<7.0f} "
            f"{stats['over_per_case']} (per-case limit {PER_CASE_TOKEN_BUDGET})"
        )
    verdict = "OVER" if total > TOTAL_TOKEN_BUDGET else "within"
    print(
        f"\n  total tokens (from server): {total:.0f} vs declared budget "
        f"{TOTAL_TOKEN_BUDGET} -> {verdict} budget"
    )
    print("  matches main.py's offline accounting: the same counters, logged.")
    return groups


# ---------------------------------------------------------------------------
# section 4: mlflow.genai.evaluate wrapping the rule evaluators
# ---------------------------------------------------------------------------


def make_eval_dataframe() -> pd.DataFrame:
    """inputs keys MUST equal the predict_fn parameter names (probed fact 3)."""
    return pd.DataFrame(
        [
            {
                "inputs": {"question": case.question, "case_id": case.case_id},
                "expectations": {
                    "expected_params": [list(p) for p in case.expected_params],
                    "solvable": case.solvable,
                },
            }
            for case in DATASET
        ]
    )


def make_predict_fn():
    """Re-run each case through the L2 graph; return answer + trajectory facts."""

    def predict_fn(question: str, case_id: str) -> dict:
        with _RUN_LOCK:
            record = run_case(CASES_BY_ID[case_id])
        breakdown = trajectory_breakdown(record)
        return {
            "answer": record.final_answer,
            "stop_reason": record.stop_reason,
            "tool_calls": len(record.tool_calls),
            "loops": breakdown["loops"],
            "wasted_steps": breakdown["wasted_steps"],
            "wrong_termination": breakdown["wrong_termination"],
        }

    return predict_fn


def build_scorers():
    """The rule evaluators of main.py, wrapped as genai scorers."""
    from mlflow.genai import scorer

    @scorer(name="l7_result_match")
    def result_match(expectations, outputs):
        # SAME rule as main.params_match, applied to the judge-side outputs.
        return 1.0 if params_match(outputs["answer"], tuple(map(tuple, expectations["expected_params"]))) else 0.0

    @scorer(name="l7_trajectory_quality")
    def trajectory_quality(outputs):
        # SAME formula as main.trajectory_score_from_facts.
        return trajectory_score_from_facts(
            {
                "loops": outputs["loops"],
                "wasted_steps": outputs["wasted_steps"],
                "wrong_termination": outputs["wrong_termination"],
            }
        )

    return [result_match, trajectory_quality]


def _warm_up_lazy_imports() -> None:
    """Import mlflow's lazily-loaded modules on the MAIN thread, up front.

    Probed deadlock on 3.16: genai.evaluate runs scorers and predict_fn on
    worker threads; a scorer's telemetry hook lazily imports
    ``mlflow.server.jobs.utils``, whose module body imports further
    modules, while predict workers hold/wait on the interpreter import
    lock -> classic import-lock deadlock. The process hangs silently (no
    exception; verified with faulthandler stack dumps). Warming the same
    imports on the main thread first removes the race entirely (probed:
    hang -> 3.7s). Best-effort by design: if these internals move in a
    future version we only lose the warm-up, never the section.
    """
    import importlib

    try:
        module = importlib.import_module("mlflow.server.jobs.utils")
        module._build_job_name_to_fn_fullname_map()
        importlib.import_module("mlflow.telemetry.events")
    except Exception as exc:  # noqa: BLE001 - warm-up is deliberately best-effort
        print(f"  (import warm-up skipped: {type(exc).__name__}: {exc})")


def section_4_genai_evaluate(client: mlflow.MlflowClient, experiment_id: str) -> tuple[str, dict]:
    banner("MLFLOW 4 - mlflow.genai.evaluate: rule evaluators as scorers (probed surface)")
    print("  probed on 3.16 before writing this section:")
    print("   * classic mlflow.evaluate: deprecated since 3.0; with a python-fn model")
    print("     against this server it HANGS before calling predict_fn (>100s probed)")
    print("     -> 未实测/不可用 for this shape; not forced.")
    print("   * mlflow.genai.make_genai_metric: does NOT exist. Use the")
    print("     mlflow.genai.scorer decorator + mlflow.genai.evaluate().")
    print("   * genai.evaluate(data, scorers, predict_fn): validates inputs keys ==")
    print("     predict_fn params, reuses the ACTIVE run, auto-traces predictions.")
    print("   * its worker threads can deadlock on lazy imports -> we warm them")
    print("     up on the main thread first (_warm_up_lazy_imports).\n")
    _warm_up_lazy_imports()

    import mlflow.genai as genai

    with mlflow.start_run(run_name="l7-genai-eval") as run:
        run_id = run.info.run_id
        result = genai.evaluate(
            data=make_eval_dataframe(),
            scorers=build_scorers(),
            predict_fn=make_predict_fn(),
        )
    metrics = {k: round(float(v), 4) for k, v in result.metrics.items()}
    print(f"  evaluation run: l7-genai-eval ({run_id[:8]}...)")
    print(f"  metrics: {metrics}")
    server_metrics = client.get_run(run_id).data.metrics
    print(f"  read back from server: {sorted(server_metrics)}")
    linked = client.search_traces(locations=[experiment_id], run_id=run_id, flush=True)
    print(f"  traces linked to the evaluation run: {len(linked)} (one per predict_fn call)")
    expected = {
        "l7_result_match/mean": 6 / 12,
        "l7_trajectory_quality/mean": 10.45 / 12,
    }
    print(f"  expected from main.py's offline pass: {expected}")
    print("  UI (evaluation view): "
          f"{DEFAULT_TRACKING_URI}/#/experiments/{experiment_id}/evaluation-runs?selectedRunUuid={run_id}")
    return run_id, metrics


if __name__ == "__main__":
    client, experiment_id = section_0_connectivity()
    run_ids = section_1_case_runs(client, experiment_id)
    section_2_failure_localization(client, experiment_id, run_ids)
    section_3_cost_readback(client, experiment_id, run_ids)
    section_4_genai_evaluate(client, experiment_id)
    banner("mlflow demo done - open the UI links above", width=72)
