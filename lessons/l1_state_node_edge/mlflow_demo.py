"""L1 x MLflow -- first contact with the tracking server.

Prerequisites (the repo root has the compose file):

    docker compose up -d      # MLflow server on http://localhost:5000

Run:

    poetry run python lessons/l1_state_node_edge/mlflow_demo.py

What this demo shows -- every section prints raw facts read back from the
server, never just what we *sent*:

  0. Connectivity: the "agent-loop" experiment is found (or created).
  1. Traces: with autolog on, every graph.invoke() becomes ONE trace; each
     node is a span. The span tree mirrors the super-step structure:
     style_check and security_check are SIBLINGS (one fan-out super-step),
     and the join barrier shows up as reduce starting only after both have
     ended. Which span-clock facts are trustworthy and which are not is the
     point of this section.
  2. Runs: one explicit mlflow.start_run() per review, logging params,
     metrics (including the super-step count) and a JSON artifact.
  3. Read-back: runs and traces are two views of one experiment, and a trace
     born inside a start_run() block is LINKED to that run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mlflow
from langgraph.checkpoint.memory import InMemorySaver

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lib import (  # noqa: E402
    DEFAULT_EXPERIMENT,
    DEFAULT_TRACKING_URI,
    banner,
    enable_graph_tracing,
    latest_traces,
    print_span_tree,
    setup_mlflow,
    spans_overlap,
)
from lessons.l1_state_node_edge.main import build_review_graph  # noqa: E402


def count_super_steps(graph_builder, payload: dict) -> int:
    """True super-step count, read from the checkpoint the runtime writes.

    Do NOT count `updates` chunks: in langgraph 1.2 they are emitted per
    completed TASK, so the fan-out step (style_check + security_check) yields
    two chunks even though it is one super-step. The checkpoint's
    `metadata.step` is the authoritative counter -- L3's mechanism borrowed
    here for a single number. app.py reviews take 4 steps
    (intake | checks | reduce | accept); .env reviews take 3 (the conditional
    edge sends reduce straight to END).
    """
    graph = graph_builder(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "count-super-steps"}}
    graph.invoke(payload, config)
    return graph.get_state(config).metadata["step"]


def section_connectivity() -> tuple[mlflow.MlflowClient, str]:
    banner("MLFLOW 0 - connectivity & the 'agent-loop' experiment")
    print(f"  tracking uri : {DEFAULT_TRACKING_URI}")
    client, experiment_id = setup_mlflow(DEFAULT_EXPERIMENT)
    experiment = client.get_experiment_by_name(DEFAULT_EXPERIMENT)
    print(f"  experiment   : {experiment.name} (id={experiment.experiment_id})")
    print(f"  stage        : {experiment.lifecycle_stage}")
    return client, experiment_id


def section_traces(client: mlflow.MlflowClient, experiment_id: str) -> None:
    banner("MLFLOW 1 - autolog: one trace per invoke, one span per node")
    enable_graph_tracing()
    graph = build_review_graph()

    graph.invoke({"filename": "app.py", "findings": [], "verdict": "pass"})
    traces = latest_traces(client, experiment_id, n=1)
    trace = traces[0]
    print(f"\n  trace_id={trace.info.trace_id}")
    print_span_tree(trace)

    print("\n  reading the fan-out from telemetry (what is reliable, what is not):")
    spans = {s.name: s for s in trace.data.spans}
    style, security, reduce_ = spans["style_check"], spans["security_check"], spans["reduce"]
    root_start = min(s.start_time_ns for s in trace.data.spans if s.parent_id is None)
    print(f"    style_check    [{_at(root_start, style)}]")
    print(f"    security_check [{_at(root_start, security)}]")
    print(f"    reduce         [{_at(root_start, reduce_)}]")
    print(f"    siblings (same parent)? {style.parent_id == security.parent_id}")
    print(f"    sibling time overlap?   {spans_overlap(style, security)}")
    print("    -> overlap is EVIDENCE of parallelism but not proof: span clocks tick")
    print("       when LangChain callbacks fire, so tiny nodes can serialise even")
    print("       while truly running in parallel (run twice and watch it flip).")
    both_checks_end = max(style.end_time_ns, security.end_time_ns)
    print(f"    join barrier?  reduce starts after both checks end: "
          f"{both_checks_end <= reduce_.start_time_ns}")
    print("    -> THIS one is reliable: reduce is scheduled by the join, so its span")
    print("       can never start before every upstream sibling has finished.")


def section_runs() -> list[str]:
    banner("MLFLOW 2 - explicit runs: params, metrics, artifacts")
    graph = build_review_graph()
    run_ids: list[str] = []
    for filename in ("app.py", ".env"):
        payload = {"filename": filename, "findings": [], "verdict": "pass"}
        # Counted OUTSIDE the run: it is a separate graph execution (with its
        # own checkpointer), and keeping it out of the run block means the run
        # ends up with exactly ONE linked trace -- the invoke() below.
        steps = count_super_steps(build_review_graph, payload)
        # autolog stays on, so this run also gets a linked trace (section 3).
        with mlflow.start_run(run_name=f"review-{filename}") as run:
            result = graph.invoke(payload)
            mlflow.log_params({"filename": filename, "nodes": "intake|checks|reduce|accept"})
            mlflow.log_metrics(
                {
                    "super_steps": steps,
                    "findings_count": len(result["findings"]),
                    "rejected": 1 if result["verdict"] == "reject" else 0,
                }
            )
            # Artifact upload goes through the server's proxied artifact endpoint.
            mlflow.log_dict({"filename": filename, **result}, "findings.json")
            run_ids.append(run.info.run_id)
        print(f"  review-{filename}: verdict={result['verdict']} super_steps={steps} -> run logged")
    return run_ids


def section_read_back(client: mlflow.MlflowClient, experiment_id: str, run_ids: list[str]) -> None:
    banner("MLFLOW 3 - read-back: runs & traces are two views of one experiment")
    wanted = set(run_ids)
    recent = client.search_runs(
        [experiment_id], max_results=10, order_by=["attributes.start_time DESC"]
    )
    runs = [r for r in recent if r.info.run_id in wanted]
    print("  runs (what we ran, and how it went):")
    for run in runs:
        name = run.data.tags.get("mlflow.runName", run.info.run_id)
        print(f"    {name:<14} params={run.data.params} metrics={run.data.metrics}")

    linked = client.search_traces(locations=[experiment_id], run_id=run_ids[-1], flush=True)
    print(f"\n  traces linked to run {run_ids[-1][:8]}...: {len(linked)}")
    if linked:
        print("    -> the trace born inside start_run() carries the run's id; open the run in")
        print("       the UI and its trace hangs off it. Open traces standalone and it is")
        print("       listed under the experiment's Traces tab instead.")
    print(f"\n  UI: {DEFAULT_TRACKING_URI}/#/experiments/{experiment_id}")
    print(f"      {DEFAULT_TRACKING_URI}/#/experiments/{experiment_id}/traces")


def _at(origin_ns: int, span) -> str:
    """Span time range as offsets from the trace's root span, in milliseconds."""
    start = (span.start_time_ns - origin_ns) / 1_000_000
    end = (span.end_time_ns - origin_ns) / 1_000_000
    return f"{start:.2f}ms..{end:.2f}ms"


if __name__ == "__main__":
    client, experiment_id = section_connectivity()
    section_traces(client, experiment_id)
    run_ids = section_runs()
    section_read_back(client, experiment_id, run_ids)
    banner("mlflow demo done - open the UI link above", width=72)
