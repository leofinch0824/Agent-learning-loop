"""Tests for the L1 x MLflow integration.

Unlike test_main.py, these tests talk to the REAL local tracking server
(``docker compose up -d`` in the repo root). They skip automatically when
the server is unreachable, so ``pytest lessons/`` still passes on a machine
without the container running.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mlflow
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lib import (  # noqa: E402
    DEFAULT_EXPERIMENT,
    enable_graph_tracing,
    latest_traces,
    mlflow_reachable,
    setup_mlflow,
)
from lessons.l1_state_node_edge.main import build_review_graph  # noqa: E402
from lessons.l1_state_node_edge.mlflow_demo import count_super_steps  # noqa: E402

# Probed once at import time and reused by every test below (each marker
# condition would otherwise re-probe, which is slow against a dead port).
_SERVER_UP = mlflow_reachable()
pytestmark = pytest.mark.skipif(
    not _SERVER_UP, reason="local MLflow server not running (docker compose up -d)"
)


@pytest.fixture(scope="module")
def workspace():
    """One experiment workspace per module: (client, experiment_id)."""
    return setup_mlflow(DEFAULT_EXPERIMENT)


@pytest.fixture(scope="module")
def tracing():
    """Module-scoped so autolog patching happens once for all tests."""
    enable_graph_tracing()
    return True


def test_experiment_agent_loop_exists(workspace):
    """setup_mlflow() must find-or-create the agent-loop experiment."""
    client, experiment_id = workspace
    experiment = client.get_experiment_by_name(DEFAULT_EXPERIMENT)
    assert experiment is not None
    assert experiment.experiment_id == experiment_id
    assert experiment.lifecycle_stage == "active"


def test_invoke_creates_trace_with_node_spans(workspace, tracing):
    """One graph.invoke() == one trace; every node == one span under a root span."""
    client, experiment_id = workspace
    graph = build_review_graph()
    graph.invoke({"filename": "app.py", "findings": [], "verdict": "pass"})

    trace = latest_traces(client, experiment_id, n=1)[0]
    names = {span.name for span in trace.data.spans}
    # Root span (the graph invocation itself) + one span per node that ran.
    assert "LangGraph" in names
    assert {"intake", "style_check", "security_check", "reduce", "accept"} <= names
    roots = [s for s in trace.data.spans if s.parent_id is None]
    assert len(roots) == 1 and roots[0].name == "LangGraph"


def test_fanout_shape_and_join_barrier_in_span_tree(workspace, tracing):
    """Lesson 1's core facts, asserted from server-side telemetry.

    Reliable: fan-out nodes share a parent span (siblings), and the join
    node's span starts only after BOTH checks ended (the join barrier).
    NOT reliable: sibling time overlap -- span clocks tick on LangChain
    callbacks, so tiny parallel nodes can serialise (see spans_overlap's
    docstring); asserting overlap would make this test flaky.
    """
    client, experiment_id = workspace
    graph = build_review_graph()
    graph.invoke({"filename": "app.py", "findings": [], "verdict": "pass"})

    trace = latest_traces(client, experiment_id, n=1)[0]
    spans = {s.name: s for s in trace.data.spans}
    style, security, reduce_ = spans["style_check"], spans["security_check"], spans["reduce"]

    assert style.parent_id == security.parent_id, "fan-out nodes must share a parent span"
    both_checks_end = max(style.end_time_ns, security.end_time_ns)
    assert both_checks_end <= reduce_.start_time_ns, (
        "join barrier violated: reduce started before both checks finished"
    )


def test_super_step_count_matches_conditional_routing(workspace):
    """The super_steps metric reflects conditional-edge routing.

    app.py -> verdict=pass  -> accept runs -> 4 steps.
    .env   -> verdict=reject -> conditional edge sends it to END -> 3 steps.
    """
    clean = {"filename": "app.py", "findings": [], "verdict": "pass"}
    dirty = {"filename": ".env", "findings": [], "verdict": "pass"}
    assert count_super_steps(build_review_graph, clean) == 4
    assert count_super_steps(build_review_graph, dirty) == 3


def test_run_logs_params_metrics_and_artifact(workspace, tracing):
    """An explicit run keeps params/metrics/artifact AND a linked trace."""
    client, experiment_id = workspace
    graph = build_review_graph()
    payload = {"filename": ".env", "findings": [], "verdict": "pass"}

    # Counted outside the run so the run keeps exactly one linked trace.
    steps = count_super_steps(build_review_graph, payload)
    with mlflow.start_run(run_name="test-review-.env") as run:
        run_id = run.info.run_id
        result = graph.invoke(payload)
        mlflow.log_params({"filename": ".env"})
        mlflow.log_metrics({"super_steps": steps, "rejected": 1})
        mlflow.log_dict({"filename": ".env", **result}, "findings.json")

    data = client.get_run(run_id).data
    assert data.params["filename"] == ".env"
    assert data.metrics["super_steps"] == 3
    assert data.metrics["rejected"] == 1
    # Artifact upload goes through the server's proxied artifact endpoint.
    artifacts = [a.path for a in client.list_artifacts(run_id)]
    assert "findings.json" in artifacts

    # The invoke() above happened inside the run -> its trace is linked to it.
    linked = client.search_traces(locations=[experiment_id], run_id=run_id, flush=True)
    assert len(linked) == 1
    assert {"intake", "security_check", "reduce"} <= {s.name for s in linked[0].data.spans}
