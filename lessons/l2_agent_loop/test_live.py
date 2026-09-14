"""Live tests for L2: the REAL model must close the tool loop.

Skipped entirely when .env is not configured (live_enabled() False); the
MLflow assertions additionally require the local tracking server. A skip is
recorded in README 完成记录 as 未验证, never counted as passed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lib import (  # noqa: E402
    DEFAULT_EXPERIMENT,
    live_client,
    live_enabled,
    live_model_name,
    mlflow_reachable,
)
from lessons.l2_agent_loop.live_demo import make_openai_respond  # noqa: E402
from lessons.l2_agent_loop.main import (  # noqa: E402
    build_loop_graph,
    clear_tool_journal,
    initial_state,
    tool_journal,
)

# Probed once at import time: without .env every test below skips.
_LIVE = live_enabled()
_MLFLOW = mlflow_reachable()

pytestmark = pytest.mark.skipif(not _LIVE, reason="live model not configured (.env: OPENAI_API_KEY)")

TASK = "1.5 mm 等于多少 mil？请使用工具换算后回答。"


def _run_live(**kwargs):
    usage_sink: dict[str, int] = {}
    respond = make_openai_respond(live_client(), live_model_name(), usage_sink)
    graph = build_loop_graph(respond, **kwargs)
    clear_tool_journal()
    state = graph.invoke(initial_state(TASK))
    return state, list(tool_journal), usage_sink


def test_live_model_closes_tool_loop():
    """A real model executes >=1 tool and produces a final answer."""
    state, journal, _ = _run_live(max_steps=8, token_budget=20_000)
    assert len(journal) >= 1, "expected at least one real tool execution"
    assert state["stop_reason"] == "success"
    assert state["final_answer"].strip()


@pytest.mark.skipif(not _MLFLOW, reason="local MLflow server not running (docker compose up -d)")
def test_live_run_records_mlflow_trace():
    """With MLflow up, the live invocation leaves a trace and a run with metrics."""
    import mlflow  # noqa: PLC0415

    from lib import enable_graph_tracing, latest_traces, setup_mlflow  # noqa: PLC0415

    _, experiment_id = setup_mlflow(DEFAULT_EXPERIMENT)
    enable_graph_tracing()
    with mlflow.start_run(run_name="l2-test-live"):
        state, journal, usage = _run_live(max_steps=8, token_budget=20_000)
        mlflow.log_params({"model": live_model_name(), "task": TASK})
        mlflow.log_metrics(
            {
                "steps": state["steps"],
                "tool_calls": len(journal),
                "prompt_tokens": usage.get("prompt", 0),
                "completion_tokens": usage.get("completion", 0),
            }
        )
        run_id = mlflow.active_run().info.run_id

    client = mlflow.MlflowClient()
    # flush=True forces the async trace exporter out before searching.
    traces = latest_traces(client, experiment_id, n=5, flush=True)
    assert traces, "expected at least one trace after the live invocation"
    run = client.get_run(run_id)
    assert run.data.metrics["steps"] >= 1
    assert run.data.metrics["tool_calls"] >= 1
