"""Tests for the L7 x MLflow integration.

Unlike test_main.py, these tests talk to the REAL local tracking server
(``docker compose up -d`` in the repo root). They skip automatically when
the server is unreachable, so ``pytest lessons/`` still passes on a machine
without the container running -- the skip pattern is the documented way to
behave when the server is down, not a way to avoid the assertions.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mlflow
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lib import (  # noqa: E402
    DEFAULT_EXPERIMENT,
    mlflow_reachable,
    setup_mlflow,
)
from lessons.l7_observability_and_evaluation.mlflow_demo import (  # noqa: E402
    attribute_from_fingerprint,
    fingerprint_from_spans,
    group_cost_from_server,
    section_1_case_runs,
    section_4_genai_evaluate,
)

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
def case_runs(workspace):
    """Log the 12 per-case runs ONCE (real runs, real traces), reuse below."""
    client, experiment_id = workspace
    return section_1_case_runs(client, experiment_id)


def test_experiment_agent_loop_exists(workspace):
    """setup_mlflow() must find-or-create the agent-loop experiment."""
    client, experiment_id = workspace
    experiment = client.get_experiment_by_name(DEFAULT_EXPERIMENT)
    assert experiment is not None
    assert experiment.experiment_id == experiment_id
    assert experiment.lifecycle_stage == "active"


def test_case_runs_have_expected_params_and_metrics(workspace, case_runs):
    """Per-case runs carry case_id/source_group params and the rule metrics."""
    client, _ = workspace
    for case_id, expected in (
        ("l7-c01", {"steps": 2, "tool_calls": 1, "tokens_sim": 100, "result_score": 1.0}),
        ("l7-c04", {"steps": 6, "tool_calls": 6, "tokens_sim": 300, "result_score": 0.0}),
    ):
        data = client.get_run(case_runs[case_id]).data
        assert data.params["case_id"] == case_id
        assert data.params["source_group"] in {
            "specbook-a", "specbook-b", "vendor-datasheet", "legacy-notes"
        }
        for metric, value in expected.items():
            assert data.metrics[metric] == value, (case_id, metric)


def test_stop_reason_param_matches_offline_attribution(workspace, case_runs):
    """The logged stop_reason param is the same termination the offline pass saw."""
    client, _ = workspace
    expected = {
        "l7-c01": "success", "l7-c02": "success", "l7-c03": "success",
        "l7-c04": "budget", "l7-c05": "tool_error", "l7-c06": "success",
        "l7-c07": "success", "l7-c08": "no_progress", "l7-c09": "success",
        "l7-c10": "success", "l7-c11": "success", "l7-c12": "success",
    }
    for case_id, stop in expected.items():
        assert client.get_run(case_runs[case_id]).data.params["stop_reason"] == stop


def test_every_case_run_has_a_linked_trace(workspace, case_runs):
    """Each per-case run owns >=1 trace, and the trace is the L2 loop graph."""
    client, experiment_id = workspace
    for case_id, run_id in case_runs.items():
        linked = client.search_traces(locations=[experiment_id], run_id=run_id, flush=True)
        assert len(linked) >= 1, case_id
        names = {span.name for span in linked[0].data.spans}
        assert {"LangGraph", "model", "tools"} <= names, case_id


def test_failure_localization_reads_fingerprint_from_span_tree(workspace, case_runs):
    """The known failing cases are attributable from the span tree ALONE."""
    client, experiment_id = workspace
    c04 = client.search_traces(
        locations=[experiment_id], run_id=case_runs["l7-c04"], flush=True
    )[0]
    fingerprint = fingerprint_from_spans(c04)
    assert fingerprint["model_spans"] == 7  # MAX_STEPS=6 + the gated turn
    assert fingerprint["tools_spans"] == 6
    assert fingerprint["ends_on_model"] is True
    assert fingerprint["duplicate_tool_args"] == 0
    assert "budget" in attribute_from_fingerprint(fingerprint)

    c02 = client.search_traces(
        locations=[experiment_id], run_id=case_runs["l7-c02"], flush=True
    )[0]
    fingerprint = fingerprint_from_spans(c02)
    assert fingerprint["duplicate_tool_args"] == 1  # 3 vs 3.0, from span inputs
    assert "loop" in attribute_from_fingerprint(fingerprint)

    # c07 is the honest limit: clean shape, localization alone cannot explain.
    c07 = client.search_traces(
        locations=[experiment_id], run_id=case_runs["l7-c07"], flush=True
    )[0]
    fingerprint = fingerprint_from_spans(c07)
    assert fingerprint == {
        "model_spans": 2, "tools_spans": 1, "ends_on_model": True, "duplicate_tool_args": 0
    }
    assert "cannot explain" in attribute_from_fingerprint(fingerprint)


def test_group_cost_aggregation_from_server(workspace, case_runs):
    """tokens_sim aggregated per source group on the server matches offline."""
    client, experiment_id = workspace
    groups = group_cost_from_server(client, experiment_id, case_runs)
    assert {g: stats["tokens"] for g, stats in groups.items()} == {
        "specbook-a": 650,
        "specbook-b": 450,
        "vendor-datasheet": 300,
        "legacy-notes": 100,
    }
    over = {g: stats["over_per_case"] for g, stats in groups.items()}
    assert over["specbook-a"] == 1 and sum(over.values()) == 1  # only c04


def test_genai_evaluate_logs_rule_metrics(workspace):
    """genai.evaluate wraps the rule evaluators and logs mean scores on a run."""
    client, _ = workspace
    run_id, metrics = section_4_genai_evaluate(client, workspace[1])
    server_metrics = client.get_run(run_id).data.metrics
    assert server_metrics["l7_result_match/mean"] == pytest.approx(6 / 12)
    assert server_metrics["l7_trajectory_quality/mean"] == pytest.approx(10.45 / 12)
    assert metrics["l7_result_match/mean"] == pytest.approx(6 / 12)
    # the run name prefix keeps our runs identifiable in the shared experiment
    assert client.get_run(run_id).data.tags["mlflow.runName"] == "l7-genai-eval"
