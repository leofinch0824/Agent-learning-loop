"""MLflow helpers shared by all lessons.

The curriculum assumes a local MLflow server. ``docker-compose.yml`` in the
repo root brings one up on http://localhost:5000:

    docker compose up -d

Every lesson records two kinds of things into MLflow:

- **runs** -- explicit ``mlflow.start_run()`` blocks carrying params, metrics
  and artifacts ("what did we run, and how did it go").
- **traces** -- one per graph invocation via ``mlflow.langchain.autolog()``.
  Each node becomes a span, so the span tree mirrors the super-step structure
  ("what actually happened inside one invocation").

Three MLflow 3.16 facts this module encodes (learned by probing, not by docs):

1. LangGraph support lives in the *langchain* flavor: there is no
   ``mlflow.langgraph`` module, ``mlflow.langchain.autolog()`` attaches an
   MLflow tracer through the LangChain callback manager that LangGraph uses.
2. Traces are exported **asynchronously** by default
   (``MLFLOW_ENABLE_ASYNC_TRACE_LOGGING=True``). A ``search_traces`` right
   after ``invoke()`` may see nothing yet. Every read helper here therefore
   passes ``flush=True``, which forces pending writes out first.
3. ``search_traces(experiment_ids=...)`` is deprecated in favour of
   ``locations=...``, and the only sortable timestamp key is ``timestamp``.
"""

from __future__ import annotations

import os
import urllib.request
from typing import Any

import mlflow
from mlflow.entities import Trace
from mlflow.langchain import autolog as _langchain_autolog

DEFAULT_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
DEFAULT_EXPERIMENT = "agent-loop"


def mlflow_reachable(tracking_uri: str | None = None, timeout: float = 2.0) -> bool:
    """True if the tracking server answers ``GET /health``.

    Deliberately a plain HTTP probe instead of an MlflowClient call: the
    client's REST layer retries with exponential backoff, so probing a dead
    port through it can hang for minutes. Used by tests to skip cleanly when
    the docker container is not running.
    """
    url = (tracking_uri or DEFAULT_TRACKING_URI).rstrip("/") + "/health"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status == 200
    except Exception:  # noqa: BLE001 - any transport error means "not reachable"
        return False


def setup_mlflow(
    experiment_name: str = DEFAULT_EXPERIMENT,
    tracking_uri: str | None = None,
) -> tuple[mlflow.MlflowClient, str]:
    """Point MLflow at the local server and make sure the experiment exists.

    Returns ``(client, experiment_id)``. Creates the experiment if it does not
    exist yet, so the first run of any lesson bootstraps its own workspace.
    """
    mlflow.set_tracking_uri(tracking_uri or DEFAULT_TRACKING_URI)
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(experiment_name)
        print(f"  created experiment {experiment_name!r} (id={experiment_id})")
    else:
        experiment_id = experiment.experiment_id
    # Sets the default destination for both runs and traces.
    mlflow.set_experiment(experiment_name)
    return client, experiment_id


def enable_graph_tracing() -> None:
    """One trace per graph.invoke(), one span per node (see module docstring)."""
    _langchain_autolog(log_traces=True)


def latest_traces(
    client: mlflow.MlflowClient,
    experiment_id: str,
    n: int = 5,
    *,
    flush: bool = True,
) -> list[Trace]:
    """Newest-first traces of an experiment, forcing async writes out first."""
    return list(
        client.search_traces(
            locations=[experiment_id],
            max_results=n,
            order_by=["timestamp DESC"],
            flush=flush,
        )
    )


def _span_latency_ms(span: Any) -> int:
    return max(0, (span.end_time_ns - span.start_time_ns) // 1_000_000)


def print_span_tree(trace: Trace) -> None:
    """Print a trace's span tree by parent_id, indented like a call stack.

    For the L1 review graph the tree reads:

        LangGraph  (root span = one graph.invoke)
          intake
          security_check   } siblings -> same super-step, they ran in
          style_check      } parallel and cannot see each other's writes
          reduce
          accept

    Sibling order within a super-step is scheduler-dependent; what is
    meaningful is the *shape* (who is whose child), not the order.
    """
    spans = trace.data.spans
    roots = [s for s in spans if s.parent_id is None]
    children: dict[str | None, list[Any]] = {}
    for span in spans:
        children.setdefault(span.parent_id, []).append(span)

    def render(span: Any, depth: int) -> list[str]:
        status = getattr(span.status.status_code, "value", span.status.status_code)
        lines = ["  " * depth + f"- {span.name} [{status}] {_span_latency_ms(span)}ms"]
        for child in children.get(span.span_id, []):
            lines.extend(render(child, depth + 1))
        return lines

    for root in roots:
        print("\n".join(render(root, 0)))


def spans_overlap(a: Any, b: Any) -> bool:
    """True if two spans' [start, end) time ranges intersect.

    CAUTION: span timestamps are taken when LangChain callback events fire,
    not when node code starts executing. For microsecond-scale teaching nodes,
    truly parallel siblings can still yield NON-overlapping spans (callback
    processing serialises them), so overlap is evidence of parallelism, never
    proof of its absence. The reliable same-super-step evidence is shape
    (shared parent) plus ordering (a join node starts only after every
    upstream sibling has ended).
    """
    return a.start_time_ns < b.end_time_ns and b.start_time_ns < a.end_time_ns
