"""L2 live run: the real model closes the tool loop, MLflow records it.

First progressive-observation live experiment of the curriculum: replace the
scripted FakeModel with a real chat-completions endpoint (same two tools,
same graph, same termination policy) and, if the local MLflow server is up,
wrap the run in mlflow.start_run with params/metrics and read the trace back.

Prerequisites (repo root):

    .env with OPENAI_API_KEY (+ optional OPENAI_BASE_URL, LIVE_MODEL_NAME)
    docker compose up -d      # optional: MLflow on http://localhost:5000

Run:

    poetry run python lessons/l2_agent_loop/live_demo.py

Without .env this prints the configuration summary and exits 0 (the offline
mechanism experiments in main.py never depend on any of this).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lib import (  # noqa: E402
    DEFAULT_EXPERIMENT,
    banner,
    describe_live_config,
    enable_graph_tracing,
    latest_traces,
    live_client,
    live_enabled,
    live_model_name,
    mlflow_reachable,
    print_span_tree,
    setup_mlflow,
)

from lessons.l2_agent_loop.main import (  # noqa: E402
    OPENAI_TOOLS,
    build_loop_graph,
    clear_tool_journal,
    initial_state,
    tool_journal,
)

# Budgets for the live loop: generous enough that a healthy run ends via the
# model's own final answer, tight enough that a runaway model still stops.
LIVE_MAX_STEPS = 8
LIVE_TOKEN_BUDGET = 20_000

TASKS = [
    "3 mm 等于多少 mil？请使用工具换算后回答。",
    "内层最小线宽和外层最小线宽分别是多少 mm？请查表后回答。",
]

SYSTEM_PROMPT = (
    "You are a PCB process parameter assistant. Use the provided tools when "
    "they help answer the question. When you have the final answer, reply "
    "with a normal message and NO tool_calls."
)

# USD per 1M tokens (prompt, completion) for cost ESTIMATION only. Rates for
# models not listed here are unknown -> tokens are logged, cost is not.
PRICE_PER_1M = {"gpt-4o-mini": (0.15, 0.60)}


def make_openai_respond(client, model: str, usage_sink: dict | None = None):
    """Adapt the OpenAI SDK to the same respond(messages) interface as FakeModel.

    Returns {"message": <assistant message dict>, "usage": {...}} where the
    message is already in the chat-completions wire format our hand-rolled
    tools node parses (tool_calls[].function.name / .arguments). The exact
    prompt/completion split from the SDK usage object is accumulated into
    `usage_sink` so the MLflow metrics need no estimation.
    """

    def respond(messages: list[dict]) -> dict:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, *messages],
            tools=OPENAI_TOOLS,
        )
        msg = resp.choices[0].message
        tool_calls = [tc.model_dump() for tc in (msg.tool_calls or [])]
        usage = getattr(resp, "usage", None)
        prompt = getattr(usage, "prompt_tokens", 0) or 0
        completion = getattr(usage, "completion_tokens", 0) or 0
        if usage_sink is not None:
            usage_sink["prompt"] = usage_sink.get("prompt", 0) + prompt
            usage_sink["completion"] = usage_sink.get("completion", 0) + completion
        return {
            "message": {
                "role": "assistant",
                "content": msg.content,
                "tool_calls": tool_calls or None,
            },
            "usage": {"prompt_tokens": prompt, "completion_tokens": completion},
        }

    return respond


def _print_run(task: str, state: dict, journal: list[tuple[str, str]]) -> None:
    print(f"  task          : {task}")
    print(f"  stop_reason   : {state['stop_reason']}")
    print(f"  steps         : {state['steps']}   tokens_used: {state['tokens_used']}")
    print(f"  tool executions ({len(journal)}):")
    for name, args in journal:
        print(f"    - {name}({args})")
    print(f"  final_answer  : {state['final_answer'][:200]}")
    print(f"  roles         : {[m['role'] for m in state['messages']]}")


def main() -> int:
    banner("L2 live - real model closes the tool loop")
    print(f"  {describe_live_config()}")
    if not live_enabled():
        print("  skip: configure .env (OPENAI_API_KEY ...) -> live 未验证。")
        print("  offline 机制实验不受影响: poetry run python lessons/l2_agent_loop/main.py")
        return 0

    client = live_client()
    model = live_model_name()
    tracking = mlflow_reachable()
    mlflow_client = None
    experiment_id = None
    if tracking:
        # Probed mlflow 3.16 facts encoded in lib: LangGraph tracing enters via
        # the langchain flavor (no mlflow.langgraph module); traces export
        # asynchronously so every read-back uses flush=True.
        mlflow_client, experiment_id = setup_mlflow(DEFAULT_EXPERIMENT)
        enable_graph_tracing()
        print(f"  mlflow        : {DEFAULT_EXPERIMENT} (id={experiment_id}) -- run + trace will be recorded")
    else:
        print("  mlflow        : server unreachable -> trace 未记录（服务器不可达）")

    import mlflow  # noqa: PLC0415 - only needed when a run actually happens

    for i, task in enumerate(TASKS):
        banner(f"LIVE {i + 1}/{len(TASKS)} - {task}")
        usage_sink: dict[str, int] = {}
        respond = make_openai_respond(client, model, usage_sink)
        graph = build_loop_graph(respond, max_steps=LIVE_MAX_STEPS, token_budget=LIVE_TOKEN_BUDGET)
        clear_tool_journal()
        if tracking:
            with mlflow.start_run(run_name=f"l2-live-{i + 1}"):
                state = graph.invoke(initial_state(task))
                journal = list(tool_journal)
                prompt_tokens = usage_sink.get("prompt", 0)
                completion_tokens = usage_sink.get("completion", 0)
                mlflow.log_params({"model": model, "task": task, "max_steps": LIVE_MAX_STEPS})
                metrics = {
                    "steps": state["steps"],
                    "tool_calls": len(journal),
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                }
                cost = _est_cost(model, prompt_tokens, completion_tokens)
                if cost is not None:
                    metrics["est_cost_usd"] = cost
                else:
                    print(f"  (no price table entry for {model!r} -> cost metric skipped)")
                mlflow.log_metrics(metrics)
        else:
            state = graph.invoke(initial_state(task))
            journal = list(tool_journal)
        _print_run(task, state, journal)

    if tracking and mlflow_client is not None:
        banner("MLflow - read the traces back")
        # flush=True: force the async exporter out before searching.
        traces = latest_traces(mlflow_client, experiment_id, n=len(TASKS) + 2, flush=True)
        print(f"  {len(traces)} trace(s) in experiment {DEFAULT_EXPERIMENT!r}:")
        for tr in traces:
            print_span_tree(tr)
        print("  NOTE: span tree SHAPE is trustworthy (model/tools nesting);")
        print("  sibling span TIME OVERLAP is not -- never assert on it.")
    return 0


def _est_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
    rates = PRICE_PER_1M.get(model)
    if rates is None:
        return None
    p, c = rates
    return round(prompt_tokens / 1e6 * p + completion_tokens / 1e6 * c, 6)


if __name__ == "__main__":
    raise SystemExit(main())
