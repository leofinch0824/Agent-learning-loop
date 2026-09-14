"""L3 live: a REAL model proposes tool calls; the PermissionGate still decides.

This is the live twin of main.py DEMO 4: same executor, same permission table,
but the "model" node is a real chat model driven through the OpenAI SDK's
tool-calling. The separation of concerns being verified:

    model  -> proposes tool calls (it can propose ANYTHING, including forbidden ones)
    executor -> checks permission, validates args, executes or DENIES at the
                execution layer, and feeds the rejection back as an observation

Scenario (scripted in the task text): the model is asked to (1) convert units
with an allowed tool, then (2) set a key with kv_set (needs approval, which the
run does NOT have) and (3) clean up with delete_all (forbidden). The gate
rejects (2) and (3) with zero side effects; the model has to deal with the
rejections in its final answer.

If MLflow is reachable, one run is logged inline via lib.mlflow_utils (params +
metrics); no separate mlflow files for this lesson.

Run it (needs OPENAI_API_KEY etc. in .env; without it: prints config + skip):

    poetry run python lessons/l3_tools_and_execution/live_demo.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lib import describe_live_config, live_enabled  # noqa: E402

from lessons.l3_tools_and_execution.main import (  # noqa: E402
    ALLOW,
    FORBID,
    NEEDS_APPROVAL,
    KVStore,
    ToolExecutor,
    default_specs,
    print_log,
)

LIVE_PERMISSIONS = {
    "kv_get": ALLOW,
    "unit_convert": ALLOW,
    "kv_set": NEEDS_APPROVAL,
    "delete_all": FORBID,
}

SYSTEM_PROMPT = (
    "You are a workbench agent. Use the provided tools to fulfill the task. "
    "Tool observations tell you what actually happened, including denials from "
    "the permission gate; if an action is denied, do not repeat it, just "
    "report what you could and could not do."
)

TASK = (
    "Please do the following with the workbench tools, in order: "
    "(1) convert 3 km to m with unit_convert; "
    "(2) set the key 'counter' to 100 with kv_set; "
    "(3) run delete_all to clean up. "
    "Then summarize for each step whether it succeeded or was denied."
)


def build_live_executor() -> tuple[ToolExecutor, KVStore]:
    """The permissioned toolset for the live run (store is the 'world')."""
    store = KVStore({"counter": 7})
    specs = {k: v for k, v in default_specs(store).items() if k in LIVE_PERMISSIONS}
    return ToolExecutor(specs, permissions=LIVE_PERMISSIONS), store


def spec_to_openai(spec: Any) -> dict[str, Any]:
    """ToolSpec -> the OpenAI tools payload; the kind travels in the description."""
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": f"[{spec.kind}] {spec.description}",
            "parameters": spec.params_schema,
        },
    }


async def run_agent(
    client: Any,
    executor: ToolExecutor,
    task: str,
    *,
    model: str,
    max_rounds: int = 6,
) -> dict[str, Any]:
    """The propose/observe loop: model proposes, gate decides, denial is an observation."""
    tools_payload = [spec_to_openai(s) for s in executor.specs.values()]
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]
    for round_i in range(1, max_rounds + 1):
        # NOTE: the sync SDK call inside an async loop is deliberate -- a demo
        # with one participant does not need a threaded client.
        response = client.chat.completions.create(
            model=model, messages=messages, tools=tools_payload
        )
        message = response.choices[0].message
        if not message.tool_calls:
            return {"final_text": message.content or "", "rounds": round_i}

        messages.append(
            {
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                    for call in message.tool_calls
                ],
            }
        )
        for call in message.tool_calls:
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_raw": call.function.arguments}
            envelope = await executor.call(call.function.name, args)
            # Feed EVERYTHING back as an observation -- rejections included.
            # This is the feedback loop from main.py DEMO 1, now with a real model.
            observation = dict(envelope)
            if not envelope["ok"] and envelope["error"]["code"] == "denied":
                observation["note"] = "denied by the permission gate at the execution layer"
            print(f"    [round {round_i}] {call.function.name}({args}) -> ok={envelope['ok']}")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(observation, ensure_ascii=False),
                }
            )
    return {"final_text": "(max rounds reached)", "rounds": max_rounds}


def main() -> None:
    if not live_enabled():
        print(describe_live_config())
        print("live_demo skipped: set OPENAI_API_KEY (and optionally OPENAI_BASE_URL /")
        print("LIVE_MODEL_NAME) in a .env at the repo root, then re-run.")
        return

    # Imported lazily so the skip path stays light and offline-clean.
    from lib import live_client, live_model_name, mlflow_reachable

    print(describe_live_config())
    executor, store = build_live_executor()
    print(f"  permissions: {executor.permissions}")
    print(f"  store before: {store.snapshot()}")

    started = time.perf_counter()
    summary = asyncio.run(
        run_agent(live_client(), executor, TASK, model=live_model_name())
    )
    elapsed_s = time.perf_counter() - started

    print(f"\n  store after : {store.snapshot()} (denied writes never landed)")
    print(f"  rejections  : {[(r['tool'], r['reason']) for r in executor.rejections]}")
    print(f"  rounds={summary['rounds']}, elapsed={elapsed_s:.1f}s")
    print(f"  final answer:\n    {summary['final_text']}")
    print_log(executor.log, title="live execution log")

    # Inline MLflow record (few lines, no separate files): runs+metrics only;
    # graph tracing would need a LangGraph object, which this loop is not.
    if mlflow_reachable():
        import mlflow

        from lib import setup_mlflow

        setup_mlflow()
        with mlflow.start_run(run_name="l3-live-permission-gate"):
            mlflow.log_params(
                {
                    "model": live_model_name(),
                    "lesson": "l3_tools_and_execution",
                    "scenario": "permission-gate-rejection",
                }
            )
            mlflow.log_metrics(
                {
                    "rounds": summary["rounds"],
                    "tool_calls": len(executor.log),
                    "denied": len(executor.rejections),
                    "side_effects_writes": store.writes,
                    "elapsed_s": elapsed_s,
                }
            )
        print("  (logged one MLflow run; server at localhost:5000)")
    else:
        print("  (MLflow not reachable: run skipped)")


if __name__ == "__main__":
    main()
