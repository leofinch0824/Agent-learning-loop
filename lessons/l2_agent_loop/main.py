"""L2 - The Agent Loop: decisions, tool cycles, and termination.

Scenario: a PCB process-parameter assistant with two deterministic tools
(`unit_convert`, `lookup_spec`). The SAME ReAct-style action-observation loop
is implemented three ways, all driven by the SAME scripted fake model so the
trajectories are directly comparable:

  (a) run_python_loop            -- a plain Python `while` loop
  (b) build_loop_graph           -- LangGraph, routing via conditional edges
  (c) build_loop_graph_with_command -- LangGraph, routing via Command(goto=...,
                                      update=...) inside the node return value

Topology (b):                  Topology (c): same nodes, but ZERO edges except
                               START -> model; every routing decision travels
  START -> model               in-band, inside the node's return value.
    |       ^
    v       |static back-edge
  tools ----+                 (c) has no back-edge at all: the tools node
    |                         returns Command(goto="model") instead.
    +--> END (when a stop condition fires)

Model responses use the OpenAI chat-completions `tools` wire format (assistant
message dicts carrying `tool_calls`: id / name / arguments-as-JSON). The tools
node parses and dispatches them BY HAND on purpose -- `langgraph.prebuilt`
ToolNode does this for you in production, but hand-parsing is what makes the
loop observable and testable in this lesson.

Four terminations, tracked in state as `stop_reason`:

  success      model replied without tool_calls
  tool_error   ERROR_LIMIT consecutive tool errors -> runtime declares failure
  no_progress  the same (tool, arguments) requested twice -> runtime stops
  budget       MAX_STEPS model turns or TOKEN_BUDGET tokens exhausted

Run it:

    poetry run python lessons/l2_agent_loop/main.py

Design promise: every demo prints raw runtime output (stream chunks, tool
observations, the exact exception text of the probed pitfall), and all three
implementations provably follow the same policy -- test_main.py asserts
identical trajectories.
"""

from __future__ import annotations

import json
import operator
import sys
from pathlib import Path
from typing import Annotated, Callable, TypedDict

from langgraph.errors import InvalidUpdateError
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lib import (  # noqa: E402
    banner,
    build_checkpointer,
    print_history,
    print_state,
    show_graph,
    thread,
    trace,
)


def say(where: str, message: str) -> None:
    """Node-side console output (flushed so stream-order stays readable)."""
    print(f"    [{where}] {message}", flush=True)


# ---------------------------------------------------------------------------
# 1. Definitions: state, tools, structured envelopes, hand dispatch
# ---------------------------------------------------------------------------


class LoopState(TypedDict):
    """State of the agent loop.

    `messages` accumulates the whole chat history in OpenAI wire format, so it
    needs a reducer: model node and tools node both append, and the same
    channel will also be written by every later turn. Everything else is a
    LastValue channel with exactly one writer per super-step.
    """

    question: str
    messages: Annotated[list[dict], operator.add]
    steps: int  # model turns actually consumed
    tokens_used: int  # accumulated fake/live usage tokens
    consecutive_errors: int  # consecutive tool errors, reset by any success
    seen_calls: list[str]  # canonical "tool:args" keys already executed
    stop_reason: str  # "" while running; success/tool_error/no_progress/budget
    final_answer: str  # set only on the success termination


def initial_state(question: str) -> LoopState:
    return {
        "question": question,
        "messages": [{"role": "user", "content": question}],
        "steps": 0,
        "tokens_used": 0,
        "consecutive_errors": 0,
        "seen_calls": [],
        "stop_reason": "",
        "final_answer": "",
    }


# Loop policy shared by ALL THREE implementations. Same constants, same check
# order -- that is what makes the trajectories comparable.
MAX_STEPS = 6  # model-turn budget
TOKEN_BUDGET = 400  # accumulated usage-token budget (fake numbers offline)
ERROR_LIMIT = 3  # consecutive tool errors before the runtime declares failure

# Factors to millimetres. Small on purpose: the lesson is the loop, not the math.
UNIT_TO_MM = {"mm": 1.0, "cm": 10.0, "um": 0.001, "mil": 0.0254, "inch": 25.4}
VALUE_RANGE = (0.0, 10_000.0)  # plausible magnitude check -> "out of range" demo

SPEC_TABLE = {
    "min_trace_width_inner_mm": 0.127,
    "min_trace_width_outer_mm": 0.1,
    "default_via_diameter_mm": 0.3,
    "solder_mask_colors": ["green", "black", "blue", "matte_black"],
}


def _err(error: str, hint: str = "") -> dict:
    """A structured error envelope: the observation a tool returns instead of raising.

    This is the bad-params feedback loop: the error goes back into the message
    history as a normal tool observation, the run SURVIVES, and the model gets
    a chance to correct its arguments on the next turn.
    """
    envelope = {"ok": False, "error": error}
    if hint:
        envelope["hint"] = hint
    return envelope


def unit_convert(**kwargs) -> dict:
    """Convert a length between mm / cm / um / mil / inch.

    A *total* function over junk input: every bad argument becomes an error
    envelope, never an exception. See demo 3 for the contrast.
    """
    value, from_unit, to_unit = (
        kwargs.get("value"),
        kwargs.get("from_unit"),
        kwargs.get("to_unit"),
    )
    if value is None or from_unit is None or to_unit is None:
        return _err(
            "missing required argument",
            "unit_convert expects value (number), from_unit and to_unit",
        )
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return _err(f"'value' must be a number, got {value!r}")
    if not VALUE_RANGE[0] <= value <= VALUE_RANGE[1]:
        return _err(f"value {value} out of range {VALUE_RANGE}")
    for unit in (from_unit, to_unit):
        if unit not in UNIT_TO_MM:
            return _err(
                f"unknown unit {unit!r}",
                f"valid units: {sorted(UNIT_TO_MM)}",
            )
    converted = value * UNIT_TO_MM[from_unit] / UNIT_TO_MM[to_unit]
    return {"ok": True, "value": round(converted, 4), "unit": to_unit}


def lookup_spec(**kwargs) -> dict:
    """Look up a PCB process parameter from the deterministic spec table."""
    item = kwargs.get("item")
    if item is None:
        return _err("missing required argument", "lookup_spec expects item (str)")
    if not isinstance(item, str):
        return _err(f"'item' must be a string, got {item!r}")
    if item not in SPEC_TABLE:
        return _err(f"unknown spec item {item!r}", f"known items: {sorted(SPEC_TABLE)}")
    return {"ok": True, "item": item, "value": SPEC_TABLE[item]}


TOOL_FUNCTIONS: dict[str, Callable[..., dict]] = {
    "unit_convert": unit_convert,
    "lookup_spec": lookup_spec,
}

# The `tools` parameter we would hand to a real chat-completions endpoint.
# live_demo.py sends exactly this array; the fake model only *simulates* it.
OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "unit_convert",
            "description": "Convert a length value between mm, cm, um, mil and inch.",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {"type": "number"},
                    "from_unit": {"type": "string"},
                    "to_unit": {"type": "string"},
                },
                "required": ["value", "from_unit", "to_unit"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_spec",
            "description": "Look up a PCB process parameter (trace widths, via defaults, ...).",
            "parameters": {
                "type": "object",
                "properties": {"item": {"type": "string"}},
                "required": ["item"],
            },
        },
    },
]

# Execution journal: EVERY dispatch lands here as (tool_name, arguments_json).
# Tests count executions directly from it -- no mocking of the scheduler, the
# counter lives in the code under test.
tool_journal: list[tuple[str, str]] = []


def clear_tool_journal() -> None:
    tool_journal.clear()


def canonical_args(arguments: str) -> str:
    """Sort keys so `{"to":"mil","value":3}` and `{"value":3,"to":"mil"}` dedupe."""
    try:
        return json.dumps(json.loads(arguments), sort_keys=True)
    except json.JSONDecodeError:
        return arguments


def dispatch_tool(name: str, arguments: str) -> dict:
    """Parse a tool call and execute it, returning an envelope (never raising).

    Domain errors (unknown tool / bad JSON / invalid values) come back as
    `{"ok": false, ...}` observations. A crash INSIDE a tool would still raise
    and kill the run -- demo 3 shows both sides on purpose.
    """
    tool_journal.append((name, arguments))
    if name not in TOOL_FUNCTIONS:
        return _err(
            f"unknown tool {name!r}",
            f"known tools: {sorted(TOOL_FUNCTIONS)}",
        )
    try:
        args = json.loads(arguments)
    except json.JSONDecodeError as exc:
        return _err(f"arguments are not valid JSON ({exc})", "send a JSON object")
    if not isinstance(args, dict):
        return _err("arguments must be a JSON object")
    return TOOL_FUNCTIONS[name](**args)


def raising_dispatcher(name: str, arguments: str) -> dict:
    """The anti-pattern: a tool that raises instead of returning an envelope."""
    tool_journal.append((name, arguments))
    raise RuntimeError(f"tool {name!r} crashed before producing an observation")


# ---------------------------------------------------------------------------
# 2. The scripted fake model and its scripts (one per termination)
# ---------------------------------------------------------------------------


class ScriptExhaustedError(RuntimeError):
    """The policy failed to stop the loop before the script ran dry.

    Reaching this error is a BUG in the termination logic: every script below
    is designed to be stopped by the runtime before its items are consumed.
    """


def tool_call_msg(call_id: str, name: str, args: dict) -> dict:
    """An assistant message in the OpenAI `tools` wire format."""
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(args, sort_keys=True),
                },
            }
        ],
    }


def final_msg(text: str) -> dict:
    return {"role": "assistant", "content": text}


USAGE = {"prompt_tokens": 30, "completion_tokens": 20}  # 50 fake tokens / turn


def _item(message: dict, usage: dict | None = None) -> dict:
    return {"message": message, "usage": usage or USAGE}


class FakeModel:
    """A scripted model: responses pop from a queue, in order.

    `respond(messages)` has the same signature and return shape as the live
    OpenAI responder in live_demo.py, so all three loop implementations and
    the live loop share one interface. `consumed` records what was actually
    handed out -- the direct model-call counter used by tests.
    """

    def __init__(self, script: list[dict]) -> None:
        self.script = list(script)
        self.consumed: list[dict] = []

    def respond(self, messages: list[dict]) -> dict:
        if not self.script:
            raise ScriptExhaustedError(
                f"script ran dry after {len(self.consumed)} turns -- a stop "
                "condition failed to fire; messages tail: "
                f"{[m.get('role') for m in messages[-3:]]}"
            )
        item = self.script.pop(0)
        self.consumed.append(item)
        return item


def script_success() -> list[dict]:
    """Happy path: two tool rounds, then a final answer. Ends via `success`."""
    return [
        _item(tool_call_msg("c1", "unit_convert", {"value": 3, "from_unit": "mm", "to_unit": "mil"})),
        _item(tool_call_msg("c2", "lookup_spec", {"item": "min_trace_width_inner_mm"})),
        _item(final_msg("3 mm 约为 118.11 mil；内层最小线宽为 0.127 mm。")),
    ]


def script_bad_params_recovery() -> list[dict]:
    """Bad params -> error envelope fed back -> model corrects itself."""
    return [
        _item(tool_call_msg("c1", "unit_convert", {"value": 3, "from_unit": "mm", "to_unit": "milimeter"})),
        _item(tool_call_msg("c2", "unit_convert", {"value": 3, "from_unit": "mm", "to_unit": "mil"})),
        _item(final_msg("更正单位后：3 mm 约为 118.11 mil。")),
    ]


def script_failure() -> list[dict]:
    """ERROR_LIMIT consecutive errors with DISTINCT args -> `tool_error` stop.

    The args must differ each turn, otherwise the no-progress detector fires
    first -- the two detectors interact (README, 关键反例 3).
    The trailing final answer is never consumed: the runtime stops the loop,
    not the model.
    """
    return [
        _item(tool_call_msg("c1", "unit_convert", {"value": 3, "from_unit": "mm", "to_unit": "milimeter"})),
        _item(tool_call_msg("c2", "unit_convert", {"value": 4, "from_unit": "mm", "to_unit": "milimeter"})),
        _item(tool_call_msg("c3", "unit_convert", {"value": 5, "from_unit": "mm", "to_unit": "milimetres"})),
        _item(final_msg("never reached -- the runtime declared failure first")),
    ]


def script_no_progress() -> list[dict]:
    """The same tool + same args requested twice -> `no_progress` stop.

    The second REQUEST becomes an assistant message (a model turn), but the
    tools node refuses to execute it a second time.
    """
    return [
        _item(tool_call_msg("c1", "unit_convert", {"value": 5, "from_unit": "mm", "to_unit": "mil"})),
        _item(tool_call_msg("c2", "unit_convert", {"value": 5, "from_unit": "mm", "to_unit": "mil"})),
    ]


def script_budget() -> list[dict]:
    """Distinct converts forever -> stopped by MAX_STEPS / TOKEN_BUDGET."""
    return [
        _item(tool_call_msg(f"c{i}", "unit_convert", {"value": i, "from_unit": "mm", "to_unit": "mil"}))
        for i in range(1, 11)
    ]


# ---------------------------------------------------------------------------
# 3. Implementation (a): the plain Python while loop
# ---------------------------------------------------------------------------


def run_python_loop(
    respond: Callable[[list[dict]], dict],
    question: str,
    *,
    max_steps: int = MAX_STEPS,
    token_budget: int = TOKEN_BUDGET,
    dispatcher: Callable[[str, str], dict] = dispatch_tool,
) -> LoopState:
    """The whole agent loop as one while-loop: the reference implementation.

    Read it top to bottom and note that every `break` corresponds to an edge
    decision in the graph versions: the budget gate is the model node's first
    check, `success` is the model routing to END, and the tools-step checks
    are the tools node's routing. Same policy constants, same check order.
    """
    state = initial_state(question)
    while True:
        # -- model turn: budget gate FIRST, before consuming a response --
        if state["steps"] >= max_steps or state["tokens_used"] >= token_budget:
            state["stop_reason"] = "budget"
            break
        item = respond(state["messages"])
        message, usage = item["message"], item.get("usage") or {}
        state["messages"] = state["messages"] + [message]
        state["steps"] += 1
        state["tokens_used"] += (
            usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
        )
        calls = message.get("tool_calls")
        if not calls:
            state["stop_reason"] = "success"
            state["final_answer"] = message.get("content") or ""
            break
        # -- tools step: no-progress check BEFORE executing the batch --
        seen, keys = list(state["seen_calls"]), []
        duplicate = False
        for call in calls:
            key = f"{call['function']['name']}:{canonical_args(call['function']['arguments'])}"
            if key in seen or key in keys:
                duplicate = True
                break
            keys.append(key)
        if duplicate:
            state["stop_reason"] = "no_progress"
            break
        observations, errors = [], 0
        for call in calls:
            envelope = dispatcher(call["function"]["name"], call["function"]["arguments"])
            observations.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": json.dumps(envelope, ensure_ascii=False),
                }
            )
            errors += 0 if envelope.get("ok") else 1
        state["messages"] = state["messages"] + observations
        state["seen_calls"] = seen + keys
        # any success resets the streak; only error batches grow it
        state["consecutive_errors"] = (
            state["consecutive_errors"] + errors if errors else 0
        )
        if state["consecutive_errors"] >= ERROR_LIMIT:
            state["stop_reason"] = "tool_error"
            break
    return state


# ---------------------------------------------------------------------------
# 4. Shared turn logic + graph nodes (conditional-edge AND Command variants)
# ---------------------------------------------------------------------------


def model_turn(
    state: dict,
    respond: Callable[[list[dict]], dict],
    *,
    max_steps: int,
    token_budget: int,
) -> dict:
    """One model turn as a partial state update (used by BOTH graph variants).

    Check order is the contract: budget gate -> consume -> success detection.
    The budget branch deliberately returns WITHOUT consuming a script item,
    so `steps`/`consumed` stay honest counters of work actually done.
    """
    steps, tokens = state.get("steps", 0), state.get("tokens_used", 0)
    if steps >= max_steps or tokens >= token_budget:
        return {"stop_reason": "budget"}
    item = respond(state.get("messages") or [])
    message, usage = item["message"], item.get("usage") or {}
    update: dict = {
        "messages": [message],
        "steps": steps + 1,
        "tokens_used": tokens
        + usage.get("prompt_tokens", 0)
        + usage.get("completion_tokens", 0),
    }
    if not message.get("tool_calls"):
        update["stop_reason"] = "success"
        update["final_answer"] = message.get("content") or ""
    return update


def tools_step(state: dict, dispatcher: Callable[[str, str], dict]) -> dict:
    """One tools turn as a partial state update (used by BOTH graph variants)."""
    calls = state["messages"][-1].get("tool_calls") or []
    seen, keys = list(state.get("seen_calls") or []), []
    for call in calls:
        key = f"{call['function']['name']}:{canonical_args(call['function']['arguments'])}"
        if key in seen or key in keys:
            # Refuse to execute a repeated request; route to END afterwards.
            return {"stop_reason": "no_progress"}
        keys.append(key)
    observations, errors = [], 0
    for call in calls:
        envelope = dispatcher(call["function"]["name"], call["function"]["arguments"])
        observations.append(
            {
                "role": "tool",
                "tool_call_id": call["id"],
                "content": json.dumps(envelope, ensure_ascii=False),
            }
        )
        errors += 0 if envelope.get("ok") else 1
    consecutive = state.get("consecutive_errors", 0)
    consecutive = consecutive + errors if errors else 0
    update: dict = {
        "messages": observations,
        "seen_calls": seen + keys,
        "consecutive_errors": consecutive,
    }
    if consecutive >= ERROR_LIMIT:
        update["stop_reason"] = "tool_error"
    return update


def make_model_node(respond, *, max_steps, token_budget):
    """(b) Conditional-edge variant: the node returns a plain partial update."""

    def model_node(state: LoopState) -> dict:
        return model_turn(state, respond, max_steps=max_steps, token_budget=token_budget)

    return model_node


def make_tools_node(dispatcher=dispatch_tool):
    def tools_node(state: LoopState) -> dict:
        return tools_step(state, dispatcher)

    return tools_node


def route_after_model(state: LoopState) -> str:
    """Conditional edge: stop_reason set -> END; tool_calls -> tools; else END."""
    if state.get("stop_reason"):
        return END
    return "tools" if state["messages"][-1].get("tool_calls") else END


def route_after_tools(state: LoopState) -> str:
    return END if state.get("stop_reason") else "model"


def build_loop_graph(
    respond: Callable[[list[dict]], dict],
    *,
    checkpointer=None,
    dispatcher: Callable[[str, str], dict] = dispatch_tool,
    max_steps: int = MAX_STEPS,
    token_budget: int = TOKEN_BUDGET,
):
    """(b) The loop as a graph: conditional edges carry every routing decision."""
    builder = StateGraph(LoopState)
    builder.add_node("model", make_model_node(respond, max_steps=max_steps, token_budget=token_budget))
    builder.add_node("tools", make_tools_node(dispatcher))
    builder.add_edge(START, "model")
    builder.add_conditional_edges("model", route_after_model, ["tools", END])
    # The back-edge that turns a DAG into a loop: tools -> model.
    builder.add_conditional_edges("tools", route_after_tools, ["model", END])
    return builder.compile(checkpointer=checkpointer) if checkpointer else builder.compile()


def make_command_model_node(respond, *, max_steps, token_budget):
    """(c) Command variant: the SAME update, but the route rides along with it.

    `Command(goto=..., update=...)` merges "what I write" and "where to go
    next" into one return value -- the update+routing combination that
    DESIGN.md's corrections table asks us to verify.
    """

    def model_node(state: LoopState) -> Command:
        update = model_turn(state, respond, max_steps=max_steps, token_budget=token_budget)
        if update.get("stop_reason") or not update.get("messages"):
            return Command(goto=END, update=update)
        goto = "tools" if update["messages"][-1].get("tool_calls") else END
        return Command(goto=goto, update=update)

    return model_node


def make_command_tools_node(dispatcher=dispatch_tool):
    def tools_node(state: LoopState) -> Command:
        update = tools_step(state, dispatcher)
        goto = END if update.get("stop_reason") else "model"
        return Command(goto=goto, update=update)

    return tools_node


def build_loop_graph_with_command(
    respond: Callable[[list[dict]], dict],
    *,
    checkpointer=None,
    dispatcher: Callable[[str, str], dict] = dispatch_tool,
    max_steps: int = MAX_STEPS,
    token_budget: int = TOKEN_BUDGET,
):
    """(c) Same loop, ZERO declared edges besides the entrypoint.

    Every hop -- including the back-edge -- travels inside Command(goto=...).
    Compare the mermaid output with build_loop_graph: same behaviour, no edges.
    """
    builder = StateGraph(LoopState)
    builder.add_node("model", make_command_model_node(respond, max_steps=max_steps, token_budget=token_budget))
    builder.add_node("tools", make_command_tools_node(dispatcher))
    builder.add_edge(START, "model")
    return builder.compile(checkpointer=checkpointer) if checkpointer else builder.compile()


def trajectory_key(state: dict, journal: list[tuple[str, str]]) -> tuple:
    """The comparable trajectory of one run: what happened, in which order.

    Two runs with equal keys executed the same model turns, the same tool
    calls in the same order, and stopped for the same reason.
    """
    return (
        state.get("stop_reason"),
        state.get("steps"),
        state.get("tokens_used"),
        state.get("final_answer"),
        tuple(m.get("role") for m in state.get("messages") or []),
        tuple(journal),
    )


# ---------------------------------------------------------------------------
# 5. The key pitfall: Command return + static outgoing edge
# ---------------------------------------------------------------------------
# Probed on langgraph 1.2.11 (see README, 关键反例 2):
#   * compile() does NOT complain; nothing is validated against Command.
#   * Command(goto=...) does NOT replace/suppress a static edge declared with
#     add_edge -- both destinations are scheduled in the SAME super-step.
#   * If both destinations write the same LastValue channel, invoke() raises:
#       InvalidUpdateError: At key 'messages': Can receive only one value per
#       step. Use an Annotated key to handle multiple values.
#   * If they write DIFFERENT channels there is NO error at all: both simply
#     run. Silent double-scheduling -- arguably worse than the crash.


class ConflictState(TypedDict):
    messages: list[dict]  # NO reducer -- the collision is the point


def build_command_static_conflict_graph():
    """`agent` routes via Command but ALSO has a static edge: both fire."""

    def agent(state: ConflictState) -> Command:
        return Command(goto="tools", update={"messages": [{"from": "agent-via-Command"}]})

    def tools(state: ConflictState) -> Command:
        return Command(goto=END, update={"messages": [{"from": "tools"}]})

    def fallback(state: ConflictState) -> dict:
        return {"messages": [{"from": "agent-via-static-edge"}]}

    builder = StateGraph(ConflictState)
    builder.add_node("agent", agent)
    builder.add_node("tools", tools)
    builder.add_node("fallback", fallback)
    builder.add_edge(START, "agent")
    builder.add_edge("agent", "fallback")  # the forgotten static edge
    builder.add_edge("fallback", END)
    return builder.compile()


class SilentState(TypedDict):
    a: list[str]
    b: list[str]


def build_command_static_silent_graph(recorder: list[str]):
    """Same mistake, channels that do not collide -> NO error, both run.

    This is the sneaky variant: nothing crashes, `via_command` and
    `via_static` both execute, and the only evidence is direct counting.
    """

    def agent(state: SilentState) -> Command:
        recorder.append("agent")
        return Command(goto="tools", update={"a": ["agent-wrote-a"]})

    def tools(state: SilentState) -> Command:
        recorder.append("tools")
        return Command(goto=END, update={"b": ["tools-wrote-b"]})

    def fallback(state: SilentState) -> dict:
        recorder.append("fallback")
        return {"a": ["fallback-wrote-a-too"]}

    builder = StateGraph(SilentState)
    builder.add_node("agent", agent)
    builder.add_node("tools", tools)
    builder.add_node("fallback", fallback)
    builder.add_edge(START, "agent")
    builder.add_edge("agent", "fallback")  # fires alongside Command(goto="tools")
    builder.add_edge("fallback", END)
    return builder.compile()


# ---------------------------------------------------------------------------
# 6. Demos
# ---------------------------------------------------------------------------


def _fmt_journal(journal: list[tuple[str, str]]) -> str:
    return " -> ".join(f"{name}({args})" for name, args in journal) or "(none)"


def demo_success_and_parity():
    banner("DEMO 1 - 成功闭环：同一脚本驱动三种实现")
    question = "3 mm 是多少 mil？内层最小线宽是多少 mm？"
    clear_tool_journal()
    fake_py = FakeModel(script_success())
    py_state = run_python_loop(fake_py.respond, question)
    py_journal = list(tool_journal)

    clear_tool_journal()
    fake_cond = FakeModel(script_success())
    cond_state = build_loop_graph(fake_cond.respond).invoke(initial_state(question))
    cond_journal = list(tool_journal)

    clear_tool_journal()
    fake_cmd = FakeModel(script_success())
    cmd_state = build_loop_graph_with_command(fake_cmd.respond).invoke(initial_state(question))
    cmd_journal = list(tool_journal)

    rows = [
        ("python while", py_state, py_journal, len(fake_py.consumed)),
        ("conditional edges", cond_state, cond_journal, len(fake_cond.consumed)),
        ("Command routing", cmd_state, cmd_journal, len(fake_cmd.consumed)),
    ]
    print(f"  question: {question}\n")
    print(f"  {'implementation':<20} {'steps':<6} {'tokens':<7} {'stop':<9} tool calls")
    for name, state, journal, _ in rows:
        print(
            f"  {name:<20} {state['steps']:<6} {state['tokens_used']:<7} "
            f"{state['stop_reason']:<9} {_fmt_journal(journal)}"
        )
    py_key = trajectory_key(py_state, py_journal)
    cond_key = trajectory_key(cond_state, cond_journal)
    cmd_key = trajectory_key(cmd_state, cmd_journal)
    print(f"\n  python == conditional edges : {py_key == cond_key}")
    print(f"  python == Command routing   : {py_key == cmd_key}")
    print(f"  final answer: {cond_state['final_answer']}")


def demo_streaming():
    banner("DEMO 2 - 流式事件：updates 与 values 各显示什么")
    question = "3 mm 是多少 mil？内层最小线宽是多少 mm？"
    fake = FakeModel(script_success())
    graph = build_loop_graph(fake.respond, checkpointer=build_checkpointer("memory"))
    config = thread("l2-streaming")
    # lib trace() prints every raw chunk of stream_mode=["updates","values"]:
    # `updates` = one chunk per completed TASK (node run), `values` = the full
    # state snapshot after each super-step.
    trace(graph, initial_state(question), config, label="success run (updates + values)")

    print("\n  values-only stream (full state after each super-step):")
    fake2 = FakeModel(script_success())
    for chunk in build_loop_graph(fake2.respond).stream(initial_state(question), stream_mode="values"):
        print(
            f"    steps={chunk['steps']} tokens={chunk['tokens_used']:>3} "
            f"stop_reason={chunk['stop_reason']!r:<13} roles={[m['role'] for m in chunk['messages']]}"
        )

    print("\n  从最后几个 chunk 读出停止原因:")
    print("  * 最后一个 updates chunk 是 model 写入 stop_reason='success' + final_answer")
    print("  * 最后一个 values chunk 的 stop_reason 字段非空且 next 为空 -> 图已终止")
    print_state(graph, config)
    print_history(graph, config)
    print("  NOTE: updates chunks count TASKS, not super-steps. Our loop runs one")
    print("  node per super-step so the numbers coincide; under fan-out (L1) they")
    print("  do not. The authoritative counter is checkpoint metadata.step above.")


def demo_bad_params_feedback():
    banner("DEMO 3 - 关键反例 1：坏参数回传（envelope vs raise）")
    question = "3 mm 是多少 mil？"
    clear_tool_journal()
    fake = FakeModel(script_bad_params_recovery())
    state = build_loop_graph(fake.respond).invoke(initial_state(question))
    print("  有回传的运行（错误作为 observation 进入消息历史，模型下一轮自我纠正）:")
    for msg in state["messages"]:
        role = msg["role"]
        if role == "user":
            print(f"    user             : {msg['content']}")
        elif role == "tool":
            print(f"    tool observation : {msg['content']}")
        elif msg.get("tool_calls"):
            call = msg["tool_calls"][0]
            print(f"    assistant calls  : {call['function']['name']}({call['function']['arguments']})")
        else:
            print(f"    assistant final  : {msg['content']}")
    print(
        f"  -> stop_reason={state['stop_reason']!r}, consecutive_errors reset to "
        f"{state['consecutive_errors']}, journal={len(tool_journal)} executions"
    )

    print("\n  对照：工具直接 raise（没有 envelope）——整次运行死掉:")
    clear_tool_journal()
    fake_boom = FakeModel(script_success())
    graph_boom = build_loop_graph(fake_boom.respond, dispatcher=raising_dispatcher)
    try:
        graph_boom.invoke(initial_state(question))
    except RuntimeError as exc:
        print(f"    invoke raised: RuntimeError: {exc}")
        print(f"    journal shows {len(tool_journal)} attempted execution; no observation")
        print("    was ever appended -- the feedback loop never got a chance.")
    else:
        raise AssertionError("the raising dispatcher must kill the run")


def demo_failure_termination():
    banner("DEMO 4 - 失败终止：连续 tool error 达到 ERROR_LIMIT")
    question = "把 3、4、5 mm 换算成 mil"
    clear_tool_journal()
    fake = FakeModel(script_failure())
    state = build_loop_graph(fake.respond).invoke(initial_state(question))
    ok_flags = ["ok" in json.loads(m["content"]) and json.loads(m["content"])["ok"] for m in state["messages"] if m["role"] == "tool"]
    print(f"  consecutive_errors={state['consecutive_errors']} (limit {ERROR_LIMIT})")
    print(f"  tool observations ok flags: {ok_flags}")
    print(f"  stop_reason={state['stop_reason']!r}  steps={state['steps']}  executions={len(tool_journal)}")
    print(f"  script items consumed={len(fake.consumed)} / {len(fake.consumed) + len(fake.script)} (trailing final answer ignored)")
    print("  终止是 runtime 的决定：模型永远看不到第三次错误回传。")


def demo_no_progress_termination():
    banner("DEMO 5 - 无进展终止：同一 tool+args 被要求第二次")
    question = "5 mm 是多少 mil？"
    clear_tool_journal()
    fake = FakeModel(script_no_progress())
    state = build_loop_graph(fake.respond).invoke(initial_state(question))
    print(f"  requested twice: unit_convert(value=5, mm->mil)")
    print(f"  executions={len(tool_journal)} (second request NOT executed)")
    print(f"  stop_reason={state['stop_reason']!r}  steps={state['steps']}")
    print("  最后一个 updates chunk 只有 tools 写入 stop_reason，没有任何 observation。")


def demo_budget_termination():
    banner("DEMO 6 - 预算终止：MAX_STEPS 与 TOKEN_BUDGET")
    question = "把 1..10 mm 逐个换算成 mil"
    clear_tool_journal()
    fake = FakeModel(script_budget())
    state = build_loop_graph(fake.respond).invoke(initial_state(question))
    print(f"  [steps budget]  MAX_STEPS={MAX_STEPS}: steps={state['steps']}, "
          f"consumed={len(fake.consumed)}/10, stop_reason={state['stop_reason']!r}")

    clear_tool_journal()
    fake_tok = FakeModel(script_budget())
    state_tok = build_loop_graph(fake_tok.respond, token_budget=120).invoke(initial_state(question))
    print(f"  [token budget]  TOKEN_BUDGET=120: tokens_used={state_tok['tokens_used']}, "
          f"steps={state_tok['steps']}, consumed={len(fake_tok.consumed)}/10, "
          f"stop_reason={state_tok['stop_reason']!r}")
    print("  两种预算同属一个 stop_reason；最后 updates chunk 是 model 只写了")
    print("  {'stop_reason': 'budget'} 而没有追加任何消息——这就是 trace 上的指纹。")


def demo_command_static_pitfall():
    banner("DEMO 7 - 关键反例 2：Command 与静态边并存")
    print("  节点返回 Command(goto='tools')，同时builder留着 add_edge('agent','fallback')。")
    print("  compile() 不报错；invoke() 时两个目的地进入同一 super-step：\n")
    try:
        build_command_static_conflict_graph().invoke({"messages": []})
    except InvalidUpdateError as exc:
        print(f"    raised as probed:\n      {type(exc).__name__}: {exc}")
        print("    修复：Command 节点不要再挂静态出边——路由只写在一处。")

    print("\n  通道不冲突的同款错误（更隐蔽）：两个目的地都执行，无任何异常:")
    recorder: list[str] = []
    final = build_command_static_silent_graph(recorder).invoke({"a": [], "b": []})
    print(f"    execution order: {recorder}")
    print(f"    final state     : {final}")
    print("    Command 没有压掉静态边，goto 是叠加而不是替换——只有直接计数能发现。")


if __name__ == "__main__":
    demo_success_and_parity()
    demo_streaming()
    demo_bad_params_feedback()
    demo_failure_termination()
    demo_no_progress_termination()
    demo_budget_termination()
    demo_command_static_pitfall()
    banner("done", width=72)
