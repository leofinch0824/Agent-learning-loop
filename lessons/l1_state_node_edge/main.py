"""L1 - State / Node / Edge.

The scenario is a tiny code-review pipeline, chosen because it has all three
shapes you will meet in real agents:

    intake -> (style_check || security_check) -> reduce -> END
              ^ fan-out: two nodes in ONE super-step

Run it:

    poetry run python lessons/l1_state_node_edge/main.py

Every section below prints raw runtime output rather than a summary, so the
super-step boundaries are visible.
"""

from __future__ import annotations

import operator
import sys
from pathlib import Path
from typing import Annotated, Literal, Sequence, TypedDict

from langgraph.errors import InvalidUpdateError
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lib import banner, print_state, show_graph, step, thread, trace  # noqa: E402


def say(node: str, message: str) -> None:
    """Node-side console output.

    `flush=True` matters: nodes in the same super-step run on different threads,
    so unflushed prints get reordered and the lesson becomes unreadable.
    """
    print(f"    [{node}] {message}", flush=True)

# ---------------------------------------------------------------------------
# 1. State: channels + reducers
# ---------------------------------------------------------------------------


class ReviewState(TypedDict):
    # No reducer -> LastValue channel. Written by exactly one node per step.
    filename: str
    # Annotated + reducer -> a parallel-capable channel. N writers per step merge.
    findings: Annotated[list[str], operator.add]
    # Routing key: how a conditional edge picks its destination.
    verdict: Literal["pass", "reject"]


# ---------------------------------------------------------------------------
# 2. Nodes: state -> partial update
# ---------------------------------------------------------------------------


def intake(state: ReviewState) -> dict:
    say("intake", f"reads filename={state['filename']!r}")
    return {
        "filename": state["filename"],
        "findings": [f"intake: {state['filename']}"],
        "verdict": "pass",
    }


def style_check(state: ReviewState) -> dict:
    # NOTE: this node cannot see security_check's write, even though both run in
    # the same super-step. It sees the snapshot produced by the previous step.
    say("style_check", f"state it can see: {state['findings']}")
    if state["filename"].endswith(".py"):
        return {"findings": ["style: line too long"]}
    return {"findings": ["style: nothing to check"]}


def security_check(state: ReviewState) -> dict:
    say("security_check", f"state it can see: {state['findings']}")
    # Deterministic stand-in for a real scanner: `.env` files are rejected.
    if state["filename"].endswith(".env"):
        return {"findings": ["security: hardcoded token"]}
    return {"findings": ["security: no secrets found"]}


def reduce_findings(state: ReviewState) -> dict:
    # Both upstream branches have joined by the time this runs: `findings` holds
    # every write from the fan-out step, folded left by operator.add.
    say("reduce", f"merged findings: {state['findings']}")
    verdict = "reject" if any("security: hardcoded" in f for f in state["findings"]) else state["verdict"]
    return {"verdict": verdict}


# ---------------------------------------------------------------------------
# 3. Edges: static fan-out, static join, dynamic routing
# ---------------------------------------------------------------------------


def build_review_graph(*, checkpointer=None):
    builder = StateGraph(ReviewState)
    builder.add_node("intake", intake)
    builder.add_node("style_check", style_check)
    builder.add_node("security_check", security_check)
    builder.add_node("reduce", reduce_findings)
    builder.add_node("accept", lambda state: {"findings": ["accept: merged into main"]})

    builder.add_edge(START, "intake")
    # Fan-out: two edges from one node -> two nodes in the SAME super-step.
    builder.add_edge("intake", "style_check")
    builder.add_edge("intake", "security_check")
    # Join: `reduce` is scheduled once all of `style_check`/`security_check` have
    # finished, because these are its only incoming edges.
    builder.add_edge("style_check", "reduce")
    builder.add_edge("security_check", "reduce")
    # A conditional edge is just a function of state returning node name(s) or END.
    builder.add_conditional_edges(
        "reduce",
        lambda state: "accept" if state["verdict"] == "pass" else END,
    )
    builder.add_edge("accept", END)
    return builder.compile(checkpointer=checkpointer) if checkpointer else builder.compile()


# ---------------------------------------------------------------------------
# Demo 2: join semantics -- what happens if one incoming edge is removed
# ---------------------------------------------------------------------------


def build_partial_join_graph(*, checkpointer=None):
    """Same nodes as the full graph, but `style_check` dangles to END.

    Nothing crashes: the style finding is written to state, `reduce` just never
    reads it as an input.
    """
    builder = StateGraph(ReviewState)
    builder.add_node("intake", intake)
    builder.add_node("style_check", style_check)
    builder.add_node("security_check", security_check)
    builder.add_node("reduce", reduce_findings)
    builder.add_edge(START, "intake")
    builder.add_edge("intake", "style_check")
    builder.add_edge("intake", "security_check")
    builder.add_edge("style_check", END)  # dangling: never feeds reduce
    builder.add_edge("security_check", "reduce")
    builder.add_edge("reduce", END)
    return builder.compile(checkpointer=checkpointer) if checkpointer else builder.compile()


def build_fanout_continuation_graph(*, checkpointer=None):
    """The bug that actually bites: one target reachable by a SHORT and a LONG path.

    `intake` fans out to a fast branch and a slow branch. Both eventually point at
    `reduce`. The trigger rule is OR -- "any incoming edge whose source has run" --
    so `reduce` fires as soon as whichever branch finishes FIRST lands, and then
    fires AGAIN when the slower branch arrives. Same node, two invocations,
    different inputs, no error.

    Contrast with `build_review_graph`, where both branches are exactly one hop
    from `intake`: there the two writes land in the same super-step, so `reduce`
    is scheduled only once and does see everything. The lesson is that correctness
    of a fan-in depends on the *topology*, not on the presence of a join node.
    """
    builder = StateGraph(ReviewState)
    builder.add_node("intake", intake)
    builder.add_node("style_check", style_check)
    builder.add_node("security_check", security_check)
    builder.add_node("reduce", reduce_findings)

    builder.add_edge(START, "intake")
    builder.add_edge("intake", "style_check")
    builder.add_edge("intake", "security_check")
    # style_check -> reduce -> (END) is the SHORT path.
    builder.add_edge("style_check", "reduce")
    builder.add_edge("reduce", END)
    # security_check -> extra_step -> reduce is the LONG path.
    builder.add_node("extra_step", _extra_step)
    builder.add_edge("security_check", "extra_step")
    builder.add_edge("extra_step", "reduce")
    return builder.compile(checkpointer=checkpointer) if checkpointer else builder.compile()


def _extra_step(state: ReviewState) -> dict:
    say("extra_step", "a second hop on the security branch")
    return {"findings": ["extra: security branch needed a follow-up"]}


# ---------------------------------------------------------------------------
# Pitfall A: concurrent writes to a channel with no reducer
# ---------------------------------------------------------------------------


class BrokenState(TypedDict):
    verdict: str  # no reducer!


def build_concurrent_write_graph():
    builder = StateGraph(BrokenState)
    builder.add_node("w1", lambda state: {"verdict": "from w1"})
    builder.add_node("w2", lambda state: {"verdict": "from w2"})
    builder.add_edge(START, "w1")
    builder.add_edge(START, "w2")
    builder.add_edge("w1", END)
    builder.add_edge("w2", END)
    return builder.compile()


# ---------------------------------------------------------------------------
# Pitfall B: input / output / private schemas, and what stream leaks
# ---------------------------------------------------------------------------


class InputState(TypedDict):
    filename: str


class OutputState(TypedDict):
    verdict: str


class InternalState(TypedDict):
    filename: str
    verdict: str


class PrivateState(TypedDict):
    secret_notes: str  # never meant to leave the graph


def build_layered_graph():
    def classify(state: InputState) -> InternalState:
        return {"verdict": "pass" if state["filename"].endswith(".py") else "reject"}

    def annotate(state: InternalState) -> PrivateState:
        return {"secret_notes": f"internal note about {state['filename']}"}

    def emit(state: PrivateState) -> OutputState:
        return {"verdict": "pass" if state["secret_notes"] else "reject"}

    builder = StateGraph(InternalState, input_schema=InputState, output_schema=OutputState)
    builder.add_node("classify", classify)
    builder.add_node("annotate", annotate)
    builder.add_node("emit", emit)
    builder.add_edge(START, "classify")
    builder.add_edge("classify", "annotate")
    builder.add_edge("annotate", "emit")
    builder.add_edge("emit", END)
    return builder.compile()


# ---------------------------------------------------------------------------
# Pydantic schema: validation happens at the FIRST node's input, not at invoke()
# ---------------------------------------------------------------------------


class PydanticState(BaseModel):
    filename: str
    attempts: int = Field(default=0, ge=0)


def build_pydantic_graph():
    def report(state: PydanticState) -> dict:
        # state is a validated Pydantic instance here
        return {"attempts": state.attempts + 1}

    builder = StateGraph(PydanticState)
    builder.add_node("report", report)
    builder.add_edge(START, "report")
    builder.add_edge("report", END)
    return builder.compile()


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------


def demo_basic():
    banner("DEMO 1 - fan-out, join, reducer")
    graph = build_review_graph()
    show_graph(graph)
    trace(graph, {"filename": "app.py", "findings": [], "verdict": "pass"}, label="clean file")


def demo_join_semantics():
    banner("DEMO 2 - when does a node fire? (the trigger rule)")
    print("The rule: a node is scheduled in a super-step if ANY incoming edge comes")
    print("from a node that has already run (this step or an earlier one). It is an OR")
    print("rule. That is what makes fan-in work -- and what makes it fail.\n")
    print("Case A: a symmetric fan-out. Both checks are one hop from `intake`, so their")
    print("writes are committed in the SAME super-step, and `reduce` fires exactly once")
    print("with both findings merged. This is the shape everyone draws on the whiteboard.\n")
    trace(
        build_review_graph(),
        {"filename": ".env", "findings": [], "verdict": "pass"},
        label="A: symmetric fan-out",
    )
    print("\n")
    print("Case B: a longer branch. The security side needs a second hop, so `reduce`")
    print("becomes reachable by a short path and a long path. Because the rule is OR,")
    print("it fires the moment the SHORT path lands -- and fires AGAIN when the long")
    print("path arrives. Same node, two runs, different inputs, no exception.\n")
    trace(
        build_fanout_continuation_graph(),
        {"filename": ".env", "findings": [], "verdict": "pass"},
        label="B: asymmetric fan-out (reduce runs twice)",
    )
    print("\n  If you ever see a node run more times than you expected in L3's")
    print("  checkpoint history, this is the first thing to check.\n")


def demo_pitfall_a():
    banner("PITFALL A - two nodes, one super-step, channel without a reducer")
    graph = build_concurrent_write_graph()
    try:
        graph.invoke({"verdict": ""})
    except InvalidUpdateError as exc:
        print(f"  raised as expected:\n    {exc}")
        print("  fix: Annotated[str, reducer] or make the writes sequential.")


def demo_pitfall_b():
    banner("PITFALL B - private channels are NOT hidden while streaming")
    graph = build_layered_graph()
    final = graph.invoke({"filename": "app.py"})
    print(f"  invoke() returns only the output schema: {final}")
    print("  but stream_mode='values' shows everything:")
    for chunk in graph.stream({"filename": "app.py"}, stream_mode="values"):
        print(f"    {chunk}")
    print("  contained with output_keys:")
    for chunk in graph.stream({"filename": "app.py"}, stream_mode="values", output_keys=["verdict"]):
        print(f"    {chunk}")


def demo_pydantic():
    banner("DEMO 3 - pydantic schema validates at the first node")
    graph = build_pydantic_graph()
    print(f"  valid input   -> {graph.invoke({'filename': 'app.py'})}")
    try:
        graph.invoke({"filename": "app.py", "attempts": -1})
    except Exception as exc:  # noqa: BLE001 - we want to show the raw error
        print(f"  invalid input -> {type(exc).__name__}: {exc}")


def demo_branching_by_verdict():
    banner("DEMO 4 - conditional edge routes on state")
    graph = build_review_graph()
    trace(graph, {"filename": "app.py", "findings": [], "verdict": "pass"}, label="verdict=pass")
    trace(graph, {"filename": ".env", "findings": [], "verdict": "pass"}, label="verdict=reject")


if __name__ == "__main__":
    demo_basic()
    demo_join_semantics()
    demo_pitfall_a()
    demo_pitfall_b()
    demo_pydantic()
    demo_branching_by_verdict()
    banner("done", width=72)
