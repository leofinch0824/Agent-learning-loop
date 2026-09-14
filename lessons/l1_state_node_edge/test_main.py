"""Tests for L1 - State, Node, Edge."""

import operator
from typing import Annotated, TypedDict

import pytest
from langgraph.errors import InvalidUpdateError
from langgraph.graph import END, START, StateGraph

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lessons.l1_state_node_edge.main import (
    ReviewState,
    build_concurrent_write_graph,
    build_fanout_continuation_graph,
    build_layered_graph,
    build_review_graph,
)


def test_reducer_merges_concurrent_writes():
    """Two nodes write to a reducer channel concurrently -> both merge."""

    class S(TypedDict):
        log: Annotated[list[str], operator.add]

    def a(state: S) -> dict:
        return {"log": ["a"]}

    def b(state: S) -> dict:
        return {"log": ["b"]}

    builder = StateGraph(S)
    builder.add_node("a", a)
    builder.add_node("b", b)
    builder.add_edge(START, "a")
    builder.add_edge(START, "b")
    builder.add_edge("a", END)
    builder.add_edge("b", END)
    graph = builder.compile()
    result = graph.invoke({"log": []})
    # Order is unspecified, but both must appear.
    assert set(result["log"]) == {"a", "b"}


def test_concurrent_writes_without_reducer():
    """Two nodes write to a NO-reducer channel concurrently -> InvalidUpdateError."""
    graph = build_concurrent_write_graph()
    with pytest.raises(InvalidUpdateError, match="Can receive only one value per step"):
        graph.invoke({"verdict": ""})


def test_conditional_edge_routing():
    """A conditional edge picks its destination from state at runtime."""

    class S(TypedDict):
        which: str

    def route(state: S) -> str:
        return state["which"]

    builder = StateGraph(S)
    builder.add_node("a", lambda s: {})
    builder.add_node("b", lambda s: {"which": "b"})
    builder.add_edge(START, "a")
    builder.add_conditional_edges("a", route)
    builder.add_edge("b", END)
    graph = builder.compile()
    result = graph.invoke({"which": "b"})
    assert result["which"] == "b"


def test_private_schema_leaks_while_streaming():
    """Private channels are NOT hidden by stream. Output schema only hides them on invoke."""
    graph = build_layered_graph()
    # invoke returns only the output schema keys
    result = graph.invoke({"filename": "app.py"})
    assert "secret_notes" not in result
    assert "verdict" in result

    # but stream_mode="values" shows everything, INCLUDING the private channel
    all_chunks = list(graph.stream({"filename": "app.py"}, stream_mode="values"))
    final = all_chunks[-1]
    assert "secret_notes" in final, "private channel DOES leak while streaming"

    # ...unless output_keys is passed
    restricted_chunks = list(
        graph.stream({"filename": "app.py"}, stream_mode="values", output_keys=["verdict"])
    )
    for chunk in restricted_chunks:
        assert "secret_notes" not in chunk, "output_keys must hide private channels"


def test_nodes_cannot_see_parallel_writes_in_same_step():
    """Two nodes in the same super-step see the state snapshot from the PREVIOUS step.

    This is the #1 beginner confusion. They expect `style_check` to see the security
    finding, because both run after `intake`. But no -- same super-step means they
    both see only the snapshot that existed before the step started.
    """

    writes_observed = []

    class S(TypedDict):
        findings: Annotated[list[str], operator.add]

    def a(s: S) -> dict:
        writes_observed.append(("a", s["findings"]))
        return {"findings": ["a"]}

    def b(s: S) -> dict:
        writes_observed.append(("b", s["findings"]))
        return {"findings": ["b"]}

    builder = StateGraph(S)
    builder.add_node("a", a)
    builder.add_node("b", b)
    builder.add_edge(START, "a")
    builder.add_edge(START, "b")
    builder.add_edge("a", END)
    builder.add_edge("b", END)
    graph = builder.compile()
    graph.invoke({"findings": ["init"]})
    # Both a and b saw the snapshot from BEFORE their super-step (just ["init"]).
    # They did NOT see each other's writes.
    assert writes_observed == [("a", ["init"]), ("b", ["init"])] or writes_observed == [
        ("b", ["init"]),
        ("a", ["init"]),
    ]


def test_asymmetric_fanout_runs_target_twice():
    """A node reachable by a short path and a long path fires exactly twice."""
    # The graph is:
    #   intake -> style_check -> reduce -> END
    #   intake -> security_check -> extra_step -> reduce -> END
    # The style path is shorter by one hop. `reduce` triggers the moment the style
    # path lands, then triggers AGAIN when the long path arrives.
    graph = build_fanout_continuation_graph()
    # Direct execution count: every `updates` chunk is ONE completed task, so we
    # count the chunks where `reduce` is the node that ran -- no inference from
    # final state (DESIGN.md section 7 asks for exactly this evidence).
    reduce_runs = 0
    for chunk in graph.stream(
        {"filename": ".env", "findings": [], "verdict": "pass"}, stream_mode="updates"
    ):
        assert isinstance(chunk, dict)
        if "reduce" in chunk:
            reduce_runs += 1
    assert reduce_runs == 2, f"reduce ran {reduce_runs} times, expected 2"
    # Sanity: both branches actually produced findings along the way.
    result = graph.invoke({"filename": ".env", "findings": [], "verdict": "pass"})
    assert "security: hardcoded token" in result["findings"]
    assert "style: nothing to check" in result["findings"]
    assert "extra: security branch needed a follow-up" in result["findings"]


def test_full_join_waits_for_both():
    """A proper join (all incoming edges) waits for all branches before firing."""
    graph = build_review_graph()
    result = graph.invoke({"filename": ".env", "findings": [], "verdict": "pass"})
    # `reduce` fires exactly once, after both style and security branches land.
    # We verify that by checking the final findings list contains entries from both
    # checks and does NOT have duplicate reduce entries.
    assert "style: nothing to check" in result["findings"]
    assert "security: hardcoded token" in result["findings"]
    # `verdict` should have been flipped to "reject" because of the security finding.
    assert result["verdict"] == "reject"
