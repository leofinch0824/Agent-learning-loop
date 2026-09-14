"""Tests for L2 - The Agent Loop: decisions, tool cycles, and termination.

Every test counts executions DIRECTLY (model turns via FakeModel.consumed,
tool executions via tool_journal) instead of inferring them from final state.
No framework scheduler internals are mocked.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from langgraph.errors import InvalidUpdateError

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lessons.l2_agent_loop.main import (  # noqa: E402
    ERROR_LIMIT,
    MAX_STEPS,
    FakeModel,
    build_command_static_conflict_graph,
    build_command_static_silent_graph,
    build_loop_graph,
    build_loop_graph_with_command,
    clear_tool_journal,
    initial_state,
    raising_dispatcher,
    run_python_loop,
    script_bad_params_recovery,
    script_budget,
    script_failure,
    script_no_progress,
    script_success,
    tool_journal,
    trajectory_key,
)


def run_graph(build, script, question="q", **kwargs):
    """Run one graph on a fresh fake; return (state, journal, fake)."""
    clear_tool_journal()
    fake = FakeModel(script)
    state = build(fake.respond, **kwargs).invoke(initial_state(question))
    journal = list(tool_journal)
    clear_tool_journal()
    return state, journal, fake


def tool_observations(state) -> list[dict]:
    return [json.loads(m["content"]) for m in state["messages"] if m["role"] == "tool"]


def test_success_terminates_with_final_answer():
    """A tool-closing run ends via success with the answer and >=1 execution."""
    state, journal, fake = run_graph(build_loop_graph, script_success())
    assert state["stop_reason"] == "success"
    assert "118.11" in state["final_answer"] and "0.127" in state["final_answer"]
    assert len(journal) == 2  # unit_convert + lookup_spec, counted directly
    assert len(fake.consumed) == 3  # two tool turns + one final turn
    assert state["steps"] == 3 and state["tokens_used"] == 150


def test_python_loop_matches_conditional_graph():
    """Plain while-loop and conditional-edge graph produce IDENTICAL trajectories."""
    question = "3 mm 是多少 mil？内层最小线宽是多少 mm？"
    clear_tool_journal()
    fake = FakeModel(script_success())
    py_state = run_python_loop(fake.respond, question)
    py_journal = list(tool_journal)
    graph_state, graph_journal, _ = run_graph(build_loop_graph, script_success(), question)
    assert trajectory_key(py_state, py_journal) == trajectory_key(graph_state, graph_journal)


def test_command_graph_matches_conditional_graph():
    """Command routing produces the same trajectory as conditional edges."""
    graph_state, graph_journal, _ = run_graph(build_loop_graph, script_success())
    cmd_state, cmd_journal, _ = run_graph(build_loop_graph_with_command, script_success())
    assert trajectory_key(graph_state, graph_journal) == trajectory_key(cmd_state, cmd_journal)


def test_budget_termination_steps_and_tokens():
    """Both budget counters stop the loop without consuming the next script item."""
    # steps budget: MAX_STEPS=6 turns consumed out of 10 scripted
    state, journal, fake = run_graph(build_loop_graph, script_budget())
    assert state["stop_reason"] == "budget"
    assert state["steps"] == MAX_STEPS and len(fake.consumed) == MAX_STEPS
    assert len(journal) == MAX_STEPS
    assert state["final_answer"] == ""  # never reached an answer
    # token budget: 50 fake tokens/turn, budget 120 -> stops entering turn 4
    state_tok, journal_tok, fake_tok = run_graph(
        build_loop_graph, script_budget(), token_budget=120
    )
    assert state_tok["stop_reason"] == "budget"
    assert state_tok["steps"] == 3 and state_tok["tokens_used"] == 150
    assert len(fake_tok.consumed) == 3 and len(journal_tok) == 3


def test_tool_error_termination_after_consecutive_limit():
    """ERROR_LIMIT consecutive tool errors make the runtime declare failure."""
    state, journal, fake = run_graph(build_loop_graph, script_failure())
    assert state["stop_reason"] == "tool_error"
    assert state["consecutive_errors"] == ERROR_LIMIT
    assert len(journal) == ERROR_LIMIT  # every bad call WAS executed (fed back)
    assert all(env["ok"] is False for env in tool_observations(state))
    # the trailing "final answer" script item is never consumed: the runtime
    # stops the loop; the model never sees the third error observation
    assert len(fake.consumed) == 3 and len(fake.script) == 1


def test_no_progress_termination_skips_second_execution():
    """A repeated (tool, args) request is detected and NOT executed again."""
    state, journal, fake = run_graph(build_loop_graph, script_no_progress())
    assert state["stop_reason"] == "no_progress"
    assert len(journal) == 1  # second request refused by the runtime
    assert len(fake.consumed) == 2  # but it DID cost a model turn
    assert len(tool_observations(state)) == 1


def test_bad_params_are_fed_back_and_model_recovers():
    """An error envelope goes back as an observation; the next turn recovers."""
    state, journal, _ = run_graph(build_loop_graph, script_bad_params_recovery())
    observations = tool_observations(state)
    assert observations[0]["ok"] is False and "milimeter" in observations[0]["error"]
    assert observations[0]["hint"]  # the envelope carries a recovery hint
    assert observations[1]["ok"] is True and observations[1]["value"] == 118.1102
    assert state["consecutive_errors"] == 0  # reset by the successful call
    assert state["stop_reason"] == "success" and len(journal) == 2


def test_raising_tool_kills_the_run():
    """The same bad call through a RAISING tool kills the whole run instead."""
    clear_tool_journal()
    fake = FakeModel(script_success())
    graph = build_loop_graph(fake.respond, dispatcher=raising_dispatcher)
    with pytest.raises(RuntimeError, match="crashed before producing an observation"):
        graph.invoke(initial_state("3 mm 是多少 mil？"))
    assert len(tool_journal) == 1  # attempted, but no observation was ever fed back


def test_command_with_static_edge_raises_invalid_update():
    """A node returning Command while a static edge also leaves it collides.

    Probed on langgraph 1.2.11: compile() is silent; at invoke() BOTH the
    Command destination and the static-edge destination run in the same
    super-step, and writing the same LastValue channel raises.
    """
    with pytest.raises(InvalidUpdateError, match="Can receive only one value per step"):
        build_command_static_conflict_graph().invoke({"messages": []})


def test_command_does_not_suppress_static_edge_silently():
    """With non-colliding channels the same mistake runs BOTH destinations."""
    recorder: list[str] = []
    final = build_command_static_silent_graph(recorder).invoke({"a": [], "b": []})
    # tools was reached via Command(goto=...), fallback via the static edge:
    # goto is ADDITIVE, it does not replace declared edges
    assert set(recorder) == {"agent", "tools", "fallback"}
    assert final["b"] == ["tools-wrote-b"]  # Command destination ran
    assert final["a"] == ["fallback-wrote-a-too"]  # static destination ran too
