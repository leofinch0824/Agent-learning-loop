"""Tests for L0 - Why Agents: foundations.

Every test counts behaviour DIRECTLY: model calls via Metrics and the fake
model's own counter, tool executions via Metrics, message flow via
ScriptedModel.message_log. Nothing is inferred from the final answer alone.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lessons.l0_agent_foundations.main import (  # noqa: E402
    BUDGET_SCRIPT,
    DETERMINISTIC_SINGLE_SHOT_SCRIPT,
    FREE_TEXT_TASK,
    LOG_LINE_TASK,
    NO_PROGRESS_SCRIPT,
    SINGLE_SHOT_SCRIPT,
    WORKFLOW_EXTRACT_SCRIPT,
    ScriptedModel,
    agent_script_for,
    make_tools,
    parse_with_pure_code,
    run_agent_loop,
    run_fixed_workflow,
    run_single_shot,
    score_answer,
)


def test_single_shot_is_exactly_one_model_call():
    """Single shot solves the whole task with exactly 1 model call and 0 tool executions."""
    model = ScriptedModel(SINGLE_SHOT_SCRIPT)
    result = run_single_shot(FREE_TEXT_TASK, model, verbose=False)
    assert result.metrics.model_calls == 1
    assert result.metrics.tool_executions == 0
    assert model.calls == 1  # cross-check: Metrics agrees with the fake's own count


def test_single_shot_quality_depends_on_the_model():
    """Single-shot validation lives inside the model: the scripted unit slip costs 1 of 3 points."""
    result = run_single_shot(FREE_TEXT_TASK, ScriptedModel(SINGLE_SHOT_SCRIPT), verbose=False)
    assert score_answer(result.answer) == (2, 3)
    # the exact failure shape: copper converted to "35 mm" (should be 1 oz),
    # then range-checked against the board-thickness limit
    assert result.answer["copper_weight"]["verdict"] == "fail"
    assert result.answer["copper_weight"]["unit"] == "mm"


def test_fixed_workflow_pays_the_model_once_and_validates_in_code():
    """The workflow calls the model only for extraction; validation is 3 direct tool executions."""
    model = ScriptedModel(WORKFLOW_EXTRACT_SCRIPT)
    result = run_fixed_workflow(FREE_TEXT_TASK, model, verbose=False)
    assert result.metrics.model_calls == 1
    assert result.metrics.tool_executions == 3  # one check_spec per parameter, in code
    assert result.metrics.loop_iterations == 0  # a pipeline, not a loop
    assert score_answer(result.answer) == (3, 3)
    # The one model call was an extraction prompt: it carries the (parse-stage
    # normalised) task text but no validation instructions -- validation never
    # left the code.
    prompt = model.message_log[0][0]["content"]
    assert FREE_TEXT_TASK.replace("；", ";") in prompt  # stage 1 normalises CJK separators
    assert "校验" not in prompt


def test_agent_loop_succeeds_with_feedback_flowing_back():
    """A successful agent run: 6 turns / 5 tool execs, and each observation reaches the next model call."""
    model = ScriptedModel(agent_script_for(FREE_TEXT_TASK))
    result = run_agent_loop(FREE_TEXT_TASK, model, make_tools(FREE_TEXT_TASK), verbose=False)
    assert result.stop_reason == "final_answer"
    assert result.metrics.model_calls == 6
    assert result.metrics.tool_executions == 5
    assert result.metrics.loop_iterations == 6
    assert score_answer(result.answer) == (3, 3)
    # latency accounting: 6 model calls + 5 tool executions, nothing else
    assert result.metrics.latency_units == 6 * 5.0 + 5 * 1.0
    # DIRECT evidence of the observation -> next-action loop: the first call
    # saw no tool messages; the second call saw read_source's observation.
    assert not any(m.get("role") == "tool" for m in model.message_log[0])
    tool_msgs = [m for m in model.message_log[1] if m.get("role") == "tool"]
    assert tool_msgs and tool_msgs[0]["content"] == FREE_TEXT_TASK
    # and the model's extract action embeds that observation: its args carry
    # the text it could only know by having looked (turn 2's tool_calls are
    # visible inside turn 3's context as an assistant message)
    turn3_calls = [
        c
        for m in model.message_log[2]
        if m.get("role") == "assistant" and "tool_calls" in m
        for c in m["tool_calls"]
    ]
    extract_calls = [c for c in turn3_calls if c["name"] == "extract_parameters"]
    assert extract_calls and extract_calls[0]["args"]["text"] == FREE_TEXT_TASK


def test_agent_loop_stops_on_no_progress():
    """Re-issuing the same (tool, args) stops the loop BEFORE executing it a second time."""
    result = run_agent_loop(
        FREE_TEXT_TASK, ScriptedModel(NO_PROGRESS_SCRIPT), make_tools(FREE_TEXT_TASK), verbose=False
    )
    assert result.stop_reason == "no_progress"
    assert result.metrics.model_calls == 3  # the repeat was proposed on turn 3
    assert result.metrics.tool_executions == 2  # and NOT executed again
    assert result.answer is None
    assert score_answer(result.answer) == (0, 3)


def test_agent_loop_stops_on_budget():
    """Ever-new arguments defeat the no-progress guard; only max_steps stops the run."""
    result = run_agent_loop(
        FREE_TEXT_TASK,
        ScriptedModel(BUDGET_SCRIPT),
        make_tools(FREE_TEXT_TASK),
        max_steps=3,
        verbose=False,
    )
    assert result.stop_reason == "budget_exhausted"
    assert result.metrics.model_calls == 3  # == max_steps, not the script length
    assert result.metrics.tool_executions == 3  # every turn still executed one tool
    assert score_answer(result.answer) == (0, 3)


def test_deterministic_task_agent_is_strictly_worse_than_code():
    """On a fixed-grammar task the agent makes strictly more calls for zero quality gain."""
    parser = parse_with_pure_code(LOG_LINE_TASK, verbose=False)
    single = run_single_shot(LOG_LINE_TASK, ScriptedModel(DETERMINISTIC_SINGLE_SHOT_SCRIPT), verbose=False)
    agent = run_agent_loop(
        LOG_LINE_TASK, ScriptedModel(agent_script_for(LOG_LINE_TASK)), make_tools(LOG_LINE_TASK), verbose=False
    )

    assert parser.metrics.model_calls == 0
    assert score_answer(parser.answer) == (3, 3)
    assert score_answer(single.answer) == (3, 3)
    assert score_answer(agent.answer) == (3, 3)
    # equal quality, strictly more model calls and latency -- the counterexample
    assert agent.metrics.model_calls > single.metrics.model_calls > parser.metrics.model_calls
    assert agent.metrics.latency_units > single.metrics.latency_units > parser.metrics.latency_units
