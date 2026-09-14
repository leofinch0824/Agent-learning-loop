"""Live tests for L3 - a real model behind the same PermissionGate.

Both tests skip cleanly when no live endpoint is configured; the README then
records them as 未验证. They call the real endpoint via lib.live_model."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lib import live_client, live_enabled, live_model_name  # noqa: E402

from lessons.l3_tools_and_execution.live_demo import (  # noqa: E402
    TASK,
    build_live_executor,
    run_agent,
)

requires_live = pytest.mark.skipif(
    not live_enabled(), reason="live model not configured (.env absent) -> 未验证"
)


@requires_live
@pytest.mark.asyncio
async def test_live_model_uses_an_allowed_tool():
    """A real model completes a read-only task through unit_convert."""
    executor, store = build_live_executor()
    task = "Convert 3 km to m using unit_convert, then report just the number."
    summary = await run_agent(live_client(), executor, task, model=live_model_name())

    allowed_calls = [r for r in executor.log if r.tool == "unit_convert" and r.outcome == "ok"]
    assert allowed_calls, "the model should have used the allowed unit_convert tool"
    assert "3000" in summary["final_text"], f"expected the converted value in: {summary['final_text']!r}"


@requires_live
@pytest.mark.asyncio
async def test_live_gate_rejects_write_the_model_proposes():
    """The model proposes kv_set/delete_all per the task; the gate denies them,
    the denial is fed back as an observation, and the store stays untouched."""
    executor, store = build_live_executor()
    before = store.snapshot()

    summary = await run_agent(live_client(), executor, TASK, model=live_model_name())

    assert executor.rejections, "expected the model to propose at least one gated tool"
    reasons = {r["reason"] for r in executor.rejections}
    assert reasons <= {"needs_approval", "forbidden"}
    assert any(r["tool"] == "kv_set" and r["reason"] == "needs_approval" for r in executor.rejections)
    # the model did not get away with anything: the world is exactly as before
    assert store.snapshot() == before
    assert store.writes == 0, "no denied write may land"
