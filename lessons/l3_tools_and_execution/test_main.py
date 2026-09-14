"""Tests for L3 - Tools and Execution Boundaries.

Every assertion targets ACTUAL behaviour: side-effect counters on the KVStore,
rejection-log contents, sandbox spawn counts and real files, and a real local
MCP stdio round-trip (no skips, no mocked framework internals)."""

import ast
import asyncio
import sys
import tempfile
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lessons.l3_tools_and_execution.main import (  # noqa: E402
    ALLOW,
    FORBID,
    NEEDS_APPROVAL,
    KVStore,
    RetryPolicy,
    Sandbox,
    ToolExecutor,
    build_gated_loop,
    default_specs,
    make_boom_spec,
    make_flaky_read_spec,
    make_flaky_write_spec,
    make_py_exec_spec,
    make_slow_read_spec,
    make_slow_write_spec,
)
from lessons.l3_tools_and_execution.mcp_server import unit_convert  # noqa: E402

LESSON_DIR = Path(__file__).resolve().parent

GATE_PERMISSIONS = {"kv_get": ALLOW, "unit_convert": ALLOW,
                    "kv_set": NEEDS_APPROVAL, "delete_all": FORBID}


def gated_executor(store: KVStore) -> ToolExecutor:
    specs = {k: v for k, v in default_specs(store).items() if k in GATE_PERMISSIONS}
    return ToolExecutor(specs, permissions=GATE_PERMISSIONS)


@pytest.mark.asyncio
async def test_invalid_args_return_envelope_not_exception():
    """Out-of-contract args come back as a structured envelope; the run continues."""
    executor = ToolExecutor(default_specs(KVStore()))
    envelope = await executor.call(
        "unit_convert", {"value": 5.0, "from_unit": "m", "to_unit": "mile"}
    )
    assert envelope["ok"] is False
    assert envelope["error"]["code"] == "invalid_args"
    assert "to_unit" in envelope["error"]["message"]

    # the failed call did NOT stop the harness: the next call runs normally
    follow_up = await executor.call(
        "unit_convert", {"value": 5.0, "from_unit": "m", "to_unit": "ft"}
    )
    assert follow_up["ok"] is True
    assert follow_up["value"]["unit"] == "ft"


@pytest.mark.asyncio
async def test_raising_tool_aborts_the_run():
    """A non-ToolFailure exception is a bug: it propagates and later steps never run."""
    store = KVStore({"hits": 1})
    executor = ToolExecutor(default_specs(store))
    executor.specs["boom"] = make_boom_spec()

    later_ran = False
    with pytest.raises(RuntimeError, match="bug inside tool code"):
        for name, args in [
            ("kv_get", {"key": "hits"}),
            ("boom", {}),
            ("kv_set", {"key": "hits", "value": 7}),
        ]:
            await executor.call(name, args)
            if name == "boom":
                later_ran = True  # unreachable: boom raises before returning

    assert later_ran is False
    assert store.writes == 0  # kv_set never executed
    assert executor.log[-1].outcome == "raised"  # logged, then re-raised


@pytest.mark.asyncio
async def test_reads_run_concurrently():
    """Two slow reads overlap: wall time is one delay, not the sum of both."""
    store = KVStore({"hits": 0})
    executor = ToolExecutor(
        default_specs(store) | {"slow_read": make_slow_read_spec(store, delay_s=0.3)}
    )
    started = time.perf_counter()
    results = await executor.call_many(
        [{"name": "slow_read", "args": {"key": "hits"}}] * 2
    )
    elapsed = time.perf_counter() - started

    assert all(r["ok"] for r in results)
    assert store.reads == 2
    assert elapsed < 0.5, "two 0.3s reads must overlap (sequential would be ~0.6s)"


@pytest.mark.asyncio
async def test_conflicting_writes_lost_without_lock_controlled_with_it():
    """Same two concurrent increments: lost update without the lock, serialized with it."""
    # WITHOUT the executor's write lock: both reads interleave with both writes.
    store_unlocked = KVStore({"hits": 0})
    unlocked = ToolExecutor(default_specs(store_unlocked), serialize_writes=False)
    await unlocked.call_many([{"name": "kv_add", "args": {"key": "hits", "delta": 1}}] * 2)
    assert store_unlocked.data["hits"] == 1, "one increment vanished: the lost update"
    assert store_unlocked.writes == 2, "both side effects happened anyway"

    # WITH the lock: the whole read-modify-write is serialized per write tool.
    store_locked = KVStore({"hits": 0})
    locked = ToolExecutor(default_specs(store_locked), serialize_writes=True)
    await locked.call_many([{"name": "kv_add", "args": {"key": "hits", "delta": 1}}] * 2)
    assert store_locked.data["hits"] == 2
    assert store_locked.writes == 2
    # note: identical write COUNTS, different final state -- counting side
    # effects alone is not enough to prove a run was conflict-free.


@pytest.mark.asyncio
async def test_timeout_returns_structured_error_and_write_never_lands():
    """A hanging write under a timeout yields a timeout envelope and zero effects."""
    store = KVStore()
    executor = ToolExecutor(
        default_specs(store) | {"slow_write": make_slow_write_spec(store, hang_s=5.0)}
    )
    envelope = await executor.call(
        "slow_write", {"key": "hits", "value": 42}, timeout_s=0.2
    )
    assert envelope["ok"] is False
    assert envelope["error"]["code"] == "timeout"
    assert store.writes == 0, "the hung write must not land"
    assert "hits" not in store.data
    assert executor.log[-1].outcome == "timeout"


@pytest.mark.asyncio
async def test_retry_policy_retries_readonly_only():
    """flaky_read is retried to success; flaky_write is never retried and lands once."""
    store = KVStore({"hits": 1})
    executor = ToolExecutor(
        default_specs(store)
        | {
            "flaky_read": make_flaky_read_spec(store, fail_times=2),
            "flaky_write": make_flaky_write_spec(store),
        },
        retry_policy=RetryPolicy(max_attempts=3),
    )

    envelope = await executor.call("flaky_read", {"key": "hits"})
    assert envelope["ok"] is True and envelope["value"] == 1
    assert executor.log[-1].attempts == 3, "readonly: initial + 2 retries"

    writes_before = store.writes
    envelope = await executor.call("flaky_write", {"key": "hits", "value": 5})
    assert envelope["ok"] is False
    assert envelope["error"]["code"] == "tool_failure"
    assert executor.log[-1].attempts == 1, "writes are never retried"
    assert store.writes - writes_before == 1, "the write landed exactly once"


@pytest.mark.asyncio
async def test_cancelled_hung_write_has_no_side_effects():
    """Task cancellation propagates; the hung write never lands; the log says cancelled."""
    store = KVStore()
    executor = ToolExecutor(
        default_specs(store) | {"slow_write": make_slow_write_spec(store, hang_s=5.0)}
    )
    task = asyncio.create_task(executor.call("slow_write", {"key": "hits", "value": 9}))
    await asyncio.sleep(0.15)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert store.writes == 0
    assert executor.log[-1].outcome == "cancelled"


@pytest.mark.asyncio
async def test_gate_rejects_unauthorized_at_execution_layer():
    """forbid/needs-approval calls are rejected before execution, with reasons logged."""
    store = KVStore({"hits": 3})
    executor = gated_executor(store)

    denied_write = await executor.call("kv_set", {"key": "hits", "value": 99})
    assert denied_write["ok"] is False and denied_write["error"]["code"] == "denied"
    denied_wipe = await executor.call("delete_all", {})
    assert denied_wipe["ok"] is False and denied_wipe["error"]["code"] == "denied"

    # zero side effects: the tool functions were never entered
    assert store.snapshot() == {"hits": 3}
    assert store.writes == 0
    # the rejection log records tool, args and reason for audit
    assert [(r["tool"], r["reason"]) for r in executor.rejections] == [
        ("kv_set", "needs_approval"),
        ("delete_all", "forbidden"),
    ]

    # approval flips the decision for needs-approval tools (L4 makes this an interrupt)
    approved = await executor.call("kv_set", {"key": "hits", "value": 99}, approved=True)
    assert approved["ok"] is True
    assert store.data["hits"] == 99
    assert store.writes == 1


@pytest.mark.asyncio
async def test_gated_loop_feeds_rejections_back_as_observations():
    """A LangGraph tool loop through the gate: denials return as observations."""
    store = KVStore({"hits": 3})
    executor = gated_executor(store)
    graph = build_gated_loop(executor, store)

    final = await graph.ainvoke(
        {"round": 0, "pending": [], "observations": [], "final": ""}
    )

    assert len(final["observations"]) == 3, "one observation per proposed call"
    denied = [o for o in final["observations"] if not o["ok"]]
    assert {o["tool"] for o in denied} == {"kv_set", "delete_all"}
    assert store.snapshot() == {"hits": 3}, "denied calls left the world untouched"
    assert store.reads == 1, "exactly the allowed kv_get executed"
    assert final["final"], "the model finished after seeing the denials"


@pytest.mark.asyncio
async def test_sandbox_rejects_out_of_bounds_access(tmp_path):
    """Absolute paths and traversal are refused pre-exec: no spawn, no escaped file."""
    sandbox = Sandbox(tmp_path / "box")
    executor = ToolExecutor({"py_exec": make_py_exec_spec(sandbox)})
    escaped_target = (tmp_path / "box" / ".." / ".." / "l3_probe_escaped.txt").resolve()

    envelope = await executor.call("py_exec", {"code": "open('/etc/hosts').read()"})
    assert envelope["value"]["executed"] is False
    assert "absolute path" in envelope["value"]["violation"]

    envelope = await executor.call(
        "py_exec", {"code": "open('../../l3_probe_escaped.txt','w').write('x')"}
    )
    assert envelope["value"]["executed"] is False
    assert "traversal" in envelope["value"]["violation"]

    assert sandbox.spawns == 0, "the guard fired BEFORE any subprocess started"
    assert not escaped_target.exists(), "nothing may appear outside the sandbox"


@pytest.mark.asyncio
async def test_sandbox_pins_cwd_and_whitelists_env(tmp_path):
    """The subprocess sees only the pinned cwd and the whitelisted env variables."""
    sandbox = Sandbox(tmp_path / "box")
    executor = ToolExecutor({"py_exec": make_py_exec_spec(sandbox)})

    envelope = await executor.call(
        "py_exec", {"code": "open('note.txt','w').write('inside')\nprint('ok')"}
    )
    assert envelope["value"]["executed"] is True
    assert envelope["value"]["returncode"] == 0
    assert (tmp_path / "box" / "note.txt").exists()

    envelope = await executor.call(
        "py_exec", {"code": "import os\nprint(os.getcwd())\nprint(sorted(os.environ.keys()))"}
    )
    cwd_line, env_line = envelope["value"]["stdout"].splitlines()
    assert cwd_line == str(sandbox.root), "cwd must be pinned to the sandbox dir"
    env_keys = ast.literal_eval(env_line)
    # exactly the whitelist (macOS injects __CF_USER_TEXT_ENCODING behind our back)
    assert set(env_keys) <= {"LANG", "PYTHONIOENCODING", "__CF_USER_TEXT_ENCODING"}
    assert "HOME" not in env_keys, "the parent's env must not leak into the sandbox"


@pytest.mark.asyncio
async def test_mcp_direct_and_server_agree_with_measurable_overhead():
    """unit_convert returns equal results direct and via local MCP; MCP costs more per call."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    cases = [
        {"value": 3.0, "from_unit": "km", "to_unit": "m"},
        {"value": 12.0, "from_unit": "in", "to_unit": "cm"},
        {"value": 0.5, "from_unit": "ft", "to_unit": "mm"},
        {"value": 42.0, "from_unit": "cm", "to_unit": "m"},
        {"value": 1.0, "from_unit": "km", "to_unit": "ft"},
    ]

    started = time.perf_counter()
    direct_results = [unit_convert(**case) for case in cases]
    direct_ms = (time.perf_counter() - started) * 1000

    params = StdioServerParameters(
        command=sys.executable, args=[str(LESSON_DIR / "mcp_server.py")]
    )
    server_stderr = tempfile.TemporaryFile(mode="w+")  # needs a real fileno
    try:
        async with stdio_client(params, errlog=server_stderr) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                listed = (await session.list_tools()).tools
                tool = next(t for t in listed if t.name == "unit_convert")
                assert tool.annotations.read_only_hint is True, (
                    "the read-only hint must travel through the protocol"
                )

                started = time.perf_counter()
                mcp_results = []
                for case in cases:
                    result = await session.call_tool("unit_convert", case)
                    assert result.is_error is False
                    mcp_results.append(result.structured_content)
                mcp_ms = (time.perf_counter() - started) * 1000
    finally:
        server_stderr.close()

    assert mcp_results == direct_results, "same tool, same args: same values"
    per_direct_ms = direct_ms / len(cases)
    per_mcp_ms = mcp_ms / len(cases)
    assert per_mcp_ms > per_direct_ms, "the protocol boundary has measurable cost"
    assert per_mcp_ms > 0.05, "MCP per-call cost is in milliseconds, not microseconds"
