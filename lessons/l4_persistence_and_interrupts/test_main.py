"""Tests for L4 - Persistence, Interrupts, and Reliable Execution.

Every assertion is backed by DIRECT counting: external log-file lines, per-node
execution records, or checkpoint history entries -- never inferred from final
state alone. The subprocess tests kill a real child process (SIGKILL) and
resume in a NEW process, using the ready-file handshake for robustness.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from langgraph.types import Command

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lessons.l4_persistence_and_interrupts.main import (
    build_accumulator_graph,
    build_durability_graph,
    build_multi_interrupt_graph,
    build_review_graph,
    build_static_breakpoint_graph,
    count_node,
    count_writes,
    crash_scenario,
    spec_store_lines,
)
from lib import build_checkpointer, thread

INIT = {"log": [], "value": 2.0, "decision": "", "status": ""}


# ---------------------------------------------------------------------------
# checkpointer / thread (DEMO 1)
# ---------------------------------------------------------------------------


def test_same_thread_continues_new_thread_fresh(tmp_path):
    """Same thread_id accumulates across invokes; a new thread_id starts fresh."""
    graph = build_accumulator_graph(checkpointer=build_checkpointer("memory"))
    t1, t2 = thread("t1"), thread("t2")
    graph.invoke({"log": [], "n": 1}, t1)
    graph.invoke({"log": [], "n": 2}, t1)
    graph.invoke({"log": [], "n": 3}, t2)
    assert graph.get_state(t1).values["log"] == ["n=1", "n=2"]
    assert graph.get_state(t2).values["log"] == ["n=3"]

    # persistence lives in the FILE: a brand-new graph + saver object sees it
    db = tmp_path / "checkpoints.sqlite"
    sqlite_graph = build_accumulator_graph(checkpointer=build_checkpointer("sqlite", str(db)))
    sqlite_graph.invoke({"log": [], "n": 1}, thread("s1"))
    sqlite_graph.invoke({"log": [], "n": 2}, thread("s1"))
    reopened = build_accumulator_graph(checkpointer=build_checkpointer("sqlite", str(db)))
    assert reopened.get_state(thread("s1")).values["log"] == ["n=1", "n=2"]
    history = list(reopened.get_state_history(thread("s1")))
    # 3 checkpoints per invoke (input + __start__ + the node's super-step), 2 invokes
    assert sorted((snap.metadata or {}).get("step") for snap in history) == [-1, 0, 1, 2, 3, 4]


# ---------------------------------------------------------------------------
# durability (DEMO 2)
# ---------------------------------------------------------------------------


def _fail_mid_run(durability):
    graph = build_durability_graph(
        checkpointer=build_checkpointer("sqlite", ":memory:"), fail_once=[True]
    )
    cfg = thread(f"dur-{durability}")
    with pytest.raises(RuntimeError, match="boom exploded"):
        graph.invoke(INIT, cfg, durability=durability)
    return graph, cfg


def test_durability_levels_leave_different_checkpoint_trails():
    """After a mid-run exception: sync/async keep per-step checkpoints (3), exit keeps exactly 1."""
    for durability, expected in (("sync", 3), ("async", 3), ("exit", 1)):
        graph, cfg = _fail_mid_run(durability)
        history = list(graph.get_state_history(cfg))
        steps = sorted((snap.metadata or {}).get("step") for snap in history)
        assert len(history) == expected, f"durability={durability}: {steps}"
        assert graph.get_state(cfg).next == ("boom",)  # resumable in all three cases


def test_resume_after_exception_re_runs_only_failed_node():
    """A crashed run resumes with invoke(None); the failed node re-runs and completes."""
    graph, cfg = _fail_mid_run("sync")
    result = graph.invoke(None, cfg)
    assert result["log"] == ["a", "boom", "c"]
    assert graph.get_state(cfg).next == ()


# ---------------------------------------------------------------------------
# dynamic interrupt vs static breakpoint (DEMO 3)
# ---------------------------------------------------------------------------


def test_dynamic_interrupt_pauses_and_resumes_via_command():
    """interrupt() pauses inside the node; only Command(resume=...) continues it."""
    graph = build_review_graph(checkpointer=build_checkpointer("memory"))
    cfg = thread("dyn")
    list(graph.stream(INIT, cfg, stream_mode=["updates"]))
    snapshot = graph.get_state(cfg)
    assert snapshot.next == ("review",)
    assert [i.value["proposal"] for t in snapshot.tasks for i in t.interrupts] == [2.0]

    # plain None input does NOT resume a dynamic interrupt (probed on 1.2.11)
    graph.invoke(None, cfg)
    assert graph.get_state(cfg).next == ("review",)

    result = graph.invoke(Command(resume={"decision": "accept", "value": 2.0}), cfg)
    assert result["status"] == "written"
    assert graph.get_state(cfg).next == ()


def test_static_breakpoint_resumes_with_none_input():
    """compile(interrupt_before=[...]) pauses with EMPTY task interrupts; plain None resumes."""
    graph = build_static_breakpoint_graph(checkpointer=build_checkpointer("memory"))
    cfg = thread("static")
    chunks = list(graph.stream(INIT, cfg, stream_mode=["updates"]))
    assert any(chunk.get("__interrupt__") == () for _mode, chunk in chunks)  # empty tuple marker
    snapshot = graph.get_state(cfg)
    assert snapshot.next == ("apply",)
    assert all(not t.interrupts for t in snapshot.tasks)  # no dynamic payload

    result = graph.invoke(None, cfg)
    assert result["status"] == "written"

    # probed: a static pause also accepts Command(resume=True)
    cfg2 = thread("static-command")
    list(graph.stream(INIT, cfg2, stream_mode=["updates"]))
    assert graph.invoke(Command(resume=True), cfg2)["status"] == "written"


# ---------------------------------------------------------------------------
# accept / edit / reject review paths (DEMO 4)
# ---------------------------------------------------------------------------


def test_accept_edit_reject_three_paths(tmp_path):
    """The three review paths end in three different states; reject skips the write node."""
    log_path = tmp_path / "spec_store.log"
    graph = build_review_graph(checkpointer=build_checkpointer("memory"), log_path=log_path)
    answers = {
        "accept": {"decision": "accept", "value": 2.0},
        "edit": {"decision": "edit", "value": 1.5},
        "reject": {"decision": "reject", "value": 2.0},
    }
    finals = {}
    for name, answer in answers.items():
        cfg = thread(f"review-{name}")
        list(graph.stream(INIT, cfg, stream_mode=["updates"]))
        finals[name] = graph.invoke(Command(resume=answer), cfg)

    assert finals["accept"]["status"] == "written" and finals["accept"]["value"] == 2.0
    assert finals["edit"]["status"] == "written" and finals["edit"]["value"] == 1.5
    assert finals["reject"]["status"] == "rejected"
    assert "apply: wrote" not in finals["reject"]["log"]  # reject never entered apply
    # direct counting of the external store: accept + edit wrote, reject did not
    assert count_writes(spec_store_lines(log_path)) == 2
    assert "WRITE copper=1.5" in spec_store_lines(log_path)


# ---------------------------------------------------------------------------
# multiple interrupts (DEMO 5)
# ---------------------------------------------------------------------------


def test_multiple_interrupts_stable_order():
    """Two nodes host interrupts and one node interrupts twice across its loop; order is stable."""
    graph = build_multi_interrupt_graph(checkpointer=build_checkpointer("memory"))
    cfg = thread("multi")
    order = []
    answers = ["FR-4", "yes", "yes"]
    inp = {"log": [], "i": 0}
    while True:
        chunks = list(graph.stream(inp, cfg, stream_mode=["updates"]))
        for mode, chunk in chunks:
            if isinstance(chunk, dict) and "__interrupt__" in chunk:
                order.extend(i.value["ask"] for i in chunk["__interrupt__"])
        if not graph.get_state(cfg).next:
            break
        inp = Command(resume=answers[len(order) - 1])
    assert order == ["material", "confirm item 0", "confirm item 1"]
    final = graph.get_state(cfg).values
    assert final["log"] == ["material=FR-4", "item0=yes", "item1=yes", "done"]
    assert final["i"] == 2


# ---------------------------------------------------------------------------
# subprocess crash + resume (DEMOS 6-7) -- real SIGKILL, new process resume
# ---------------------------------------------------------------------------


def test_crash_before_write_recovers_with_single_write():
    """SIGKILL before the external write -> resume re-runs apply -> the write lands exactly once."""
    result = crash_scenario("before_write")
    assert result["reached"] and result["rc"] == -9
    assert count_writes(result["pre_lines"]) == 0
    assert "next=('apply',)" in result["resume_stdout"]
    assert count_writes(result["final_lines"]) == 1


def test_crash_after_write_duplicates_external_effect():
    """SIGKILL after the write but before its checkpoint -> resume re-runs apply -> TWO writes."""
    result = crash_scenario("after_write")
    assert count_writes(result["pre_lines"]) == 1  # the write landed before the kill
    assert "next=('apply',)" in result["resume_stdout"]  # checkpoint says apply never finished
    assert count_writes(result["final_lines"]) == 2  # checkpoint != exactly-once for effects
    assert result["final_lines"].count("node=apply") == 2


def test_crash_after_checkpoint_replays_only_unfinished_work():
    """SIGKILL after apply's checkpoint -> resume starts at report; apply never re-runs."""
    result = crash_scenario("after_checkpoint")
    assert "next=('report',)" in result["resume_stdout"]
    pre, final = result["pre_lines"], result["final_lines"]
    assert count_node(pre, "propose") == count_node(final, "propose") == 1
    assert count_node(pre, "apply") == count_node(final, "apply") == 1  # not re-executed
    assert count_node(pre, "report") == 0 and count_node(final, "report") == 1  # replay scope
    assert count_writes(final) == 1


def test_ledger_makes_replay_idempotent():
    """A thread+task_id ledger lets the re-executed node skip the write: effect lands once."""
    result = crash_scenario("after_write", ledger=True)
    # the node still re-ran (execution is replayed), but the EFFECT did not duplicate
    assert result["final_lines"].count("node=apply") == 2
    assert count_writes(result["final_lines"]) == 1
    assert any(line.startswith("SKIP task=") for line in result["final_lines"])


def test_durability_exit_sigkill_leaves_no_checkpoint():
    """durability='exit' + SIGKILL: no checkpoint is ever written; the write is stranded."""
    result = crash_scenario("after_write", durability="exit")
    assert count_writes(result["pre_lines"]) == 1  # the external write did land
    assert "NO-CHECKPOINT" in result["resume_stdout"]
    assert "HISTORY n=0" in result["resume_stdout"]
    assert count_writes(result["final_lines"]) == 1  # nothing re-ran: nothing to resume from


# ---------------------------------------------------------------------------
# history walk, fork, external undo (DEMO 8)
# ---------------------------------------------------------------------------


def test_fork_diverges_but_external_undo_is_separate(tmp_path):
    """Forking branches STATE only; external writes accumulate until WE compensate."""
    log_path = tmp_path / "spec_store.log"
    graph = build_review_graph(checkpointer=build_checkpointer("memory"), log_path=log_path)
    cfg = thread("main")
    list(graph.stream(INIT, cfg, stream_mode=["updates"]))
    graph.invoke(Command(resume={"decision": "accept", "value": 2.0}), cfg)
    accept_result = graph.get_state(cfg).values
    assert spec_store_lines(log_path) == ["WRITE copper=2.0"]

    history = list(graph.get_state_history(cfg))
    paused = next(h for h in history if h.tasks and any(t.interrupts for t in h.tasks))
    fork_cfg = graph.update_state(paused.config, {"decision": "edit", "value": 1.5}, as_node="review")
    fork_result = graph.invoke(None, fork_cfg)
    assert fork_result["value"] == 1.5
    assert spec_store_lines(log_path) == ["WRITE copper=2.0", "WRITE copper=1.5"]  # no rollback
    assert graph.get_state(cfg).values == fork_result  # probed: the head moved to the fork

    # probed: new thread_id + old checkpoint_id does NOT fork on 1.2.11 -- it
    # silently starts from an EMPTY state (and still runs the write node!)
    raw = {"configurable": {"thread_id": "raw", "checkpoint_id": paused.config["configurable"]["checkpoint_id"]}}
    assert graph.get_state(raw).values == {}
    raw_cfg = graph.update_state(raw, {"decision": "edit", "value": 9.9}, as_node="review")
    raw_result = graph.invoke(None, raw_cfg)
    assert raw_result["log"] == ["apply: wrote copper=9.9"]  # fresh start: propose is gone
    assert "WRITE copper=9.9" in spec_store_lines(log_path)

    # restoring the head replays the original branch (with its cached answer)
    list(graph.stream(Command(resume={"decision": "accept", "value": 2.0}), paused.config, stream_mode=["updates"]))
    assert graph.get_state(cfg).values == accept_result
    lines = spec_store_lines(log_path)
    assert lines.count("WRITE copper=2.0") == 2 and len(lines) == 4  # replay duplicated the write

    # compensation is our own code: keep exactly the intended write
    kept = [line for i, line in enumerate(lines) if line == "WRITE copper=2.0" and i == 0]
    log_path.write_text("".join(line + "\n" for line in kept))
    assert spec_store_lines(log_path) == ["WRITE copper=2.0"]
