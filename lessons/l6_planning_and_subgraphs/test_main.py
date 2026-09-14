"""Tests for L6 - Planning, Subgraphs and Multi-Agent.

Every test counts executions DIRECTLY via the model-call journal
(MODEL_JOURNAL records the exact context each fake model was handed) and
TOOL_JOURNAL, or reads checkpoint metadata -- nothing is inferred from final
state alone, and no framework scheduler internals are mocked.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from langgraph.errors import InvalidUpdateError
from langgraph.types import Command

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lessons.l6_planning_and_subgraphs.main import (  # noqa: E402
    HISTORY_MARKER,
    MODEL_JOURNAL,
    TOOL_JOURNAL,
    base_task,
    build_delegation_graph,
    build_fixed_pipeline_graph,
    build_map_reduce_graph,
    build_ns_demo_graph,
    build_parent_command_graph,
    build_plan_execute_graph,
    build_shared_subgraph_graph,
    build_translated_subgraph_graph,
    clear_journals,
    discover_namespaces,
    expected_report,
    final_step,
    run_delegation,
    run_plan_execute,
    run_single_agent,
    superstep_segments,
    tricky_task,
    worker_contexts,
)
from lib import build_checkpointer, thread  # noqa: E402


def journal_nodes() -> list[str]:
    return [r["node"] for r in MODEL_JOURNAL]


def test_single_agent_processes_all_chunks():
    """The single-agent loop consumes every chunk in ONE context, then merges."""
    clear_journals()
    state, graph = run_single_agent(base_task())
    assert state["stop_reason"] == "success"
    assert len([r for r in MODEL_JOURNAL if r["node"] == "single:model"]) == 5  # 4 tool turns + 1 merge
    assert len(TOOL_JOURNAL) == 4  # every chunk summarized exactly once
    assert state["final_report"] == expected_report(base_task())
    # context duplication, measured: the same history is re-sent every turn
    ctx_sizes = [len(r["context"]) for r in MODEL_JOURNAL]
    assert ctx_sizes == [1, 3, 5, 7, 9]
    assert sum(ctx_sizes) == 25


def test_fixed_pipeline_is_parallel_and_cheapest():
    """Compile-time fan-out: 3 super-steps, 5 isolated model calls, and it is the cheapest structure on a fixed task."""
    clear_journals()
    cp = build_checkpointer("memory")
    cfg = thread("l6-fixed")
    chunks = base_task()
    graph = build_fixed_pipeline_graph(chunks=chunks, checkpointer=cp)
    state = graph.invoke({"chunks": [], "results": [], "report": ""}, cfg)
    calls = list(MODEL_JOURNAL)  # snapshot BEFORE the second (segment-measuring) invoke

    # node completion order inside a super-step is thread-scheduling dependent,
    # so compare sorted names per segment -- the SEGMENTATION is what is asserted
    segments = superstep_segments(graph, {"chunks": [], "results": [], "report": ""}, thread("l6-fixed-b"))
    assert [sorted(seg) for seg in segments] == [
        ["extract"],
        ["summary_0", "summary_1", "summary_2", "summary_3"],  # ONE super-step
        ["merge"],
    ]
    assert final_step(graph, cfg) == 3
    assert len(calls) == 5  # 4 summaries + 1 merge, zero coordination calls
    for record in calls:
        if record["kind"] == "summarize":
            assert len(record["context"]) == 1  # each call saw ONLY its chunk
    assert state["report"] == expected_report(chunks)


def test_plan_execute_parity_and_coordination_cost():
    """Plan-and-Execute produces the identical report but pays 1 planner call and sequential super-steps."""
    clear_journals()
    state_a, _ = run_single_agent(base_task())
    clear_journals()
    chunks = base_task()
    cp = build_checkpointer("memory")
    cfg = thread("l6-planexec")
    state_c, graph_c = run_plan_execute(chunks, checkpointer=cp, config=cfg)
    assert state_c["report"] == state_a["final_report"]  # parity with the single agent
    assert len(MODEL_JOURNAL) == 6  # planner + 4 workers + report
    assert any(r["kind"] == "plan" for r in MODEL_JOURNAL)
    assert final_step(graph_c, cfg) == 6  # strictly sequential executor loop

    # 协作何时无收益，量化: on the fixed task the fixed pipeline is cheaper on
    # every axis we can count.
    clear_journals()
    cp_b = build_checkpointer("memory")
    cfg_b = thread("l6-planexec-b")
    graph_b = build_fixed_pipeline_graph(chunks=chunks, checkpointer=cp_b)
    graph_b.invoke({"chunks": [], "results": [], "report": ""}, cfg_b)
    calls_b = len(MODEL_JOURNAL)
    steps_b = final_step(graph_b, cfg_b)
    assert calls_b == 5 < 6
    assert steps_b == 3 < 6


def test_tricky_task_fixed_pipeline_stuck_plan_execute_recovers():
    """When a step needs an unplanned preprocessing action, only the plan-based structure recovers."""
    tricky = tricky_task()
    clear_journals()
    state_b = build_fixed_pipeline_graph(chunks=tricky).invoke({"chunks": [], "results": [], "report": ""})
    assert "3/4" in state_b["report"] and "不完整" in state_b["report"]  # stuck, no node exists to recover

    clear_journals()
    state_c, _ = run_plan_execute(tricky)
    assert state_c["replans"] == 1
    assert "4/4" in state_c["report"] and "完整" in state_c["report"]
    assert any(r["kind"] == "replan" for r in MODEL_JOURNAL)


def test_send_every_task_once_in_one_super_step():
    """N runtime Sends: each task executes EXACTLY once and all N land in ONE super-step."""
    chunks = [
        {"chunk_id": "etching-A1", "format": "text", "section": "蚀刻", "n_params": 6},
        {"chunk_id": "drilling-A2", "format": "text", "section": "钻孔", "n_params": 4},
        {"chunk_id": "plating-B1", "format": "text", "section": "电镀", "n_params": 5},
        {"chunk_id": "testing-B2", "format": "text", "section": "测试", "n_params": 7},
        {"chunk_id": "solder-C1", "format": "text", "section": "阻焊", "n_params": 3},
    ]
    clear_journals()
    cp = build_checkpointer("memory")
    cfg = thread("l6-send")
    graph = build_map_reduce_graph(checkpointer=cp)
    state = graph.invoke({"chunks": chunks, "results": [], "report": ""}, cfg)

    worker_calls = [r for r in MODEL_JOURNAL if r["node"].startswith("worker:")]
    per_task: dict[str, int] = {}
    for r in worker_calls:
        cid = r["context"][0]["task"]["chunk"]["chunk_id"]
        per_task[cid] = per_task.get(cid, 0) + 1
    assert len(per_task) == 5 and all(v == 1 for v in per_task.values())  # exactly once each

    segments = superstep_segments(graph, {"chunks": chunks, "results": [], "report": ""}, thread("l6-send-b"))
    assert [len(seg) for seg in segments] == [1, 5, 1]  # all 5 workers in ONE super-step
    assert final_step(graph, cfg) == 3
    assert len(state["results"]) == 5  # the reducer channel merged all N; join saw them all


def test_send_without_reducer_raises_invalid_update():
    """Removing the Annotated reducer on the merge channel: N Send writes -> InvalidUpdateError."""
    graph = build_map_reduce_graph(with_reducer=False)
    chunks = base_task()[:3]
    with pytest.raises(InvalidUpdateError, match="Can receive only one value per step"):
        graph.invoke({"chunks": chunks, "results": [], "report": ""})


def test_shared_subgraph_writes_leak_and_duplicate():
    """Shared-schema subgraph: internal writes reach the parent AND its final state is re-applied through the parent reducer."""
    naive = build_shared_subgraph_graph().invoke({"doc_id": "DOC-1", "notes": []})
    assert "sub scratch (meant to be internal)" in naive["notes"]  # no private space: it leaks
    assert naive["notes"].count("parent intake note") == 2  # received values re-added on exit (probed)

    trimmed = build_shared_subgraph_graph(trim=True).invoke({"doc_id": "DOC-1", "notes": []})
    assert trimmed["notes"].count("parent intake note") == 1  # input/output schemas cut both directions
    assert "sub scratch (meant to be internal)" in trimmed["notes"]  # but sharing itself is unchanged


def test_translated_subgraph_keeps_keys_private():
    """Own-schema subgraph + explicit translation: sub keys never appear in parent state."""
    clear_journals()
    state = build_translated_subgraph_graph().invoke({"doc_id": "DOC-2", "notes": []})
    assert "scratch" not in state  # the subgraph's channel does not exist in the parent
    inner = next(r for r in MODEL_JOURNAL if r["node"] == "sub.inner")
    assert list(inner["context"][0].keys()) == ["payload"]  # the model saw ONLY the translated payload
    assert any(n.startswith("translated back:") for n in state["notes"])


def test_command_parent_semantics():
    """Command.PARENT routes the parent, skips the subgraph tail; with a static exit left in place BOTH destinations run."""
    # normal document: a plain update stays inside the subgraph
    normal = build_parent_command_graph().invoke({"doc_id": "NORMAL-9", "trail": [], "verdict": ""})
    assert "inner_tail ran" in normal["trail"]
    assert "normal_next ran (static edge)" in normal["trail"]
    assert "escalate ran" not in normal["trail"]

    # urgent document, no static exit: parent routed, subgraph tail SKIPPED
    clean = build_parent_command_graph(keep_static_exit=False).invoke(
        {"doc_id": "URGENT-01", "trail": [], "verdict": ""}
    )
    assert "inner_tail ran" not in clean["trail"]  # bypassed the normal subgraph exit
    assert "escalate ran (parent node)" in clean["trail"]
    assert "inner_check fired Command.PARENT" in clean["trail"]  # the Command update hit PARENT state
    assert clean["verdict"] == "escalated"

    # urgent document WITH the static exit left in place: goto is ADDITIVE
    additive = build_parent_command_graph(keep_static_exit=True).invoke(
        {"doc_id": "URGENT-01", "trail": [], "verdict": ""}
    )
    assert "escalate ran (parent node)" in additive["trail"]
    assert "normal_next ran (static edge)" in additive["trail"]  # both ran: silent double-scheduling


def test_subgraph_state_via_checkpoint_namespace():
    """A finished subgraph's internal state is NOT under get_state(subgraphs=True) -- it lives under its checkpoint namespace."""
    cp = build_checkpointer("memory")
    cfg = thread("l6-ns")
    graph = build_ns_demo_graph(checkpointer=cp)
    graph.invoke({"doc_id": "DOC-5", "notes": []}, cfg)

    snap = graph.get_state(cfg, subgraphs=True)
    assert snap.tasks == ()  # completed run: no nested state surfaced here (probed on 1.2.11)

    ns_cfg = thread("l6-ns-b")
    ns_list = discover_namespaces(graph, {"doc_id": "DOC-5b", "notes": []}, ns_cfg)
    assert len(ns_list) == 1 and ns_list[0].startswith("worker:")  # ns = "worker:<uuid>"

    child_cfg = {"configurable": {"thread_id": "l6-ns-b", "checkpoint_ns": ns_list[0]}}
    child = graph.get_state(child_cfg)
    assert child.values["inner_log"] == ["step one", "step two"]  # sub-private state, inspectable after the run
    assert child.metadata["step"] == 2  # the subgraph has its OWN step counter
    child_history = list(graph.get_state_history(child_cfg))
    assert [h.metadata["step"] for h in child_history] == [2, 1, 0, -1]  # its OWN history

    # pending-task path: while a subgraph task is PENDING, subgraphs=True DOES
    # expose the nested snapshot (interrupt used only as a visibility tool)
    cfg2 = thread("l6-ns-int")
    graph2 = build_ns_demo_graph(checkpointer=build_checkpointer("memory"), interrupt_inside=True)
    graph2.invoke({"doc_id": "DOC-6", "notes": []}, cfg2)
    snap2 = graph2.get_state(cfg2, subgraphs=True)
    assert snap2.next == ("worker",)
    nested = snap2.tasks[0].state
    assert nested.values["doc_id"] == "DOC-6"
    assert nested.next == ("ns_step_one",)
    payload = [i.value for nt in nested.tasks for i in (nt.interrupts or ())]
    assert payload == [{"approval_needed": "DOC-6"}]
    resumed = graph2.invoke(Command(resume="approved"), cfg2)
    assert any("ns step one resumed (approved)" in n for n in resumed["notes"])


def test_delegation_isolation_and_sibling_hint():
    """Workers see ONLY the delegated payload; the anti-pattern leaks history; a sibling-dependent task fails without a hint."""
    clear_journals()
    isolated = run_delegation()
    assert not any(HISTORY_MARKER in str(r["context"]) for r in worker_contexts())  # never saw parent messages
    outcomes = {o["worker"]: o for o in isolated["outcomes"]}
    assert outcomes["w_a"]["ok"] is True
    assert outcomes["w_b"]["ok"] is False  # cross-check needs sibling context it was not given

    clear_journals()
    with_history = run_delegation(include_history=True)
    assert any(HISTORY_MARKER in str(r["context"]) for r in worker_contexts())  # the leak is countable
    assert len(worker_contexts()) == 2

    clear_journals()
    hinted = run_delegation(include_hint=True)
    assert all(o["ok"] for o in hinted["outcomes"])  # explicit hint restores the missing context


def test_replanning_does_not_reexecute_completed_tasks():
    """After a failed task, the replanner re-plans only the remaining work: completed tasks run exactly once."""
    clear_journals()
    state, _ = run_plan_execute(tricky_task())
    counts: dict[str, int] = {}
    for r in MODEL_JOURNAL:
        if r["node"].startswith("executor:"):
            key = r["node"].split(":", 1)[1]
            counts[key] = counts.get(key, 0) + 1
    assert counts["t1"] == 1 and counts["t3"] == 1 and counts["t4"] == 1  # NOT re-executed
    assert counts["t2"] == 1 and counts["t2-prep"] == 1 and counts["t2-retry"] == 1
    assert state["replans"] == 1
    ok_chunks = sorted(r["chunk_id"] for r in state["results"] if r.get("ok") and r.get("summary"))
    assert ok_chunks == sorted(c["chunk_id"] for c in tricky_task())  # final result complete
    assert "4/4" in state["report"]
