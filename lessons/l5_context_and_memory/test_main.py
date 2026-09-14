"""Tests for L5 - Context, Retrieval and Memory.

Every test asserts on DIRECT evidence: the message lists the fake models
were actually handed (FakeModel.seen / DistractibleModel.seen), the recall
journal, store entry states, or byte/token counts. No framework internals
are mocked. Phenomena first -- each docstring names the behavior under test.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from langgraph.store.memory import InMemoryStore

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lessons.l5_context_and_memory.main import (  # noqa: E402
    DEFAULT_USER,
    FACT_INNER_TRACE,
    NOISE_MEMORIES,
    STABLE_PREFIX,
    AgentState,
    DistractibleModel,
    FakeModel,
    build_injectedstore_pitfall_graph,
    build_long_history,
    build_memory_graph,
    build_tool_graph,
    clear_recall_journal,
    collapse_with_summary,
    common_prefix_len,
    confirm_memory,
    digest_table,
    estimate_tokens,
    fake_summarizer,
    fetch_parameter_table,
    initial_state,
    message_tokens,
    messages_tokens,
    namespace_for,
    put_memory,
    recall_journal,
    recall_memories,
    supersede_memory,
    thread,
    trim_messages,
    wire_payload,
)
from lib import build_checkpointer  # noqa: E402


def system_seen(model) -> str:
    """The full system text of the FIRST message list the model was handed."""
    return "\n".join(m.get("content", "") for m in model.seen[0] if m.get("role") == "system")


def test_new_thread_sees_only_the_store_layer():
    """A new thread sees Store memories; A's working context / task_note / checkpoint do not leak."""
    store = InMemoryStore()
    put_memory(store, DEFAULT_USER, "spec_inner", FACT_INNER_TRACE)
    checkpointer = build_checkpointer("memory")  # shared by both threads
    task_note_a = "inner min trace width = 0.127 mm (session-state layer)"

    fake_a = FakeModel(["A"])
    graph_a = build_memory_graph(fake_a.respond, store=store, checkpointer=checkpointer)
    cfg_a = thread("t-a", user_id=DEFAULT_USER)
    graph_a.invoke(
        initial_state("记住：inner min trace width = 0.127 mm（工作上下文层）", task_note=task_note_a),
        cfg_a,
    )

    clear_recall_journal()
    fake_b = FakeModel(["B"])
    graph_b = build_memory_graph(fake_b.respond, store=store, checkpointer=checkpointer)
    cfg_b = thread("t-b", user_id=DEFAULT_USER)
    graph_b.invoke(initial_state("内层最小线宽是多少？"), cfg_b)

    seen_b = fake_b.seen[0]
    assert [m["role"] for m in seen_b] == ["system", "user"]  # nothing else entered the context
    assert "0.127 mm" in system_seen(fake_b)  # Store layer DID cross the thread boundary
    assert "工作上下文层" not in json.dumps(seen_b, ensure_ascii=False)  # A's user message stayed in A
    # session state: A's task_note never became B's task_note and never entered B's messages
    assert graph_b.get_state(cfg_b).values["task_note"] != task_note_a
    assert "session-state layer" not in json.dumps(seen_b, ensure_ascii=False)
    # checkpoint: A's history exists under its own thread_id, B's holds none of it
    assert len(list(graph_a.get_state_history(cfg_a))) >= 1
    b_blob = json.dumps([s.values for s in graph_b.get_state_history(cfg_b)], ensure_ascii=False)
    assert "工作上下文层" not in b_blob


def test_candidate_memory_not_injected_until_confirmed():
    """Write-back lands as a candidate; a NEW thread sees it only after confirmation."""
    store = InMemoryStore()
    clear_recall_journal()
    fake1 = FakeModel(["好的，以后默认用 mm。"])
    build_memory_graph(
        fake1.respond,
        store=store,
        writeback=("unit_pref", "user prefers answers in mm units"),
    ).invoke(initial_state("以后结果默认用 mm。"), thread("w1", user_id=DEFAULT_USER))

    candidate = store.get(namespace_for(DEFAULT_USER), "unit_pref")
    assert candidate.value["status"] == "candidate"
    assert candidate.value["source"] == "thread:w1"  # provenance travels with the entry

    fake2 = FakeModel(["r2"])
    build_memory_graph(fake2.respond, store=store).invoke(
        initial_state("内层最小线宽是多少？"), thread("w2", user_id=DEFAULT_USER)
    )
    assert "prefers answers in mm" not in system_seen(fake2)  # candidate NOT injected
    assert recall_journal[-1]["skipped"] == [("unit_pref", "status=candidate")]

    confirm_memory(store, DEFAULT_USER, "unit_pref")
    confirmed = store.get(namespace_for(DEFAULT_USER), "unit_pref")
    assert confirmed.value["status"] == "confirmed"
    assert confirmed.value["version"] == 2  # the promotion is a version bump

    fake3 = FakeModel(["r3"])
    build_memory_graph(fake3.respond, store=store).invoke(
        initial_state("内层最小线宽是多少？"), thread("w3", user_id=DEFAULT_USER)
    )
    # The assertion that matters: the string is INSIDE the messages the new
    # thread's model received, not merely inside the store.
    assert "user prefers answers in mm units [v2]" in system_seen(fake3)


def test_conflicting_candidates_coexist_with_sources():
    """Two conflicting candidates are both kept with their sources; neither is injected."""
    store = InMemoryStore()
    put_memory(store, DEFAULT_USER, "via_c1", "user prefers via diameters in mil",
               status="candidate", source="thread:c1")
    put_memory(store, DEFAULT_USER, "via_c2", "user prefers via diameters in mm",
               status="candidate", source="thread:c2")

    items = store.search(namespace_for(DEFAULT_USER))
    cands = {it.key: it.value for it in items if it.value["status"] == "candidate"}
    assert set(cands) == {"via_c1", "via_c2"}  # both survive, not last-write-wins
    assert {cands["via_c1"]["source"], cands["via_c2"]["source"]} == {"thread:c1", "thread:c2"}

    clear_recall_journal()
    fake = FakeModel(["ok"])
    build_memory_graph(fake.respond, store=store).invoke(initial_state("q"), thread("c", user_id=DEFAULT_USER))
    assert "prefers via" not in system_seen(fake)  # no candidate reaches the context
    assert dict(recall_journal[-1]["skipped"])["via_c1"] == "status=candidate"


def test_wrong_memory_correction_supersedes():
    """A confirmed-but-wrong memory is superseded by a correction; the new thread sees only the fix."""
    store = InMemoryStore()
    put_memory(store, DEFAULT_USER, "via_dia_v1", "default via diameter = 0.5 mm", source="thread:old")
    put_memory(store, DEFAULT_USER, "via_dia_v2", "default via diameter = 0.3 mm (correction, spec v3)",
               status="candidate", source="thread:fix")
    supersede_memory(store, DEFAULT_USER, "via_dia_v1", by_key="via_dia_v2")
    confirm_memory(store, DEFAULT_USER, "via_dia_v2")

    clear_recall_journal()
    fake = FakeModel(["ok"])
    build_memory_graph(fake.respond, store=store).invoke(initial_state("过孔默认直径？"), thread("f", user_id=DEFAULT_USER))

    system = system_seen(fake)
    assert "0.3 mm (correction, spec v3)" in system  # corrected value injected
    assert "0.5 mm" not in system  # wrong value never reaches the model again
    old = store.get(namespace_for(DEFAULT_USER), "via_dia_v1")
    assert old.value["status"] == "superseded" and old.value["superseded_by"] == "via_dia_v2"
    assert dict(recall_journal[-1]["skipped"])["via_dia_v1"] == "status=superseded"  # audit trail, not deletion


def test_expired_memory_is_not_injected():
    """An expired entry is skipped at recall with reason 'expired'; an active one still loads."""
    store = InMemoryStore()
    put_memory(store, DEFAULT_USER, "spec_outer", "outer min trace width = 0.1 mm")
    put_memory(store, DEFAULT_USER, "stale_offer", "prototype discount 20% (2020)",
               expires_at="2020-01-01T00:00:00+00:00")
    put_memory(store, DEFAULT_USER, "future_note", "roadmap item for 2099",
               expires_at="2099-01-01T00:00:00+00:00")

    clear_recall_journal()
    fake = FakeModel(["ok"])
    build_memory_graph(fake.respond, store=store).invoke(initial_state("线宽？"), thread("e", user_id=DEFAULT_USER))

    system = system_seen(fake)
    assert "discount" not in system  # expired -> absent from the model's context
    assert "0.1 mm" in system and "roadmap" in system  # active entries still injected
    assert dict(recall_journal[-1]["skipped"])["stale_offer"] == "expired"


def test_trim_keeps_system_and_recent_within_budget():
    """trim_messages keeps every system message plus the newest turns that fit the budget."""
    history = build_long_history()
    budget = messages_tokens([history[0], history[-2], history[-1]]) + 5

    kept, dropped = trim_messages(history, budget)
    assert kept[0]["role"] == "system"  # system always survives
    assert kept[-2:] == history[-2:]  # the key recent turn pair survives
    assert [m["content"][:6] for m in dropped] == ["我上周在苏州", "好的，了解了", "外层最小线宽", "已记录外层最"]
    # the kept non-system window really is within budget
    assert messages_tokens(kept) - messages_tokens([m for m in kept if m["role"] == "system"]) <= budget


def test_collapse_summary_preserves_key_recent_information():
    """After trim+summarize the model still sees the key spec line verbatim; dropped text only survives in the summary."""
    history = build_long_history()
    budget = messages_tokens([history[0], history[-2], history[-1]]) + 5

    collapsed = collapse_with_summary(history, budget, fake_summarizer)
    fake = FakeModel(["ok"])
    fake.respond(collapsed)
    seen = json.dumps(fake.seen[0], ensure_ascii=False)

    assert "内层最小线宽 = 0.127 mm（spec v3" in seen  # the MUST-SEE line, verbatim
    assert "苏州" not in seen  # dropped filler is gone from the context
    summaries = [m for m in fake.seen[0] if m["content"].startswith("[summary")]
    assert len(summaries) == 1 and "4 earlier messages" in summaries[0]["content"]
    assert "0.1" in summaries[0]["content"]  # numbers survive the summary...
    assert "外层最小线宽" not in seen  # ...but their binding to the parameter does NOT


def test_selective_loading_injects_only_relevant_memories():
    """With selective=True only topic-matching memories enter the model's context."""
    store = InMemoryStore()
    put_memory(store, DEFAULT_USER, "spec_inner", FACT_INNER_TRACE, topics="线宽 trace spec 内层")
    put_memory(store, DEFAULT_USER, "cat_name", "user's cat is named Mira", topics="cat 个人")
    put_memory(store, DEFAULT_USER, "unit_pref", "user prefers answers in mm units", topics="单位 unit")
    question = "内层最小线宽是多少？"

    fake = FakeModel(["ok"])
    build_memory_graph(fake.respond, store=store, selective=True).invoke(
        initial_state(question), thread("s", user_id=DEFAULT_USER)
    )
    system = system_seen(fake)
    assert "0.127 mm" in system  # relevant memory injected
    assert "Mira" not in system and "prefers answers" not in system  # irrelevant ones held back
    injected, skipped = recall_memories(store, DEFAULT_USER, question=question, selective=True)
    assert [it.key for it in injected] == ["spec_inner"]
    assert dict(skipped)["cat_name"] == "irrelevant"


def test_long_tool_result_managed_as_digest_and_reference():
    """The managed variant hands the model a digest + artifact reference, not the 200-row blob."""
    blob = fetch_parameter_table()

    naive = FakeModel(["naive"])
    naive_state = build_tool_graph(naive.respond, managed=False).invoke(initial_state("列出参数表概要"))
    managed = FakeModel(["managed"])
    managed_state = build_tool_graph(managed.respond, managed=True).invoke(initial_state("列出参数表概要"))

    naive_tool = [m for m in naive.seen[-1] if m["role"] == "tool"][0]["content"]
    managed_tool = [m for m in managed.seen[-1] if m["role"] == "tool"][0]["content"]
    assert naive_tool == blob  # naive really did ship the whole table
    assert "[artifact table:a1]" in managed_tool and "200 rows" in managed_tool
    assert "param_100" not in managed_tool  # no table row leaked into the context
    assert len(naive_tool) > 20 * len(managed_tool)  # the size win is structural, not marginal
    # the blob itself is parked in session state, retrievable by reference
    assert managed_state["artifacts"]["table:a1"] == blob
    assert naive_state["artifacts"] == {}
    assert digest_table(blob).startswith("200 rows")


def test_noise_context_flips_the_scripted_answer():
    """Same task, same tool-free graph: 5 irrelevant injected lines flip the answer's unit."""
    store = InMemoryStore()
    put_memory(store, DEFAULT_USER, "spec_inner", FACT_INNER_TRACE, topics="线宽 trace spec")
    question = "内层最小线宽是多少 mm？请以 mm 回答。"

    clean = DistractibleModel()
    clean_state = build_memory_graph(clean.respond, store=store).invoke(
        initial_state(question), thread("n1", user_id=DEFAULT_USER)
    )
    noisy = DistractibleModel()
    noisy_state = build_memory_graph(noisy.respond, store=store, extra_noise=NOISE_MEMORIES).invoke(
        initial_state(question), thread("n2", user_id=DEFAULT_USER)
    )

    assert clean_state["final_answer"] == "内层最小线宽为 0.127 mm。"  # correct unit
    assert noisy_state["final_answer"] == "内层最小线宽约为 5.000 mil。"  # hijacked by "user prefers mil"
    # the cause is visible in what the models were handed: identical except the noise block
    clean_system, noisy_system = system_seen(clean), system_seen(noisy)
    assert "user prefers mil" not in clean_system and "user prefers mil" in noisy_system
    assert estimate_tokens(noisy_system) > estimate_tokens(clean_system)  # and it cost MORE context


def test_prefix_stability_is_necessary_not_sufficient():
    """Identical stable prefixes make all payload pairs share exactly that prefix -- and no more."""
    payloads = [wire_payload(s) for s in ["3 mm 是多少 mil？", "0.127 mm 是多少 mil？", "内层最小线宽是多少 mm？"]]
    shared = common_prefix_len(payloads[0], payloads[1])
    assert shared == common_prefix_len(payloads[0], payloads[2])  # the shared part is CONSTANT
    assert shared >= len(STABLE_PREFIX)  # it covers the whole stable prefix...
    assert payloads[0][shared:] != payloads[1][shared:]  # ...and stops where the suffix begins
    # DESIGN §7: prefix equality is a NECESSARY condition for provider cache
    # hits, never proof of one -- only usage telemetry (cached_tokens /
    # cache_read_input_tokens) can prove a hit. Not asserted here; 未实测.


def test_injectedstore_annotation_fails_on_plain_nodes():
    """Probed on 1.2.11: prebuilt.InjectedStore on a plain function node raises TypeError at invoke."""
    graph = build_injectedstore_pitfall_graph(InMemoryStore())
    with pytest.raises(TypeError, match=r"missing 1 required keyword-only argument: 'store'"):
        graph.invoke(initial_state("q"))


def test_state_channels_never_leak_into_the_model_context():
    """Session-state channels (task_note/artifacts) live in State only; the model sees messages only."""
    store = InMemoryStore()
    fake = FakeModel(["ok"])
    state = build_memory_graph(fake.respond, store=store).invoke(
        initial_state("问题", task_note="SECRET-internal-note"), thread("leak", user_id=DEFAULT_USER)
    )
    assert state["task_note"] == "SECRET-internal-note"  # present in session state...
    assert "SECRET-internal-note" not in json.dumps(fake.seen[0], ensure_ascii=False)  # ...absent from the context
