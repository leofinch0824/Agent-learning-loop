"""L5 - Context, Retrieval and Memory.

Scenario: the PCB process-parameter assistant from L2 grows a memory. The
lesson separates FOUR places a "fact" can live, and proves -- run by run --
which ones a NEW thread can see:

    ONE thread (t1)                                 across ALL threads
  +-------------------------------------------+   +-----------------------------+
  | working context                           |   | Store (cross-session)       |
  |   = the messages list handed to the model |   |   keyed by (namespace, key) |
  | session state                             |   |   survives every thread     |
  |   = State channels (per thread)           |   +-----------------------------+
  | checkpoint                                |
  |   = per-thread execution history          |        recall node   : Store -> working context
  +-------------------------------------------+        writeback node : conversation -> Store

Every demo uses fake models that record the EXACT message lists they were
handed (FakeModel.seen / DistractibleModel.seen). The whole point of the
lesson is to assert on WHAT THE MODEL SAW, not on what anyone "knows":
memory injection only counts if the string is inside the messages the model
received.

Run it:

    poetry run python lessons/l5_context_and_memory/main.py

Design promise: all seven demos are deterministic offline; every claim is
backed by a direct record (model-seen messages, recall journal, store entry
states, byte/token counts), and the prefix-cache section explicitly proves
only the NECESSARY condition offline -- real hits require provider
telemetry, which is marked 未实测 (no provider telemetry configured).
"""

from __future__ import annotations

import json
import operator
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Callable, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import InjectedStore
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lib import (  # noqa: E402
    banner,
    build_checkpointer,
    print_history,
    print_state,
    thread,
    trace,
)


def say(where: str, message: str) -> None:
    """Node-side console output (flushed so it stays readable)."""
    print(f"    [{where}] {message}", flush=True)


# ---------------------------------------------------------------------------
# 1. Definitions: state, memory value schema, recall filter chain, journals
# ---------------------------------------------------------------------------

DEFAULT_USER = "u-pcb"
FACT_INNER_TRACE = "inner min trace width = 0.127 mm"

UNIT_TO_MM = {"mm": 1.0, "cm": 10.0, "um": 0.001, "mil": 0.0254, "inch": 25.4}


class AgentState(TypedDict):
    """State of the memory-carrying agent.

    `messages` is the working context: the ONLY thing the model ever sees.
    It needs a reducer because recall / tool / model nodes all append. Every
    other channel is session state -- visible to nodes and checkpoints, but
    NEVER to the model unless a node explicitly copies it into a message.
    """

    question: str
    messages: Annotated[list[dict], operator.add]
    task_note: str  # session-state layer: a fact that must NOT leak across threads
    artifacts: dict[str, str]  # long tool outputs kept OUT of messages (demo 5)
    final_answer: str


def initial_state(question: str, *, task_note: str = "") -> AgentState:
    return {
        "question": question,
        "messages": [],  # the recall node seeds system+user, in that order
        "task_note": task_note,
        "artifacts": {},
        "final_answer": "",
    }


def namespace_for(user_id: str) -> tuple[str, ...]:
    """Store namespace convention: ("memories", <user_id>)."""
    return ("memories", user_id)


def put_memory(
    store: BaseStore,
    user_id: str,
    key: str,
    text: str,
    *,
    status: str = "confirmed",
    version: int = 1,
    source: str = "seed",
    topics: str = "",
    expires_at: str | None = None,
) -> None:
    """Write one memory entry.

    langgraph 1.2.11 probed fact: `put` has NO metadata kwarg -- provenance
    (status/version/source/expiry) lives INSIDE the value dict, where our own
    filter chain can read it. That is deliberate: the metadata contract is
    ours, not the store's.
    """
    store.put(
        namespace_for(user_id),
        key,
        {
            "text": text,
            "status": status,  # confirmed | candidate | superseded
            "version": version,
            "source": source,
            "topics": topics,  # space-separated keywords for selective loading
            "expires_at": expires_at,  # ISO timestamp, app-level expiry
        },
    )


def confirm_memory(store: BaseStore, user_id: str, key: str, *, confirmed_by: str = "human") -> None:
    """Promote a candidate to confirmed and bump the version."""
    ns = namespace_for(user_id)
    item = store.get(ns, key)
    value = dict(item.value)
    value.update(status="confirmed", version=value.get("version", 1) + 1, confirmed_by=confirmed_by)
    store.put(ns, key, value)


def supersede_memory(store: BaseStore, user_id: str, old_key: str, by_key: str) -> None:
    """Mark an old entry superseded_by a correction -- keep it for audit, stop injecting it."""
    ns = namespace_for(user_id)
    item = store.get(ns, old_key)
    value = dict(item.value)
    value.update(status="superseded", superseded_by=by_key)
    store.put(ns, old_key, value)


def is_expired(expires_at: str | None, now: datetime | None = None) -> bool:
    """App-level expiry check.

    langgraph 1.2.11 probed fact: InMemoryStore.put(..., ttl=...) raises
    NotImplementedError ("TTL is not supported by InMemoryStore..."), so
    expiry is our own rule over an `expires_at` field. Deterministic in
    tests via fixed ISO timestamps.
    """
    if not expires_at:
        return False
    moment = now or datetime.now(timezone.utc)
    return datetime.fromisoformat(expires_at) <= moment


def is_relevant(value: dict, question: str) -> bool:
    """Cheap stand-in for retrieval scoring: keyword overlap on `topics`.

    No topic info -> load it (conservative default: recall everything that
    survived the status/expiry gates).
    """
    topics = (value.get("topics") or "").split()
    if not topics:
        return True
    q = question.lower()
    return any(tok.lower() in q for tok in topics)


def recall_memories(
    store: BaseStore | None,
    user_id: str,
    *,
    question: str = "",
    selective: bool = False,
    now: datetime | None = None,
) -> tuple[list, list[tuple[str, str]]]:
    """The ONE visible filter chain every run goes through:

        raw search -> status gate -> expiry gate -> (optional) relevance gate

    Returns (injected_items, skipped) where skipped is [(key, reason)] --
    the direct evidence demos print and tests assert on. Sorted by key so
    output is deterministic regardless of store iteration order.
    """
    injected: list = []
    skipped: list[tuple[str, str]] = []
    if store is None:
        return injected, skipped
    items = sorted(store.search(namespace_for(user_id)), key=lambda it: it.key)
    for item in items:
        value = item.value
        if value.get("status") != "confirmed":
            skipped.append((item.key, f"status={value.get('status')}"))
            continue
        if is_expired(value.get("expires_at"), now):
            skipped.append((item.key, "expired"))
            continue
        if selective and not is_relevant(value, question):
            skipped.append((item.key, "irrelevant"))
            continue
        injected.append(item)
    return injected, skipped


# Recall journal: what each thread's recall node injected / skipped and why.
# Tests assert on this directly -- no framework internals are mocked.
recall_journal: list[dict] = []


def clear_recall_journal() -> None:
    recall_journal.clear()


def format_memory_line(item) -> str:
    value = item.value
    return f"- {value['text']} [v{value.get('version', 1)}]"


# ---------------------------------------------------------------------------
# 2. The scripted models: they record EXACTLY what they were handed
# ---------------------------------------------------------------------------


class ScriptExhaustedError(RuntimeError):
    """A fake ran out of scripted replies -- a bug in the demo setup."""


class FakeModel:
    """A scripted responder that RECORDS every message list it is handed.

    `respond(messages)` matches the live adapter's interface (L2 pattern).
    `seen` is the lesson's core evidence: the working context as the model
    actually received it, not as we assume it did.
    """

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.seen: list[list[dict]] = []

    def respond(self, messages: list[dict]) -> str:
        if not self.replies:
            raise ScriptExhaustedError(
                f"script ran dry after {len(self.seen)} turns; last roles seen: "
                f"{[m.get('role') for m in messages[-3:]]}"
            )
        self.seen.append([dict(m) for m in messages])
        return self.replies.pop(0)


class DistractibleModel:
    """A scripted model with ONE rule that reads ONLY its system prompt:

    if any line matches `prefers <unit>`, format the answer in THAT unit --
    even when the question explicitly asked for another one.

    This is the deterministic stand-in for attention hijacking by noise. A
    real model is not this mechanical; the point is to make the CAUSAL chain
    visible: the injected text, and nothing else, flipped the answer.
    """

    def __init__(self, value_mm: float = 0.127) -> None:
        self.value_mm = value_mm
        self.seen: list[list[dict]] = []

    def respond(self, messages: list[dict]) -> str:
        self.seen.append([dict(m) for m in messages])
        system = "\n".join(m.get("content", "") for m in messages if m.get("role") == "system")
        match = re.search(r"prefers ([a-z]+)", system)
        unit = match.group(1) if match else "mm"
        if unit not in UNIT_TO_MM:
            unit = "mm"
        if unit == "mm":
            return f"内层最小线宽为 {self.value_mm} mm。"
        converted = self.value_mm * UNIT_TO_MM["mm"] / UNIT_TO_MM[unit]
        return f"内层最小线宽约为 {converted:.3f} {unit}。"


# ---------------------------------------------------------------------------
# 3. Graph builders: recall -> model -> writeback, plus the probed pitfall
# ---------------------------------------------------------------------------


def make_recall_node(*, selective: bool = False, extra_noise: list[str] | None = None):
    """Inject confirmed memories into the system prompt at run start.

    langgraph 1.2.11 probed fact: a plain function node receives the store
    ONLY via a keyword-only parameter named `store` annotated exactly as
    `BaseStore` (matched by name + annotation from the runtime object), or
    via `langgraph.config.get_store()`. See demo 2 for the InjectedStore
    anti-pattern.
    """

    def recall_node(state: AgentState, config, *, store: BaseStore) -> dict:
        tid = config["configurable"].get("thread_id", "?")
        user_id = config["configurable"].get("user_id", DEFAULT_USER)
        injected, skipped = recall_memories(
            store, user_id, question=state["question"], selective=selective
        )
        lines = [format_memory_line(item) for item in injected]
        lines += [f"- {noise}" for noise in (extra_noise or [])]
        if lines:
            system = "Known facts about this user (confirmed memory store):\n" + "\n".join(lines)
        else:
            system = "Known facts about this user: (none confirmed)"
        recall_journal.append(
            {
                "thread": tid,
                "user": user_id,
                "injected": [item.key for item in injected],
                "skipped": skipped,
                "noise": len(extra_noise or []),
            }
        )
        say("recall", f"injected {len(injected)} memory line(s), skipped {skipped or '{}'}")
        # Seed working context in display order: system BEFORE user.
        return {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": state["question"]},
            ]
        }

    return recall_node


def make_model_node(respond: Callable[[list[dict]], str]):
    def model_node(state: AgentState) -> dict:
        # The model sees `state["messages"]` -- the whole lesson in one line.
        reply = respond(state["messages"])
        say("model", f"received {len(state['messages'])} message(s), roles={[m['role'] for m in state['messages']]}")
        return {"messages": [{"role": "assistant", "content": reply}], "final_answer": reply}

    return model_node


def make_writeback_node(key: str, candidate_text: str):
    """End-of-run write-back: propose ONE candidate memory with provenance.

    The candidate carries source=thread:<id> and status=candidate; nothing is
    injected until an explicit confirmation flips it (demo 2).
    """

    def writeback_node(state: AgentState, config, *, store: BaseStore) -> dict:
        tid = config["configurable"].get("thread_id", "?")
        user_id = config["configurable"].get("user_id", DEFAULT_USER)
        put_memory(
            store,
            user_id,
            key,
            candidate_text,
            status="candidate",
            version=1,
            source=f"thread:{tid}",
        )
        say("writeback", f"proposed candidate {key!r} (status=candidate, source=thread:{tid})")
        return {}

    return writeback_node


def build_memory_graph(
    respond: Callable[[list[dict]], str],
    *,
    store: BaseStore | None,
    checkpointer=None,
    selective: bool = False,
    extra_noise: list[str] | None = None,
    writeback: tuple[str, str] | None = None,
):
    """START -> recall -> model -> (writeback) -> END."""
    builder = StateGraph(AgentState)
    builder.add_node("recall", make_recall_node(selective=selective, extra_noise=extra_noise))
    builder.add_node("model", make_model_node(respond))
    builder.add_edge(START, "recall")
    builder.add_edge("recall", "model")
    if writeback is not None:
        key, text = writeback
        builder.add_node("writeback", make_writeback_node(key, text))
        builder.add_edge("model", "writeback")
        builder.add_edge("writeback", END)
    else:
        builder.add_edge("model", END)
    return builder.compile(store=store, checkpointer=checkpointer)


def build_injectedstore_pitfall_graph(store: BaseStore):
    """The probed anti-pattern: InjectedStore() on a PLAIN function node.

    On langgraph 1.2.11 `langgraph.prebuilt.InjectedStore` serves ToolNode-
    style tools; a plain node annotated with it does NOT get the store
    injected and dies with TypeError at invoke time.
    """

    def bad_recall(state: AgentState, *, store: Annotated[BaseStore, InjectedStore()]) -> dict:  # noqa: ARG001
        return {"final_answer": "never reached"}

    builder = StateGraph(AgentState)
    builder.add_node("bad_recall", bad_recall)
    builder.add_edge(START, "bad_recall")
    builder.add_edge("bad_recall", END)
    return builder.compile(store=store)


# ---------------------------------------------------------------------------
# 4. Token budgeting: estimator, trimming, mid-history summary, selective
#    loading (in recall), long tool results
# ---------------------------------------------------------------------------


def estimate_tokens(text: str) -> int:
    """Declared approximation: chars // 4, minimum 1 (the classic heuristic).

    It is NOT a tokenizer; it is a deterministic budget unit so the trimming
    arithmetic below is reproducible in tests.
    """
    return max(1, len(text) // 4)


def message_tokens(message: dict) -> int:
    return estimate_tokens(message.get("content") or "")


def messages_tokens(messages: list[dict]) -> int:
    return sum(message_tokens(m) for m in messages)


def trim_messages(messages: list[dict], budget: int) -> tuple[list[dict], list[dict]]:
    """Keep every system message + the most recent non-system messages that
    fit in `budget`; return (new_message_list, dropped).

    Written by hand on purpose (no framework magic): the policy -- system
    messages survive, recency wins, everything older is dropped -- is the
    thing under study, so it must be readable.
    """
    system = [m for m in messages if m.get("role") == "system"]
    rest = [m for m in messages if m.get("role") != "system"]
    used = messages_tokens(system)
    kept_rev: list[dict] = []
    for message in reversed(rest):
        cost = message_tokens(message)
        if used + cost > budget:
            break
        used += cost
        kept_rev.append(message)
    kept_recent = list(reversed(kept_rev))
    kept_set = {id(m) for m in kept_recent}
    dropped = [m for m in rest if id(m) not in kept_set]
    return system + kept_recent, dropped


def fake_summarizer(dropped: list[dict]) -> str:
    """Scripted mid-history summarizer: deterministic, deliberately lossy.

    It keeps the COUNT of dropped messages and the bare numbers that
    appeared -- and loses the binding of number to parameter. The precision
    loss is the teaching point, mirrored by the real summarizer in
    live_demo.py under the same contract.
    """
    numbers = sorted({tok for m in dropped for tok in re.findall(r"\d+\.?\d*", m.get("content") or "")})[:6]
    return (
        f"Earlier {len(dropped)} messages exchanged greetings and background; "
        f"numeric facts mentioned: {', '.join(numbers) or 'none'}."
    )


def collapse_with_summary(
    messages: list[dict],
    budget: int,
    summarizer: Callable[[list[dict]], str],
) -> list[dict]:
    """trim -> summarize the dropped middle -> [system..., summary, recent...]."""
    kept, dropped = trim_messages(messages, budget)
    if not dropped:
        return kept
    summary = {
        "role": "system",
        "content": f"[summary of {len(dropped)} earlier messages] {summarizer(dropped)}",
    }
    n_system = sum(1 for m in kept if m.get("role") == "system")
    return kept[:n_system] + [summary] + kept[n_system:]


def build_long_history() -> list[dict]:
    """A conversation whose middle is filler and whose END carries the key spec.

    `u3` ("内层最小线宽 = 0.127 mm（spec v3...）") is the line the model MUST
    still see after trimming; `u2` (outer width) is the fact that only
    survives as a bare number inside the summary.
    """
    return [
        {"role": "system", "content": "You are a PCB process parameter assistant."},
        {"role": "user", "content": "我上周在苏州出差，顺便看了工厂的湿制程线。" + "细节略。" * 30},
        {"role": "assistant", "content": "好的，了解了。" + "（背景已记录。）" * 20},
        {"role": "user", "content": "外层最小线宽 0.1 mm，记住这个数。"},
        {"role": "assistant", "content": "已记录外层最小线宽 0.1 mm。" + "（已记录。）" * 20},
        {"role": "user", "content": "内层最小线宽 = 0.127 mm（spec v3，2026-06 生效）。"},
        {"role": "assistant", "content": "已确认：内层最小线宽 0.127 mm，spec v3。"},
    ]


# --- long tool results ------------------------------------------------------


def fetch_parameter_table(rows: int = 200) -> str:
    """A 200-row fake tool result -- the blob we do NOT want in messages."""
    lines = ["idx,parameter,value_mm,note"]
    for i in range(rows):
        lines.append(f"{i},param_{i:03d},{(i % 37) * 0.05:.2f},batch-{i % 7}")
    return "\n".join(lines)


def digest_table(blob: str) -> str:
    """Short digest of the table: count, min/max, first and last row."""
    rows = blob.splitlines()[1:]
    values = [float(r.split(",")[2]) for r in rows]
    return (
        f"{len(rows)} rows, value_mm min={min(values):.2f} max={max(values):.2f}, "
        f"first={rows[0]}, last={rows[-1]}"
    )


def make_fetch_node(*, managed: bool):
    """Demo 5's two variants of ONE tool node.

    naive:   append the full blob as the tool observation.
    managed: park the blob in the `artifacts` state channel, append only a
             reference + digest as the observation. The model still gets the
             numbers it needs; the transcript stays small and reusable.
    """

    def fetch_node(state: AgentState) -> dict:
        blob = fetch_parameter_table()
        if not managed:
            say("fetch", f"appending FULL table to messages ({len(blob)} chars)")
            return {
                "messages": [
                    {"role": "user", "content": state["question"]},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "c1",
                                "type": "function",
                                "function": {"name": "fetch_parameter_table", "arguments": "{}"},
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": "c1", "content": blob},
                ]
            }
        digest = digest_table(blob)
        reference = f"[artifact table:a1] {digest} -- full table stored in state, not in messages"
        say("fetch", f"storing blob in artifacts ({len(blob)} chars), appending digest ({len(reference)} chars)")
        return {
            "artifacts": {"table:a1": blob},
            "messages": [
                {"role": "user", "content": state["question"]},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "fetch_parameter_table", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": reference},
            ],
        }

    return fetch_node


def build_tool_graph(respond: Callable[[list[dict]], str], *, managed: bool):
    """START -> fetch -> model -> END (demo 5, naive vs managed)."""
    builder = StateGraph(AgentState)
    builder.add_node("fetch", make_fetch_node(managed=managed))
    builder.add_node("model", make_model_node(respond))
    builder.add_edge(START, "fetch")
    builder.add_edge("fetch", "model")
    builder.add_edge("model", END)
    return builder.compile()


# ---------------------------------------------------------------------------
# 5. Prefix-cache mechanics (necessary condition only -- hits need telemetry)
# ---------------------------------------------------------------------------

STABLE_PREFIX = (
    "You are a PCB process-parameter assistant.\n"
    "Available tools: unit_convert (mm/cm/um/mil/inch), lookup_spec (spec table).\n"
    "Policy: always answer in the unit the question explicitly asks for.\n"
    "House style: short answers, one fact per line.\n"
)


def wire_payload(suffix: str) -> str:
    """The serialized request a provider would receive for one call."""
    return json.dumps(
        [
            {"role": "system", "content": STABLE_PREFIX},
            {"role": "user", "content": suffix},
        ],
        ensure_ascii=False,
    )


def common_prefix_len(a: str, b: str) -> int:
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    return i


# ---------------------------------------------------------------------------
# 6. Demos
# ---------------------------------------------------------------------------


def demo_four_layers():
    banner("DEMO 1 - 四层分工：同一条事实放进四层，新 thread 能看到哪几层")
    store = InMemoryStore()
    put_memory(store, DEFAULT_USER, "spec_inner", FACT_INNER_TRACE, source="seed")
    # ONE checkpointer + ONE store shared by BOTH graphs: only the model
    # script differs, so "what survives into a new thread" is not an artifact
    # of fresh objects.
    checkpointer = build_checkpointer("memory")

    # The same fact, placed in each layer around thread A:
    fake_a = FakeModel(["A 回复：好的。"])
    graph_a = build_memory_graph(fake_a.respond, store=store, checkpointer=checkpointer)
    cfg_a = thread("layer-A", user_id=DEFAULT_USER)
    state_a_task_note = "inner min trace width = 0.127 mm (session-state layer)"
    trace(
        graph_a,
        initial_state("记住：inner min trace width = 0.127 mm（工作上下文层）", task_note=state_a_task_note),
        cfg_a,
        label="thread A: fact enters working context / state / checkpoint",
    )
    print_state(graph_a, cfg_a)
    print_history(graph_a, cfg_a, limit=3)

    # NEW thread, same checkpointer + same store:
    clear_recall_journal()
    fake_b = FakeModel(["B 回复：0.127 mm。"])
    graph_b = build_memory_graph(fake_b.respond, store=store, checkpointer=checkpointer)
    cfg_b = thread("layer-B", user_id=DEFAULT_USER)
    graph_b.invoke(initial_state("内层最小线宽是多少？"), cfg_b)

    seen_b = fake_b.seen[0]
    system_b = "\n".join(m["content"] for m in seen_b if m["role"] == "system")
    vals_b = graph_b.get_state(cfg_b).values
    b_history_blob = json.dumps([s.values for s in graph_b.get_state_history(cfg_b)], ensure_ascii=False)
    a_history_exists = len(list(graph_a.get_state_history(cfg_a))) > 0
    print("\n  thread B 的模型实际收到的消息:")
    for m in seen_b:
        preview = (m.get("content") or "")[:80].replace("\n", " ")
        print(f"    {m['role']:<9} : {preview}")
    print("\n  各层在新 thread 的可见性:")
    rows = [
        ("Store (跨会话)", "0.127 mm" in system_b),
        ("工作上下文 (A 的 user 消息)", "工作上下文层" in json.dumps(seen_b, ensure_ascii=False)),
        ("会话状态 (A 的 task_note)", vals_b.get("task_note", "") == state_a_task_note),
        # A's history still exists (a_history_exists) but is reachable only
        # under A's thread_id: B's own checkpoint stream holds none of it.
        ("checkpoint (A 的执行历史)", "工作上下文层" in b_history_blob),
    ]
    for layer, visible in rows:
        print(f"    {layer:<38} -> {'可见' if visible else '不可见'}")
    print("\n  recall journal:", recall_journal)
    print(f"  A 的 checkpoint 仍在 (可经 cfg_a 查询): {a_history_exists}; 但 B 的历史里没有 A 的任何内容。")
    print("  结论: 只有 Store 层跨 thread 存活; checkpoint/state/工作上下文都随 thread 终结。")
    print("  (checkpoint 是按 thread_id 隔离的执行历史, 不能当跨会话知识库用。)")


def demo_store_roundtrip():
    banner("DEMO 2 - 跨会话 Store 检索与写回：candidate -> confirm -> 新 thread 可见")
    print("  先看 probed 反例: langgraph.prebuilt.InjectedStore 用在普通函数节点上:")
    store = InMemoryStore()
    pitfall = build_injectedstore_pitfall_graph(store)
    try:
        pitfall.invoke(initial_state("q"))
    except TypeError as exc:
        print(f"    invoke raised: TypeError: {exc}")
        print("    修复: keyword-only `*, store: BaseStore`（按名字+注解注入）或 get_store()。")
        print("    更隐蔽的坑: compile 时不挂 store, 节点拿到的是静默 None, 不报错。")

    store = InMemoryStore()
    clear_recall_journal()

    # Turn 1 (thread r1): a conversation that yields a candidate memory.
    fake1 = FakeModel(["好的，之后都默认用 mm 单位给你结果。"])
    graph1 = build_memory_graph(
        fake1.respond,
        store=store,
        writeback=("unit_pref", "user prefers answers in mm units"),
    )
    graph1.invoke(initial_state("帮我把 3 mm 换成 mil，以后结果默认用 mm。"), thread("r1", user_id=DEFAULT_USER))
    candidate = store.get(namespace_for(DEFAULT_USER), "unit_pref")
    print(f"\n  写回的候选: status={candidate.value['status']}, version={candidate.value['version']}, source={candidate.value['source']}")

    # Turn 2 (NEW thread r2, BEFORE confirmation): candidate must NOT appear.
    fake2 = FakeModel(["r2 回复。"])
    build_memory_graph(fake2.respond, store=store).invoke(initial_state("内层最小线宽是多少？"), thread("r2", user_id=DEFAULT_USER))
    system_r2 = "\n".join(m["content"] for m in fake2.seen[0] if m["role"] == "system")
    print(f"  确认前, 新 thread 的 system 是否包含偏好: {'prefers answers in mm' in system_r2} (journal: {recall_journal[-1]['skipped']})")

    # Human confirmation flips the candidate.
    confirm_memory(store, DEFAULT_USER, "unit_pref", confirmed_by="human-demo")
    confirmed = store.get(namespace_for(DEFAULT_USER), "unit_pref")
    print(f"  确认后: status={confirmed.value['status']}, version={confirmed.value['version']}, confirmed_by={confirmed.value['confirmed_by']}")

    # Turn 3 (NEW thread r3, AFTER confirmation): the model SEES it.
    fake3 = FakeModel(["r3 回复：0.127 mm。"])
    build_memory_graph(fake3.respond, store=store).invoke(initial_state("内层最小线宽是多少？"), thread("r3", user_id=DEFAULT_USER))
    system_r3 = "\n".join(m["content"] for m in fake3.seen[0] if m["role"] == "system")
    print(f"  确认后, 新 thread 的 system 是否包含偏好: {'prefers answers in mm' in system_r3}")
    print(f"\n  r3 模型收到的 system 原文:\n    {system_r3}")
    print("  掌握证据 = 断言写在模型 seen 的消息里, 而不是 store 里 '应该有'。")


def demo_memory_lifecycle():
    banner("DEMO 3 - 记忆更新 / 冲突 / 过期")
    store = InMemoryStore()
    clear_recall_journal()

    # -- probed fact: InMemoryStore has no native TTL ------------------------
    try:
        store.put(namespace_for(DEFAULT_USER), "ttl_probe", {"text": "x"}, ttl=60)
    except NotImplementedError as exc:
        print(f"  probed: put(ttl=...) -> NotImplementedError: {exc}")
        print("  -> 过期是应用层规则: value 里带 expires_at, recall 过滤链判断。\n")

    # -- expiry ---------------------------------------------------------------
    put_memory(store, DEFAULT_USER, "spec_outer", "outer min trace width = 0.1 mm")
    put_memory(store, DEFAULT_USER, "stale_offer", "prototype discount 20% (2020)", expires_at="2020-01-01T00:00:00+00:00")
    fake = FakeModel(["ok"])
    build_memory_graph(fake.respond, store=store).invoke(initial_state("线宽？"), thread("exp", user_id=DEFAULT_USER))
    system = "\n".join(m["content"] for m in fake.seen[0] if m["role"] == "system")
    print(f"  过期条目被注入: {'discount' in system} | 未过期条目被注入: {'0.1 mm' in system}")
    print(f"  journal skipped: {recall_journal[-1]['skipped']}\n")

    # -- conflict: two candidates with sources, neither confirmed ------------
    put_memory(store, DEFAULT_USER, "via_pref_c1", "user prefers via diameters in mil", status="candidate", source="thread:c1")
    put_memory(store, DEFAULT_USER, "via_pref_c2", "user prefers via diameters in mm", status="candidate", source="thread:c2")
    items = store.search(namespace_for(DEFAULT_USER))
    cands = sorted(it.key for it in items if it.value["status"] == "candidate")
    print(f"  冲突写回共存: candidates={cands} (各自带 source, 都不注入, 等人工确认)\n")

    # -- wrong memory corrected via supersede ---------------------------------
    put_memory(store, DEFAULT_USER, "via_dia_v1", "default via diameter = 0.5 mm", source="thread:old")
    put_memory(store, DEFAULT_USER, "via_dia_v2", "default via diameter = 0.3 mm (correction, spec v3)", status="candidate", source="thread:fix")
    supersede_memory(store, DEFAULT_USER, "via_dia_v1", by_key="via_dia_v2")
    confirm_memory(store, DEFAULT_USER, "via_dia_v2", confirmed_by="human-demo")

    fake2 = FakeModel(["ok"])
    build_memory_graph(fake2.respond, store=store).invoke(initial_state("过孔默认直径？"), thread("fix", user_id=DEFAULT_USER))
    system2 = "\n".join(m["content"] for m in fake2.seen[0] if m["role"] == "system")
    old = store.get(namespace_for(DEFAULT_USER), "via_dia_v1")
    print(f"  纠正后新 thread 模型看到 0.3: {'0.3 mm' in system2} | 仍看到 0.5: {'0.5 mm' in system2}")
    print(f"  旧条目保留审计痕迹: status={old.value['status']}, superseded_by={old.value['superseded_by']}")
    print(f"  journal skipped: {recall_journal[-1]['skipped']}")
    print("  原则: 纠正 = 新版本上位 + 旧版本标记, 不是删除 -- 冲突双方依据都可回查。")


def demo_budget_trim_summary():
    banner("DEMO 4 - token 预算裁剪 / 中段摘要 / 选择性加载")
    history = build_long_history()
    sys_u3_a3 = [history[0], history[-2], history[-1]]
    budget = messages_tokens(sys_u3_a3) + 5  # just enough for system + the key turn pair

    print("  每条消息的 token 估算 (chars//4, 声明为近似值):")
    for i, m in enumerate(history):
        print(f"    [{i}] {m['role']:<9} ~{message_tokens(m):>3} tok  {(m['content'] or '')[:28]}...")
    print(f"  预算 = system + 最后两问 {budget} tok")

    kept, dropped = trim_messages(history, budget)
    collapsed = collapse_with_summary(history, budget, fake_summarizer)
    fake = FakeModel(["ok"])
    fake.respond(collapsed)  # the model is handed the COLLAPSED list
    seen = fake.seen[0]

    print(f"\n  裁剪后保留 {len(kept)} 条, 丢弃 {len(dropped)} 条; 折叠后总 token ~{messages_tokens(collapsed)}")
    print(f"  关键 spec 行仍在模型可见消息中: {'内层最小线宽 = 0.127 mm（spec v3' in json.dumps(seen, ensure_ascii=False)}")
    print(f"  被裁的旧背景整句仍在: {'苏州' in json.dumps(seen, ensure_ascii=False)}")
    print(f"  摘要消息: {[m['content'] for m in seen if m['content'].startswith('[summary')][0]}")
    print("  注意精度损失: '外层 0.1 mm' 只以裸数字 0.1 存活于摘要, 参数绑定丢了 --")
    print("  所以关键信息应尽量留在近期窗口, 摘要是兜底不是保险箱。\n")

    # -- selective loading ----------------------------------------------------
    store = InMemoryStore()
    put_memory(store, DEFAULT_USER, "spec_inner", FACT_INNER_TRACE, topics="线宽 trace spec 内层")
    put_memory(store, DEFAULT_USER, "cat_name", "user's cat is named Mira", topics="cat 个人")
    put_memory(store, DEFAULT_USER, "unit_pref", "user prefers answers in mm units", topics="单位 unit")
    question = "内层最小线宽是多少？"
    injected, skipped = recall_memories(store, DEFAULT_USER, question=question, selective=True)
    print(f"  选择性加载 (query={question!r}):")
    print(f"    injected = {[it.key for it in injected]}, skipped = {skipped}")


def demo_long_tool_result():
    banner("DEMO 5 - 长工具结果管理：200 行表格的两种进消息方式")
    blob = fetch_parameter_table()
    digest = digest_table(blob)
    print(f"  工具产物: {len(blob)} chars (~{estimate_tokens(blob)} tok) | digest: {len(digest)} chars")

    naive = FakeModel(["naive 回复。"])
    state_naive = build_tool_graph(naive.respond, managed=False).invoke(initial_state("列出参数表概要"))
    naive_tool_msg = [m for m in naive.seen[-1] if m["role"] == "tool"][0]
    print(f"\n  naive  : 模型收到 tool 消息 {len(naive_tool_msg['content'])} chars (整表进消息)")

    managed = FakeModel(["managed 回复：200 行, value_mm 0.00-1.80。"])
    state_managed = build_tool_graph(managed.respond, managed=True).invoke(initial_state("列出参数表概要"))
    managed_tool_msg = [m for m in managed.seen[-1] if m["role"] == "tool"][0]
    print(f"  managed: 模型收到 tool 消息 {len(managed_tool_msg['content'])} chars (digest + artifact 引用)")
    print(f"  managed 工具消息原文: {managed_tool_msg['content']}")
    print(f"  全表去向: state['artifacts'] keys = {list(state_managed['artifacts'])}, 原文 {len(state_managed['artifacts']['table:a1'])} chars")
    ratio = len(naive_tool_msg["content"]) // max(1, len(managed_tool_msg["content"]))
    print(f"  体积比 naive/managed ~= {ratio}x; 后续每轮模型调用都省下这部分上下文。")


NOISE_MEMORIES = [
    "user prefers mil",  # the poisoned line: mentions a UNIT
    "user's cat is named Mira",
    "the office printer jams on Thursdays",
    "user enjoyed the book 'Structure and Interpretation'",
    "team lunch is usually on Wednesdays",
]


def demo_noise_contrast():
    banner("DEMO 6 - 噪声上下文对照：更多上下文 != 更好")
    store = InMemoryStore()
    put_memory(store, DEFAULT_USER, "spec_inner", FACT_INNER_TRACE, topics="线宽 trace spec")
    question = "内层最小线宽是多少 mm？请以 mm 回答。"

    clean = DistractibleModel()
    clean_state = build_memory_graph(clean.respond, store=store).invoke(
        initial_state(question), thread("clean", user_id=DEFAULT_USER)
    )

    noisy = DistractibleModel()
    noisy_state = build_memory_graph(noisy.respond, store=store, extra_noise=NOISE_MEMORIES).invoke(
        initial_state(question), thread("noisy", user_id=DEFAULT_USER)
    )

    clean_system = "\n".join(m["content"] for m in clean.seen[0] if m["role"] == "system")
    noisy_system = "\n".join(m["content"] for m in noisy.seen[0] if m["role"] == "system")
    print(f"  问题: {question}")
    print(f"  clean: system ~{estimate_tokens(clean_system)} tok -> 答案: {clean_state['final_answer']}")
    print(f"  noisy : system ~{estimate_tokens(noisy_system)} tok (多 ~{estimate_tokens(noisy_system) - estimate_tokens(clean_system)} tok) -> 答案: {noisy_state['final_answer']}")
    print("\n  机制: 脚本模型的唯一规则是从 system 里找 'prefers <unit>' -- 找到就用那个单位。")
    print("  这是对真实模型注意力劫持的确定性替身(DESIGN §1: 离线断言确定性行为);")
    print("  它证明的是因果链: 注入的噪声文本, 而且只有它, 翻转了答案单位。")
    print("  修复方向: 选择性加载 + 确认门槛(DEMO 4/2), 而不是 '多记总没错'。")


def demo_prefix_cache():
    banner("DEMO 7 - 前缀缓存: 离线只能证明必要条件")
    suffixes = ["3 mm 是多少 mil？", "0.127 mm 是多少 mil？", "内层最小线宽是多少 mm？"]
    payloads = [wire_payload(s) for s in suffixes]
    wrapper = len(wire_payload("x")) - len(STABLE_PREFIX)  # JSON packaging overhead
    for (a, b), sa, sb in zip(zip(payloads, payloads[1:]), suffixes, suffixes[1:]):
        k = common_prefix_len(a, b)
        print(f"  {sa!r} vs {sb!r}: 公共前缀 {k} chars = 稳定前缀 {len(STABLE_PREFIX)} + 固定包装 {wrapper}")
    print("\n  离线可证明的全部: 三次请求共享同一段稳定前缀 -- 这是 provider 侧前缀缓存")
    print("  命中的必要条件, 不是命中本身。")
    print("  DESIGN §7 校正: 前缀哈希相等不能证明缓存命中; 压缩阈值/费用倍数也不普适。")
    print("  真实命中的唯一证据是 provider 遥测, 例如:")
    print("    - OpenAI: usage.prompt_tokens_details.cached_tokens (自动缓存, 有最低 token 水位线)")
    print("    - Anthropic 风格: usage.cache_read_input_tokens / cache_creation_input_tokens (显式 cache_control)")
    print("  水位线与折扣是各家可变参数, 只能当可调实验方案, 不能当通用常数。")
    print("\n  未实测（无 provider 遥测配置）: live_demo.py 会在配置 .env 后打印两次同前缀")
    print("  调用的 usage 字段; 字段缺失时不做任何命中宣称。")


if __name__ == "__main__":
    demo_four_layers()
    demo_store_roundtrip()
    demo_memory_lifecycle()
    demo_budget_trim_summary()
    demo_long_tool_result()
    demo_noise_contrast()
    demo_prefix_cache()
    banner("done", width=72)
