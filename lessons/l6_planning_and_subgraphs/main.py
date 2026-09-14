"""L6 - Planning, Subgraphs and Multi-Agent.

Scenario (one task family for the whole lesson): several PCB process
documents are split into chunks; each chunk gets a scripted summary; the
summaries are merged into an inspection report. Every "model" in this file
is a deterministic scripted fake, so structures -- not model luck -- are the
only variable.

The SAME task runs through three structures (ASCII topologies):

  (a) single agent (one big loop, L2 shape)      (b) fixed pipeline (compile-time plan)

    START -> model -> tools -> model ... -> END     START -> extract -> ( summary_0 || summary_1 || ... )
            one context accumulates                        |            ( compile-time fan-out, N fixed
            ALL chunks and observations                    +-> merge -> END   when the graph is BUILT )

  (c) planner + executor + replanner (Plan-and-Execute)

    START -> planner -> executor -> executor -> ... -> report -> END
                          ^  | failure
                          |  v
                          replanner  (regenerates the REMAINING plan, loops back)

On top of that, five mechanisms:
  * Send / map-reduce: the planner emits N tasks at RUNTIME, all workers run
    in ONE super-step, a reducer channel merges them.
  * subgraphs: shared parent state vs own schema + explicit translation.
  * Command.PARENT: a node INSIDE a subgraph routes the PARENT graph.
  * checkpoint namespaces: inspecting a finished subgraph's internal state.
  * delegation and context isolation: workers see only the delegated payload.

Run it:

    poetry run python lessons/l6_planning_and_subgraphs/main.py

Design promise: every demo counts executions DIRECTLY (a model-call journal
records what each fake model was handed and what it answered) and prints raw
runtime evidence (stream chunks, checkpoint namespaces, probed exception
text). Nothing is inferred from final state alone.
"""

from __future__ import annotations

import json
import operator
import sys
from pathlib import Path
from typing import Annotated, Callable, TypedDict

from langgraph.errors import InvalidUpdateError
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send, interrupt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lib import (  # noqa: E402
    banner,
    build_checkpointer,
    print_history,
    show_graph,
    thread,
    trace,
)


def say(node: str, message: str) -> None:
    """Node-side console output (flushed: parallel workers run on threads)."""
    print(f"    [{node}] {message}", flush=True)


# ---------------------------------------------------------------------------
# 1. Definitions: the task family, the model-call journal, the fakes
# ---------------------------------------------------------------------------

DOC_CHUNKS: list[dict] = [
    {"chunk_id": "etching-A1", "source_doc": "工艺规范A", "section": "蚀刻", "format": "text", "n_params": 6},
    {"chunk_id": "drilling-A2", "source_doc": "工艺规范A", "section": "钻孔", "format": "text", "n_params": 4},
    {"chunk_id": "plating-B1", "source_doc": "工艺规范B", "section": "电镀", "format": "text", "n_params": 5},
    {"chunk_id": "testing-B2", "source_doc": "工艺规范B", "section": "测试", "format": "text", "n_params": 7},
    {"chunk_id": "solder-C1", "source_doc": "工艺规范C", "section": "阻焊", "format": "text", "n_params": 3},
]
CHUNK_BY_ID = {c["chunk_id"]: c for c in DOC_CHUNKS}


def base_task() -> list[dict]:
    """The task every structure must solve identically: 4 plain text chunks."""
    return [dict(c) for c in DOC_CHUNKS[:4]]


def tricky_task() -> list[dict]:
    """The variant where planning pays: chunk 2 is a scanned TABLE IMAGE.

    Summarizing it requires a preprocessing step that no compile-time plan
    knew about. A fixed pipeline has no node for it and cannot recover; a
    Plan-and-Execute graph can replan.
    """
    chunks = [dict(c) for c in DOC_CHUNKS[:4]]
    chunks[1]["format"] = "table_image"
    return chunks


# The single source of truth for counting: EVERY fake-model invocation lands
# here with the exact context it was handed. Tests assert on this journal --
# "the worker never saw the parent's messages" is a journal lookup, not a guess.
MODEL_JOURNAL: list[dict] = []
TOOL_JOURNAL: list[tuple[str, str]] = []


def clear_journals() -> None:
    MODEL_JOURNAL.clear()
    TOOL_JOURNAL.clear()


def fake_summary(chunk: dict) -> str:
    """A deterministic 'model' reply: a pure function of the chunk."""
    return f"{chunk['chunk_id']} 摘要: {chunk['section']} 关键参数 {chunk['n_params']} 项"


def merge_report(results: list[dict], expected: int) -> str:
    """The merge 'model call': sorted so parallel fold order never matters."""
    ok = sorted(
        (r for r in results if r.get("ok") and r.get("summary")),
        key=lambda r: r["chunk_id"],
    )
    head = f"检查报告: {len(ok)}/{expected} 段摘要 " + ("完整" if len(ok) == expected else "不完整")
    return head + " -> " + " | ".join(r["summary"] for r in ok)


def summarize_envelope(chunk: dict, *, preprocessed: list[str] | None = None) -> dict:
    """One worker task result. Failures are ENVELOPES (L2 lesson), not raises."""
    if chunk["format"] == "table_image" and chunk["chunk_id"] not in (preprocessed or []):
        return {
            "task_id": None,
            "ok": False,
            "chunk_id": chunk["chunk_id"],
            "error": f"{chunk['chunk_id']} 是扫描表格图片，未经预处理无法摘要",
        }
    return {"task_id": None, "ok": True, "chunk_id": chunk["chunk_id"], "summary": fake_summary(chunk)}


def expected_report(chunks: list[dict]) -> str:
    """What a fully successful run must print -- the parity oracle."""
    return merge_report([summarize_envelope(c) for c in chunks], len(chunks))


class FakeModel:
    """Queue-scripted model for the single-agent loop (L2 shape).

    Also a context recorder: `respond` journals the full message list it was
    handed, which is how we MEASURE context duplication instead of hand-waving.
    """

    def __init__(self, script: list[dict], *, node: str) -> None:
        self.script = list(script)
        self.node = node
        self.consumed = 0

    def respond(self, messages: list[dict]) -> dict:
        MODEL_JOURNAL.append(
            {"node": self.node, "kind": "model", "context": [dict(m) for m in messages], "detail": ""}
        )
        self.consumed += 1
        if not self.script:
            raise RuntimeError("script ran dry -- a stop condition failed to fire")
        return self.script.pop(0)


def tool_call_msg(call_id: str, name: str, args: dict) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args, sort_keys=True)},
            }
        ],
    }


def final_msg(text: str) -> dict:
    return {"role": "assistant", "content": text}


def single_agent_script(chunks: list[dict]) -> list[dict]:
    """N tool rounds then one merged final answer, all in ONE conversation."""
    items = [
        {"message": tool_call_msg(f"c{i + 1}", "summarize_chunk", {"chunk_id": c["chunk_id"]})}
        for i, c in enumerate(chunks)
    ]
    items.append({"message": final_msg(expected_report(chunks))})
    return items


def tool_summarize_chunk(chunk_id: str) -> dict:
    """The single agent's ONLY tool. Executions land in TOOL_JOURNAL."""
    TOOL_JOURNAL.append(("summarize_chunk", chunk_id))
    return summarize_envelope(CHUNK_BY_ID[chunk_id])


def superstep_segments(graph, input_, config=None) -> list[list[str]]:
    """Group `updates` chunks by the `values` snapshots that follow them.

    Returns one list of node names per super-step (L2 fact: updates chunks
    count TASKS; the values snapshot between them is the super-step barrier).
    """
    segments: list[list[str]] = []
    current: list[str] = []
    for mode, chunk in graph.stream(input_, config, stream_mode=["updates", "values"]):
        if mode == "updates":
            current.extend(chunk.keys())
        else:
            if current:
                segments.append(current)
            current = []
    if current:
        segments.append(current)
    return segments


def final_step(graph, config) -> int:
    """The authoritative super-step count: checkpoint metadata.step."""
    return graph.get_state(config).metadata["step"]


def context_total(node_prefix: str = "") -> int:
    """Sum of context items handed to fake models -- the duplication meter."""
    return sum(len(r["context"]) for r in MODEL_JOURNAL if r["node"].startswith(node_prefix))


# ---------------------------------------------------------------------------
# 2. Structure (a): the single agent -- one loop, one growing context
# ---------------------------------------------------------------------------


class SingleAgentState(TypedDict):
    question: str
    messages: Annotated[list[dict], operator.add]
    steps: int
    stop_reason: str
    final_report: str


def build_single_agent_graph(respond: Callable[[list[dict]], dict], *, checkpointer=None):
    """One model node + one tools node, exactly the L2 loop shape.

    The agent walks through every chunk sequentially, carrying the whole
    conversation. That is the baseline every multi-agent claim must beat.
    """

    def model_node(state: SingleAgentState) -> dict:
        if state["steps"] >= 10:  # budget guard (L2) -- never reached by the script
            return {"stop_reason": "budget"}
        item = respond(state["messages"])
        message = item["message"]
        update: dict = {"messages": [message], "steps": state["steps"] + 1}
        if not message.get("tool_calls"):
            update["stop_reason"] = "success"
            update["final_report"] = message["content"]
        return update

    def tools_node(state: SingleAgentState) -> dict:
        calls = state["messages"][-1]["tool_calls"]
        observations = []
        for call in calls:
            args = json.loads(call["function"]["arguments"])
            envelope = tool_summarize_chunk(args["chunk_id"])
            observations.append(
                {"role": "tool", "tool_call_id": call["id"], "content": json.dumps(envelope, ensure_ascii=False)}
            )
        return {"messages": observations}

    builder = StateGraph(SingleAgentState)
    builder.add_node("model", model_node)
    builder.add_node("tools", tools_node)
    builder.add_edge(START, "model")
    builder.add_conditional_edges("model", lambda s: END if s["stop_reason"] else "tools", ["tools", END])
    builder.add_conditional_edges("tools", lambda s: "model", ["model"])
    return builder.compile(checkpointer=checkpointer) if checkpointer else builder.compile()


def run_single_agent(chunks: list[dict], *, checkpointer=None, config=None) -> tuple[dict, object]:
    fake = FakeModel(single_agent_script(chunks), node="single:model")
    graph = build_single_agent_graph(fake.respond, checkpointer=checkpointer)
    state = graph.invoke(
        {
            "question": "汇总全部工艺文档分段并输出检查报告",
            "messages": [{"role": "user", "content": "汇总全部工艺文档分段并输出检查报告"}],
            "steps": 0,
            "stop_reason": "",
            "final_report": "",
        },
        config,
    )
    return state, graph


# ---------------------------------------------------------------------------
# 3. Structure (b): the fixed pipeline -- the plan IS the graph
# ---------------------------------------------------------------------------


class FixedState(TypedDict):
    chunks: list[dict]
    results: Annotated[list[dict], operator.add]
    report: str


def build_fixed_pipeline_graph(*, chunks: list[dict] | None = None, checkpointer=None):
    """extract -> (summary_0 || ... || summary_{n-1}) -> merge.

    The fan-out is built at COMPILE time: N is the number of `add_node`
    calls, decided when the graph is constructed. Change the task at runtime
    and this graph is already wrong (see the tricky task). That is the price
    and the benefit: zero coordination calls, maximal parallelism, zero
    adaptivity.
    """
    chunks = chunks if chunks is not None else base_task()

    def make_summary_node(i: int, chunk: dict):
        def summary_node(state: FixedState) -> dict:
            # The fake model sees ONLY this chunk: context isolation for free.
            MODEL_JOURNAL.append(
                {
                    "node": f"summary_{i}",
                    "kind": "summarize",
                    "context": [{"chunk": chunk["chunk_id"], "format": chunk["format"]}],
                    "detail": chunk["section"],
                }
            )
            envelope = summarize_envelope(chunk)
            envelope["task_id"] = f"fixed-{i}"
            outcome = "ok" if envelope["ok"] else f"FAILED: {envelope['error']}"
            say(f"summary_{i}", f"summarizes {chunk['chunk_id']} (context: 1 chunk only) -> {outcome}")
            return {"results": [envelope]}

        return summary_node

    def extract(state: FixedState) -> dict:
        say("extract", f"compile-time plan: {len(chunks)} summary nodes were BUILT")
        return {"chunks": chunks}

    def merge(state: FixedState) -> dict:
        MODEL_JOURNAL.append(
            {"node": "merge", "kind": "merge", "context": [{"summaries": len(state["results"])}], "detail": ""}
        )
        return {"report": merge_report(state["results"], len(state["chunks"]))}

    builder = StateGraph(FixedState)
    builder.add_node("extract", extract)
    for i, chunk in enumerate(chunks):
        builder.add_node(f"summary_{i}", make_summary_node(i, chunk))
    builder.add_node("merge", merge)
    builder.add_edge(START, "extract")
    for i in range(len(chunks)):
        builder.add_edge("extract", f"summary_{i}")
        builder.add_edge(f"summary_{i}", "merge")
    builder.add_edge("merge", END)
    return builder.compile(checkpointer=checkpointer) if checkpointer else builder.compile()


# ---------------------------------------------------------------------------
# 4. Structure (c): planner + executor + replanner (Plan-and-Execute)
# ---------------------------------------------------------------------------

MAX_REPLANS = 1  # runtime budget: give up (report incomplete) instead of looping forever


class PlanState(TypedDict):
    chunks: list[dict]
    plan: list[dict]
    cursor: int
    results: Annotated[list[dict], operator.add]
    preprocessed: list[str]
    failure: str
    replans: int
    report: str


def scripted_plan(chunks: list[dict]) -> list[dict]:
    """The planner 'model call': one summarize task per chunk, in order."""
    return [
        {"task_id": f"t{i + 1}", "action": "summarize", "chunk_id": c["chunk_id"]}
        for i, c in enumerate(chunks)
    ]


def run_task(task: dict, state: PlanState, *, node: str) -> dict:
    """Execute ONE plan task with a fake model call; journal the context."""
    MODEL_JOURNAL.append(
        {
            "node": f"{node}:{task['task_id']}",
            "kind": task["action"],
            "context": [{"task": dict(task)}],
            "detail": "",
        }
    )
    say(f"{node}:{task['task_id']}", f"action={task['action']} chunk={task.get('chunk_id')}")
    if task["action"] == "preprocess":
        return {"task_id": task["task_id"], "ok": True, "preprocessed": task["chunk_id"]}
    # look the chunk up in THIS run's input, not in a global table: the tricky
    # scenario mutates chunk formats and the failure must be visible here
    chunk = next(c for c in state["chunks"] if c["chunk_id"] == task["chunk_id"])
    envelope = summarize_envelope(chunk, preprocessed=state.get("preprocessed") or [])
    envelope["task_id"] = task["task_id"]
    return envelope


def build_plan_execute_graph(*, checkpointer=None):
    """planner -> executor(loop) -> report, with replanner on task failure."""

    def planner(state: PlanState) -> dict:
        plan = scripted_plan(state["chunks"])
        MODEL_JOURNAL.append(
            {
                "node": "planner",
                "kind": "plan",
                "context": [{"docs": [c["chunk_id"] for c in state["chunks"]]}],
                "detail": "",
            }
        )
        say("planner", f"emits {len(plan)} tasks: {[t['task_id'] for t in plan]}")
        return {"plan": plan}

    def executor(state: PlanState) -> dict:
        task = state["plan"][state["cursor"]]
        envelope = run_task(task, state, node="executor")
        preprocessed = list(state.get("preprocessed") or [])
        if envelope.get("preprocessed"):
            preprocessed.append(envelope["preprocessed"])
        return {
            "results": [envelope],
            "cursor": state["cursor"] + 1,
            "preprocessed": preprocessed,
            "failure": "" if envelope["ok"] else task["task_id"],
        }

    def replanner(state: PlanState) -> dict:
        """Regenerate only the REMAINING plan around the failed task."""
        failed = state["results"][-1]
        # scripted 'replan model call': preprocess the bad chunk, then retry it
        tail = [
            {"task_id": f"{failed['task_id']}-prep", "action": "preprocess", "chunk_id": failed["chunk_id"]},
            {"task_id": f"{failed['task_id']}-retry", "action": "summarize", "chunk_id": failed["chunk_id"]},
        ]
        plan = state["plan"][: state["cursor"] - 1] + tail + state["plan"][state["cursor"] :]
        MODEL_JOURNAL.append(
            {
                "node": "replanner",
                "kind": "replan",
                "context": [
                    {
                        "failed": failed["task_id"],
                        "remaining": [t["task_id"] for t in state["plan"][state["cursor"] - 1 :]],
                    }
                ],
                "detail": "prep + retry",
            }
        )
        say("replanner", f"failed={failed['task_id']} -> new tail {[t['task_id'] for t in tail]}")
        return {"plan": plan, "cursor": state["cursor"] - 1, "failure": "", "replans": state["replans"] + 1}

    def route_after_executor(state: PlanState) -> str:
        if state["failure"]:
            # budget first: a second failure goes straight to the report (the L2
            # lesson: TERMINATION is the runtime's decision, not the model's)
            return "replanner" if state["replans"] < MAX_REPLANS else "report"
        return "report" if state["cursor"] >= len(state["plan"]) else "executor"

    def report(state: PlanState) -> dict:
        MODEL_JOURNAL.append(
            {"node": "report", "kind": "merge", "context": [{"results": len(state["results"])}], "detail": ""}
        )
        return {"report": merge_report(state["results"], len(state["chunks"]))}

    builder = StateGraph(PlanState)
    builder.add_node("planner", planner)
    builder.add_node("executor", executor)
    builder.add_node("replanner", replanner)
    builder.add_node("report", report)
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "executor")
    builder.add_conditional_edges("executor", route_after_executor, ["executor", "replanner", "report"])
    builder.add_edge("replanner", "executor")
    builder.add_edge("report", END)
    return builder.compile(checkpointer=checkpointer) if checkpointer else builder.compile()


def run_plan_execute(chunks: list[dict], *, checkpointer=None, config=None) -> tuple[dict, object]:
    graph = build_plan_execute_graph(checkpointer=checkpointer)
    state = graph.invoke(
        {
            "chunks": chunks,
            "plan": [],
            "cursor": 0,
            "results": [],
            "preprocessed": [],
            "failure": "",
            "replans": 0,
            "report": "",
        },
        config,
    )
    return state, graph


# ---------------------------------------------------------------------------
# 5. Send / map-reduce: runtime fan-out into ONE super-step
# ---------------------------------------------------------------------------


class MapReduceState(TypedDict):
    chunks: list[dict]
    tasks: list[dict]  # written once by the planner -- the runtime fan-out plan
    results: Annotated[list[dict], operator.add]
    report: str


class BrokenMapReduceState(TypedDict):
    """Same shape but `results` has NO reducer -- the Send collision probe."""

    chunks: list[dict]
    tasks: list[dict]
    results: list
    report: str


def build_map_reduce_graph(*, checkpointer=None, with_reducer: bool = True):
    """planner --Send--> worker (x N, one super-step) -> join.

    Probed on 1.2.11: each Send's payload is the ENTIRE input state for that
    worker invocation (the worker sees only {"task": ...}), and N Sends
    schedule N tasks inside ONE super-step -- the join barrier guarantees all
    results are folded before `join` runs.
    """
    state_cls = MapReduceState if with_reducer else BrokenMapReduceState

    def planner(state) -> dict:
        # Tasks carry their own chunk: the Send worker cannot see graph state,
        # so everything it needs must ride inside the payload (real delegation).
        tasks = [{"task_id": f"s{i + 1}", "chunk": dict(c)} for i, c in enumerate(state["chunks"])]
        MODEL_JOURNAL.append(
            {
                "node": "planner",
                "kind": "plan",
                "context": [{"docs": [c["chunk_id"] for c in state["chunks"]]}],
                "detail": "",
            }
        )
        say("planner", f"emits {len(tasks)} tasks at RUNTIME")
        return {"tasks": tasks}

    def fanout(state):
        # The conditional function returns Send objects: one per task. The
        # number of workers is decided HERE, at runtime, not at compile time.
        return [Send("worker", {"task": t}) for t in state["tasks"]]

    def worker(payload: dict) -> dict:
        # NOTE the parameter is the SEND PAYLOAD, not the graph state: this
        # worker physically cannot see `chunks`, `results`, or anything else.
        task = payload["task"]
        chunk = task["chunk"]
        MODEL_JOURNAL.append(
            {"node": f"worker:{chunk['chunk_id']}", "kind": "summarize", "context": [{"task": dict(task)}], "detail": ""}
        )
        envelope = summarize_envelope(chunk)
        envelope["task_id"] = task["task_id"]
        return {"results": [envelope]}

    def join(state) -> dict:
        MODEL_JOURNAL.append(
            {"node": "join", "kind": "merge", "context": [{"results": len(state["results"])}], "detail": ""}
        )
        return {"report": merge_report(state["results"], len(state["chunks"]))}

    builder = StateGraph(state_cls)
    builder.add_node("planner", planner)
    builder.add_node("worker", worker)
    builder.add_node("join", join)
    builder.add_edge(START, "planner")
    builder.add_conditional_edges("planner", fanout, ["worker"])
    builder.add_edge("worker", "join")
    builder.add_edge("join", END)
    return builder.compile(checkpointer=checkpointer) if checkpointer else builder.compile()


# ---------------------------------------------------------------------------
# 6. Subgraphs: shared parent state (A) vs own schema + translation (B)
# ---------------------------------------------------------------------------


class DocState(TypedDict):
    doc_id: str
    notes: Annotated[list[str], operator.add]


def build_shared_subgraph_graph(*, checkpointer=None, trim: bool = False):
    """Variant A: the subgraph uses the SAME schema as the parent.

    Probed on 1.2.11: the subgraph node's update to the parent is the
    subgraph's FINAL state (of the shared channels), folded through the
    PARENT's reducers. With `operator.add` that re-adds everything the
    subgraph RECEIVED -> duplication. The fix is input/output schemas on the
    subgraph (trim=True), which cut both directions of the boundary. But the
    deeper fact stands: in a shared schema there is no private scratch space.
    """

    def sub_read(state: DocState) -> dict:
        # Proof of shared visibility: this inner node reads the parent's key.
        say("sub.read", f"sees parent doc_id={state['doc_id']}")
        return {"notes": [f"sub read doc_id={state['doc_id']}"]}

    def sub_scratch(state: DocState) -> dict:
        # In a shared schema there is NOWHERE PRIVATE: this "internal" note
        # lands on the shared channel and leaks straight into parent state.
        return {"notes": ["sub scratch (meant to be internal)"]}

    if trim:

        class SubIn(TypedDict):
            doc_id: str  # do not hand `notes` to the subgraph at all

        class SubOut(TypedDict):
            notes: list[str]  # and only give back the notes it produced

        sub_builder = StateGraph(DocState, input_schema=SubIn, output_schema=SubOut)
    else:
        sub_builder = StateGraph(DocState)
    sub_builder.add_node("read", sub_read)
    sub_builder.add_node("scratch", sub_scratch)
    sub_builder.add_edge(START, "read")
    sub_builder.add_edge("read", "scratch")
    sub_builder.add_edge("scratch", END)
    sub = sub_builder.compile()

    builder = StateGraph(DocState)
    builder.add_node("intake", lambda s: {"notes": ["parent intake note"]})
    builder.add_node("worker", sub)  # compiled subgraph AS A NODE
    builder.add_node("done", lambda s: {"notes": ["parent done note"]})
    builder.add_edge(START, "intake")
    builder.add_edge("intake", "worker")
    builder.add_edge("worker", "done")
    builder.add_edge("done", END)
    return builder.compile(checkpointer=checkpointer) if checkpointer else builder.compile()


class SubWorkState(TypedDict):
    """Variant B: the worker subgraph has its OWN schema, nothing shared."""

    payload: str
    scratch: Annotated[list[str], operator.add]


def build_worker_subgraph():
    def inner(state: SubWorkState) -> dict:
        # The fake model journals the EXACT context it received: only `payload`.
        MODEL_JOURNAL.append(
            {"node": "sub.inner", "kind": "summarize", "context": [{"payload": state["payload"]}], "detail": ""}
        )
        say("sub.inner", f"context keys it received: {sorted(state.keys())}")
        return {"scratch": [f"inner processed {state['payload']}"]}

    builder = StateGraph(SubWorkState)
    builder.add_node("inner", inner)
    builder.add_edge(START, "inner")
    builder.add_edge("inner", END)
    return builder.compile()


WORKER_SUBGRAPH = build_worker_subgraph()


def build_translated_subgraph_graph(*, checkpointer=None):
    """Variant B: explicit 状态翻译 in a wrapper node -- the boundary is CODE.

    Parent schema in, translated payload into the subgraph's schema, explicit
    translate-back of exactly one field. `scratch` physically cannot leak:
    the parent graph has no such channel. (Probed: payload keys that are not
    in the subgraph schema are DROPPED on the way in -- isolation by schema.)
    """

    def translated_worker(state: DocState) -> dict:
        # TRANSLATE IN: wrap what the parent knows into the subgraph schema.
        sub_input: SubWorkState = {"payload": f"translate(doc_id={state['doc_id']})", "scratch": []}
        sub_out = WORKER_SUBGRAPH.invoke(sub_input)
        # TRANSLATE BACK: only the one field we chose crosses the boundary.
        return {"notes": [f"translated back: {sub_out['scratch'][0]}"]}

    builder = StateGraph(DocState)
    builder.add_node("intake", lambda s: {"notes": ["parent intake note"]})
    builder.add_node("worker", translated_worker)
    builder.add_node("done", lambda s: {"notes": ["parent done note"]})
    builder.add_edge(START, "intake")
    builder.add_edge("intake", "worker")
    builder.add_edge("worker", "done")
    builder.add_edge("done", END)
    return builder.compile(checkpointer=checkpointer) if checkpointer else builder.compile()


# ---------------------------------------------------------------------------
# 7. Command.PARENT: a subgraph node routes the PARENT graph
# ---------------------------------------------------------------------------


class EscalateState(TypedDict):
    doc_id: str
    trail: Annotated[list[str], operator.add]
    verdict: str


URGENT_DOC = "URGENT-01"


def build_parent_command_graph(*, checkpointer=None, keep_static_exit: bool = True):
    """A review subgraph escalates urgent docs straight to the parent.

    Probed on 1.2.11:
      * Command(goto=..., update=..., graph=Command.PARENT) routes the PARENT
        and SKIPS the rest of the subgraph (inner_tail never runs).
      * It works with or without a checkpointer.
      * It does NOT suppress the parent-side static edge leaving the subgraph
        node: goto is ADDITIVE, both destinations run in one super-step (the
        L2 Command/static-edge pitfall, subgraph edition).
    """

    def inner_check(state: EscalateState) -> Command | dict:
        if state["doc_id"] == URGENT_DOC:
            say("sub.inner_check", "URGENT -> Command(graph=Command.PARENT, goto='escalate')")
            return Command(
                goto="escalate",
                update={"trail": ["inner_check fired Command.PARENT"], "verdict": "escalated"},
                graph=Command.PARENT,
            )
        # The contrast: a normal update stays INSIDE the subgraph.
        say("sub.inner_check", "normal doc -> plain update, subgraph continues")
        return {"trail": ["inner_check normal update"]}

    def inner_tail(state: EscalateState) -> dict:
        say("sub.inner_tail", "ran (the part Command.PARENT skips)")
        return {"trail": ["inner_tail ran"]}

    # Trim the boundary (demo 3's lesson applied): the subgraph receives only
    # doc_id and returns only what it produced, so shared-reducer duplication
    # (the demo-3 phenomenon) does not muddy this demo's trail.
    class SubIn(TypedDict):
        doc_id: str

    class SubOut(TypedDict):
        trail: list[str]
        verdict: str

    sub_builder = StateGraph(EscalateState, input_schema=SubIn, output_schema=SubOut)
    sub_builder.add_node("inner_check", inner_check)
    sub_builder.add_node("inner_tail", inner_tail)
    sub_builder.add_edge(START, "inner_check")
    sub_builder.add_edge("inner_check", "inner_tail")
    sub_builder.add_edge("inner_tail", END)
    sub = sub_builder.compile()

    builder = StateGraph(EscalateState)
    builder.add_node("intake", lambda s: {"trail": ["intake ran"]})
    builder.add_node("review", sub)
    builder.add_node("escalate", lambda s: {"trail": ["escalate ran (parent node)"]})
    builder.add_node("report", lambda s: {"trail": ["report ran"]})
    builder.add_edge(START, "intake")
    builder.add_edge("intake", "review")
    if keep_static_exit:
        builder.add_node("normal_next", lambda s: {"trail": ["normal_next ran (static edge)"]})
        builder.add_edge("review", "normal_next")
        builder.add_edge("normal_next", END)
    builder.add_edge("escalate", "report")
    builder.add_edge("report", END)
    return builder.compile(checkpointer=checkpointer) if checkpointer else builder.compile()


# ---------------------------------------------------------------------------
# 8. Checkpoint namespaces: inspecting a subgraph's internal state
# ---------------------------------------------------------------------------


class NSParent(TypedDict):
    doc_id: str
    notes: Annotated[list[str], operator.add]


class NSSub(TypedDict):
    doc_id: str  # shared BY NAME -> flows in from the parent
    notes: Annotated[list[str], operator.add]  # shared BY NAME -> flows back out
    inner_log: Annotated[list[str], operator.add]  # sub-private: stays in the namespace


def build_ns_demo_graph(*, checkpointer=None, interrupt_inside: bool = False):
    """A worker subgraph with TWO inner steps, added as a node.

    Probed on 1.2.11 (this corrects the common doc-summary claim):
      * After the run COMPLETES, get_state(config, subgraphs=True) shows an
        EMPTY tasks tuple -- nested states are only surfaced there while a
        task is PENDING (e.g. inside an interrupt).
      * The finished subgraph IS inspectable via its checkpoint namespace:
        ns = "worker:<uuid>" (one uuid per subgraph instance), addressed by
        get_state({"configurable": {"thread_id": t, "checkpoint_ns": ns}}).
        It has its OWN values, its OWN step counter, its OWN history.
      * get_state_history() does not accept subgraphs= at all on 1.2.11.
    """

    def ns_step_one(state: NSSub) -> dict:
        if interrupt_inside:
            # interrupt() used purely as a debugging-visibility tool: it leaves
            # the subgraph task PENDING so subgraphs=True exposes the nested
            # snapshot. (interrupt itself is L4's subject.)
            answer = interrupt({"approval_needed": state["doc_id"]})
            say("sub.ns_step_one", f"resumed with answer={answer!r}")
            return {"notes": [f"ns step one resumed ({answer})"], "inner_log": ["step one after resume"]}
        say("sub.ns_step_one", "writes notes + inner_log")
        return {"notes": ["ns sub note"], "inner_log": ["step one"]}

    def ns_step_two(state: NSSub) -> dict:
        return {"inner_log": ["step two"]}

    # Trimmed boundary (demo 3's lesson): receives only doc_id, returns only
    # the notes it produced -- keeps this demo about namespaces, not reducers.
    class NSIn(TypedDict):
        doc_id: str

    class NSOut(TypedDict):
        notes: list[str]
        inner_log: list[str]

    sub_builder = StateGraph(NSSub, input_schema=NSIn, output_schema=NSOut)
    sub_builder.add_node("ns_step_one", ns_step_one)
    sub_builder.add_node("ns_step_two", ns_step_two)
    sub_builder.add_edge(START, "ns_step_one")
    sub_builder.add_edge("ns_step_one", "ns_step_two")
    sub_builder.add_edge("ns_step_two", END)
    sub = sub_builder.compile()

    builder = StateGraph(NSParent)
    builder.add_node("intake", lambda s: {"notes": ["parent intake note"]})
    builder.add_node("worker", sub)
    builder.add_node("done", lambda s: {"notes": ["parent done note"]})
    builder.add_edge(START, "intake")
    builder.add_edge("intake", "worker")
    builder.add_edge("worker", "done")
    builder.add_edge("done", END)
    return builder.compile(checkpointer=checkpointer) if checkpointer else builder.compile()


def discover_namespaces(graph, input_, config) -> list[str]:
    """Collect subgraph namespaces from stream(subgraphs=True) chunks."""
    seen: list[str] = []
    for ns, _payload in graph.stream(input_, config, stream_mode="updates", subgraphs=True):
        if ns and ns[0] not in seen:
            seen.append(ns[0])
    return seen


# ---------------------------------------------------------------------------
# 9. Delegation and context isolation: the supervisor pattern
# ---------------------------------------------------------------------------

PARENT_HISTORY = [
    {"role": "user", "content": "请整理全部工艺文档并输出检查报告"},
    {"role": "assistant", "content": "收到，我先按章节分工，再汇总"},
]
HISTORY_MARKER = PARENT_HISTORY[0]["content"]  # what must NEVER reach a worker


class TeamState(TypedDict):
    request: str
    history: Annotated[list[dict], operator.add]
    outcomes: Annotated[list[dict], operator.add]


class WorkerState(TypedDict):
    task: dict
    out: Annotated[list[dict], operator.add]


def build_delegation_graph(*, checkpointer=None, include_history: bool = False, include_hint: bool = False):
    """A supervisor delegates to two workers via minimal task payloads.

    `include_history` is the anti-pattern lever: the supervisor stuffs the
    parent conversation INTO the delegated task (payload-level extra keys are
    dropped by the subgraph input schema -- probed), so the difference is
    directly countable. `include_hint` shows the price of isolation: a task
    that NEEDS sibling context fails without an explicit hint.
    """

    def make_worker_inner(name: str):
        def inner(state: WorkerState) -> dict:
            task = state["task"]
            # The fake worker model journals the EXACT task it was handed.
            MODEL_JOURNAL.append(
                {"node": f"{name}.run", "kind": "worker", "context": [dict(task)], "detail": ""}
            )
            say(f"{name}.run", f"task keys received: {sorted(task.keys())}")
            if "needs_summary_of" in task:
                if "sibling_summary" not in task:
                    return {
                        "out": [
                            {
                                "worker": name,
                                "ok": False,
                                "error": (
                                    f"需要 {task['needs_summary_of']} 的摘要，但委派载荷里没有——"
                                    "上下文隔离把它挡在了外面"
                                ),
                            }
                        ]
                    }
                return {
                    "out": [
                        {"worker": name, "ok": True, "summary": f"{task['chunk_id']} 对照 {task['needs_summary_of']}: 一致"}
                    ]
                }
            chunk = CHUNK_BY_ID[task["chunk_id"]]
            return {"out": [{"worker": name, "ok": True, "summary": fake_summary(chunk)}]}

        return inner

    def make_worker_node(name: str, task: dict):
        worker_sub = StateGraph(WorkerState)
        worker_sub.add_node("run", make_worker_inner(name))
        worker_sub.add_edge(START, "run")
        worker_sub.add_edge("run", END)
        worker_compiled = worker_sub.compile()

        def worker_node(state: TeamState) -> dict:
            delegated = dict(task)
            if include_history:
                # the anti-pattern: the whole parent conversation rides along
                delegated["history"] = [dict(m) for m in state["history"]]
            if include_hint and "needs_summary_of" in task:
                sibling = CHUNK_BY_ID[task["needs_summary_of"]]
                delegated["sibling_summary"] = fake_summary(sibling)
            out = worker_compiled.invoke({"task": delegated, "out": []})
            return {"outcomes": out["out"]}

        return worker_node

    def supervisor(state: TeamState) -> dict:
        context = [{"request": state["request"]}] + [dict(m) for m in state["history"]]
        MODEL_JOURNAL.append({"node": "supervisor", "kind": "delegate", "context": context, "detail": ""})
        say(
            "supervisor",
            f"delegates minimal payloads (include_history={include_history}, include_hint={include_hint})",
        )
        return {}  # the delegation itself travels through the worker payloads

    summarize_task = {"kind": "summarize", "chunk_id": "etching-A1"}
    cross_task = {"kind": "cross_check", "chunk_id": "testing-B2", "needs_summary_of": "solder-C1"}

    builder = StateGraph(TeamState)
    builder.add_node("supervisor", supervisor)
    builder.add_node("w_a", make_worker_node("w_a", summarize_task))
    builder.add_node("w_b", make_worker_node("w_b", cross_task))
    builder.add_node("collect", lambda s: {})
    builder.add_edge(START, "supervisor")
    builder.add_edge("supervisor", "w_a")
    builder.add_edge("supervisor", "w_b")
    builder.add_edge("w_a", "collect")
    builder.add_edge("w_b", "collect")
    builder.add_edge("collect", END)
    return builder.compile(checkpointer=checkpointer) if checkpointer else builder.compile()


def run_delegation(*, include_history: bool = False, include_hint: bool = False) -> dict:
    graph = build_delegation_graph(include_history=include_history, include_hint=include_hint)
    return graph.invoke({"request": "分工整理并交叉检查", "history": list(PARENT_HISTORY), "outcomes": []})


def worker_contexts() -> list[dict]:
    """Journal entries of fake worker models, in call order."""
    return [r for r in MODEL_JOURNAL if r["node"].endswith(".run")]


# ---------------------------------------------------------------------------
# 10. Demos
# ---------------------------------------------------------------------------


def demo_structure_comparison():
    banner("DEMO 1 - 三种结构同一任务：单 Agent / 固定流程 / Plan-and-Execute")
    chunks = base_task()
    print(f"  任务: {len(chunks)} 段工艺文档 -> 分段摘要 -> 检查报告\n")

    clear_journals()
    cp_a = build_checkpointer("memory")
    cfg_a = thread("l6-demo1a")
    state_a, graph_a = run_single_agent(chunks, checkpointer=cp_a, config=cfg_a)
    calls_a, tools_a = len(MODEL_JOURNAL), len(TOOL_JOURNAL)
    steps_a, ctx_a = final_step(graph_a, cfg_a), context_total("single")

    clear_journals()
    cp_b = build_checkpointer("memory")
    cfg_b = thread("l6-demo1b")
    graph_b = build_fixed_pipeline_graph(chunks=chunks, checkpointer=cp_b)
    state_b = graph_b.invoke({"chunks": [], "results": [], "report": ""}, cfg_b)
    calls_b, steps_b, ctx_b = len(MODEL_JOURNAL), final_step(graph_b, cfg_b), context_total()

    clear_journals()
    cp_c = build_checkpointer("memory")
    cfg_c = thread("l6-demo1c")
    state_c, graph_c = run_plan_execute(chunks, checkpointer=cp_c, config=cfg_c)
    calls_c, steps_c, ctx_c = len(MODEL_JOURNAL), final_step(graph_c, cfg_c), context_total()

    print("  结构            model调用  super-steps  传入上下文条目  报告一致")
    print(f"  (a) 单Agent     {calls_a}         {steps_a}            {ctx_a}             {state_a['final_report'] == state_b['report']}")
    print(f"  (b) 固定流程    {calls_b}         {steps_b}             {ctx_b}             -")
    print(f"  (c) Plan-Exec   {calls_c}         {steps_c}            {ctx_c}             {state_c['report'] == state_b['report']}")
    print(f"  (a) 工具执行 {tools_a} 次; 报告: {state_a['final_report']}")
    print("\n  直接读数:")
    print("  * (a) 逐轮重复传入全部历史 -> 25 条消息；单上下文的隐性成本随段数平方增长。")
    print("  * (b) 编译期扇出: 4 个摘要节点同一 super-step 并行，每次调用只见 1 段。")
    print("  * (c) 多付 1 次 planner 调用、串行执行 -> super-steps 最多。协调本身是开销。")

    print("\n  原始证据（(b) 的一次运行，updates chunk + values 快照）:")
    trace(
        build_fixed_pipeline_graph(chunks=chunks),
        {"chunks": [], "results": [], "report": ""},
        label="(b) 固定流程: 4 个摘要节点夹在两个 values 快照之间 = 同一 super-step",
    )

    print("\n  变体任务: 第 2 段是扫描表格图片（需要计划外的预处理步骤）\n")
    tricky = tricky_task()

    clear_journals()
    state_b2 = build_fixed_pipeline_graph(chunks=tricky).invoke({"chunks": [], "results": [], "report": ""})
    calls_b2 = len(MODEL_JOURNAL)
    clear_journals()
    state_c2, _ = run_plan_execute(tricky)
    calls_c2 = len(MODEL_JOURNAL)
    print(f"  (b) 固定流程 ({calls_b2} 调用)  -> {state_b2['report']}")
    print(f"  (c) Plan-Exec ({calls_c2} 调用) -> {state_c2['report']}")
    print(f"      (c) replans={state_c2['replans']}, 报告完整={'4/4' in state_c2['report']}")
    print("\n  结论（协作何时无收益 / 何时有收益）:")
    print("  * 任务固定、可整段拆分、无失败 -> (b) 最便宜: 0 次协调调用、天然并行、天然隔离。")
    print("  * 步骤集合运行时才可知、或需要失败后改计划 -> 只有 (c) 能赢，代价是多付协调调用。")
    print("  * 段数少且一次性 -> (a) 可接受，其真实成本是上下文逐轮膨胀（25 条消息）。")


def demo_send_map_reduce():
    banner("DEMO 2 - Send/map-reduce：运行时扇出，N 个 worker 同一个 super-step")
    show_graph(build_map_reduce_graph())
    for label, chunks in (("N=3", DOC_CHUNKS[:3]), ("N=5", DOC_CHUNKS[:5])):
        clear_journals()
        cp = build_checkpointer("memory")
        cfg = thread(f"l6-demo2-{label}")
        graph = build_map_reduce_graph(checkpointer=cp)
        segments = superstep_segments(graph, {"chunks": [dict(c) for c in chunks], "results": [], "report": ""}, cfg)
        state = graph.get_state(cfg).values
        per_task: dict[str, int] = {}
        for r in MODEL_JOURNAL:
            if r["node"].startswith("worker:"):
                key = r["context"][0]["task"]["chunk"]["chunk_id"]
                per_task[key] = per_task.get(key, 0) + 1
        print(f"  {label}: super-step 划分 {segments}")
        print(f"      final metadata.step={final_step(graph, cfg)}  每任务执行次数 {per_task}")
        print(f"      join 收到 {len(state['results'])} 份结果 -> {state['report']}")
        assert all(v == 1 for v in per_task.values()), "every task must run EXACTLY once"
        assert [len(s) for s in segments] == [1, len(chunks), 1]

    print("\n  反例: 去掉 results 的 Annotated reducer，N 个 Send 写同一 channel")
    try:
        build_map_reduce_graph(with_reducer=False).invoke(
            {"chunks": [dict(c) for c in DOC_CHUNKS[:3]], "results": [], "report": ""}
        )
    except InvalidUpdateError as exc:
        print(f"    raised as probed:\n      {exc}")
        print("    与 L1 的并发写冲突是同一条规则；Send 让并发发生在同节点的多个任务之间。")


def demo_subgraph_states():
    banner("DEMO 3 - 子图状态: 共享 schema (A) vs 独立 schema + 状态翻译 (B)")
    print("  [A] 子图与父图共用 DocState:")
    naive = build_shared_subgraph_graph().invoke({"doc_id": "DOC-1", "notes": []})
    print(f"      naive 出口 notes: {naive['notes']}")
    print(f"      'parent intake note' 出现 {naive['notes'].count('parent intake note')} 次")
    print("      -> 子图最终 state 经父图 reducer 重放，收到的又被加回去（probed）")
    trimmed = build_shared_subgraph_graph(trim=True).invoke({"doc_id": "DOC-1", "notes": []})
    print(f"      trim 后 notes: {trimmed['notes']}")
    print(
        f"      'parent intake note' 出现 {trimmed['notes'].count('parent intake note')} 次;"
        f" 但泄漏仍在: {'sub scratch' in str(trimmed['notes'])} (共享 schema 没有私有空间)"
    )
    print("\n  [B] 子图自己的 schema，包装节点做显式翻译:")
    clear_journals()
    translated = build_translated_subgraph_graph().invoke({"doc_id": "DOC-2", "notes": []})
    inner_ctx = next(r["context"] for r in MODEL_JOURNAL if r["node"] == "sub.inner")
    print(f"      父图终态 keys: {sorted(translated.keys())}  (scratch 不存在: {'scratch' not in translated})")
    print(f"      子图模型收到的 context: {inner_ctx}")
    print(f"      翻译回写: {[n for n in translated['notes'] if n.startswith('translated back')]}")


def demo_command_parent():
    banner("DEMO 4 - Command.PARENT: 子图内部节点直接路由父图")
    print("  对照组: 普通文档 -> 普通更新留在子图内部，父图沿静态边走 normal_next\n")
    normal = build_parent_command_graph().invoke({"doc_id": "NORMAL-9", "trail": [], "verdict": ""})
    print(f"    trail: {normal['trail']}")
    print(f"    verdict: {normal['verdict']!r} (未升级)\n")

    print("  Command.PARENT: URGENT 文档，且 builder 不留静态出边\n")
    clean = build_parent_command_graph(keep_static_exit=False).invoke({"doc_id": URGENT_DOC, "trail": [], "verdict": ""})
    print(f"    trail: {clean['trail']}")
    print(f"    verdict: {clean['verdict']!r}")
    print("    inner_tail 被跳过（Command.PARENT 中断子图剩余路径），父图直达 escalate。\n")

    print("  反例: 同样的图但保留 review -> normal_next 静态边\n")
    additive = build_parent_command_graph(keep_static_exit=True).invoke({"doc_id": URGENT_DOC, "trail": [], "verdict": ""})
    print(f"    trail: {additive['trail']}")
    print("    goto 是叠加不是替换: normal_next 与 escalate 都执行（L2 反例的子图版）。")
    print("    修复: Command.PARENT 节点在父图侧不挂静态出边。")


def demo_namespaces():
    banner("DEMO 5 - checkpoint 命名空间与子图状态检查")
    cp = build_checkpointer("memory")
    cfg = thread("l6-demo5")
    graph = build_ns_demo_graph(checkpointer=cp)
    graph.invoke({"doc_id": "DOC-5", "notes": []}, cfg)

    snap = graph.get_state(cfg, subgraphs=True)
    print(f"  完成后 get_state(subgraphs=True): tasks={snap.tasks}  <-- 空! 嵌套状态不会在这里出现")

    ns_list = discover_namespaces(graph, {"doc_id": "DOC-5b", "notes": []}, thread("l6-demo5b"))
    print(f"  stream(subgraphs=True) 发现的命名空间: {ns_list}")
    child_cfg = {"configurable": {"thread_id": "l6-demo5b", "checkpoint_ns": ns_list[0]}}
    child = graph.get_state(child_cfg)
    print(f"  按命名空间取子图终态: values={child.values}")
    print(f"      子图自己的 step={child.metadata.get('step')}  parents={child.metadata.get('parents')}")
    print_history(graph, child_cfg)

    print("\n  子图内部 interrupt -> 任务 PENDING 时 subgraphs=True 才显示嵌套快照:")
    cfg2 = thread("l6-demo5-int")
    graph2 = build_ns_demo_graph(checkpointer=build_checkpointer("memory"), interrupt_inside=True)
    graph2.invoke({"doc_id": "DOC-6", "notes": []}, cfg2)
    snap2 = graph2.get_state(cfg2, subgraphs=True)
    print(f"      parent next={snap2.next}")
    for t in snap2.tasks:
        nested = t.state
        print(f"      tasks[0].state.values = {nested.values}")
        print(f"      tasks[0].state.next   = {nested.next}")
        print(f"      nested interrupts     = {[i.value for nt in nested.tasks for i in (nt.interrupts or ())]}")
    resumed = graph2.invoke(Command(resume="approved"), cfg2)
    print(f"      resume 后父图终态 notes: {resumed['notes']}")
    print("\n  这是调试可见性，不是跨会话记忆: Store 解决的是跨 thread 记忆（DESIGN.md §7 校正行），")
    print("  与子图状态可见性是两件事；本课自始至终没有用到 Store。")


def demo_delegation_isolation():
    banner("DEMO 6 - 委派与上下文隔离: worker 只见任务载荷")
    clear_journals()
    isolated = run_delegation()
    ctxs = worker_contexts()
    leaked = any(HISTORY_MARKER in str(r["context"]) for r in ctxs)
    print(f"  隔离委派: outcomes={[(o['worker'], o['ok']) for o in isolated['outcomes']]}")
    print(f"      每个收到的 task 字段数: {[len(r['context'][0]) for r in ctxs]}")
    print(f"      worker context 含父会话内容: {leaked}")
    print(f"      交叉检查任务(需兄弟摘要, 无 hint) 失败: {[o['error'][:34] + '...' for o in isolated['outcomes'] if not o['ok']]}\n")

    clear_journals()
    with_history = run_delegation(include_history=True)
    ctxs2 = worker_contexts()
    leaked2 = any(HISTORY_MARKER in str(r["context"]) for r in ctxs2)
    print(f"  反模式 include_history=True: task 字段数 {[len(r['context'][0]) for r in ctxs2]}, 父会话内容可见={leaked2}")
    print("      上下文被复制进每个 worker 的载荷: 隔离收益消失，且没有换来任何新能力。\n")

    clear_journals()
    hinted = run_delegation(include_hint=True)
    print(f"  include_hint=True: supervisor 把兄弟摘要放进载荷 -> outcomes={[(o['worker'], o['ok']) for o in hinted['outcomes']]}")
    print("      隔离的代价: 需要兄弟上下文的任务必须由 supervisor 显式携带 hint，否则 worker 必败。")


def demo_replanning():
    banner("DEMO 7 - 任务失败后的重规划: 已完成任务不重跑")
    tricky = tricky_task()
    clear_journals()
    state, _ = run_plan_execute(tricky)
    counts: dict[str, int] = {}
    for r in MODEL_JOURNAL:
        if r["node"].startswith("executor:"):
            key = r["node"].split(":", 1)[1]
        elif r["node"] == "replanner":
            key = "(replan 调用)"
        else:
            continue
        counts[key] = counts.get(key, 0) + 1
    print(f"  执行/重规划调用计数: {counts}")
    print(f"  replans={state['replans']}  preprocessed={state['preprocessed']}")
    print(f"  报告: {state['report']}")
    assert counts["t1"] == 1 and counts["t3"] == 1 and counts["t4"] == 1, "completed tasks must NOT re-run"
    assert counts["t2"] == 1 and counts["t2-prep"] == 1 and counts["t2-retry"] == 1
    print("  t1/t3/t4 各执行 1 次（未重跑）；t2 首次失败 1 次 + 预处理 1 次 + 重试成功 1 次；最终 4/4 完整。")


if __name__ == "__main__":
    demo_structure_comparison()
    demo_send_map_reduce()
    demo_subgraph_states()
    demo_command_parent()
    demo_namespaces()
    demo_delegation_isolation()
    demo_replanning()
    banner("done", width=72)
