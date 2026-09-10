"""Shared teaching helpers for the LangGraph curriculum.

Everything in here exists to make *runtime behaviour visible*: which super-step
is running, what each node wrote, how the checkpoint stream evolves. The lessons
deliberately avoid magic -- if a helper hides something, it prints it instead.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Iterable, Literal, Mapping

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

StreamMode = Literal["updates", "values", "messages", "tasks", "debug", "custom"]

BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
RESET = "\033[0m"


def banner(title: str, width: int = 72) -> None:
    """Print a section header so lesson output stays readable."""
    print(f"\n{BOLD}{CYAN}{'=' * width}{RESET}")
    print(f"{BOLD}{CYAN}{title}{RESET}")
    print(f"{BOLD}{CYAN}{'=' * width}{RESET}")


def step(n: int, label: str) -> None:
    """Annotate a super-step boundary in the output."""
    print(f"\n{BOLD}{YELLOW}-- super-step {n}: {label} --{RESET}")


def _short(value: Any, limit: int = 160) -> str:
    text = repr(value)
    return text if len(text) <= limit else f"{text[:limit]}..."


def trace(
    graph: Any,
    input_: Any,
    config: Mapping[str, Any] | None = None,
    *,
    stream_mode: StreamMode = "updates",
    durability: str | None = None,
    subgraphs: bool = False,
    label: str = "run",
) -> Any:
    """Run a graph, printing every chunk, and return the final state.

    This is the workhorse for the lessons. It prints the raw chunks that
    ``graph.stream`` yields -- untouched -- so you can see exactly what the
    runtime emits per super-step instead of trusting a summary.
    """
    banner(f"trace: {label}")
    last_values: Any = None
    kwargs: dict[str, Any] = {"stream_mode": ["updates", "values"], "subgraphs": subgraphs}
    if durability is not None:
        kwargs["durability"] = durability

    for mode, chunk in graph.stream(input_, config, **kwargs):
        if mode == "updates":
            if chunk == {}:
                # An empty update dict is meaningful: it marks a step where no
                # node produced output (e.g. the resume of an interrupt).
                print(f"  {DIM}updates: {{}} (no writes this step){RESET}")
                continue
            for node, payload in chunk.items():
                if node == "__interrupt__":
                    print(f"  {RED}INTERRUPT -> {_short(payload)}{RESET}")
                else:
                    print(f"  {GREEN}{node}{RESET} wrote {_short(payload)}")
        else:
            last_values = chunk
            print(f"  {DIM}state: {_short(chunk)}{RESET}")

    return last_values


def print_state(graph: Any, config: Mapping[str, Any]) -> Any:
    """Print the current (latest) state plus a few facts that matter for debugging."""
    snapshot = graph.get_state(config)
    banner("current state")
    print(f"  values        : {_short(snapshot.values)}")
    print(f"  next          : {snapshot.next}")
    print(f"  created_at    : {getattr(snapshot, 'created_at', None)}")
    print(f"  step (version): {snapshot.metadata.get('step') if snapshot.metadata else None}")
    print(f"  source        : {snapshot.metadata.get('source') if snapshot.metadata else None}")
    print(f"  checkpoint_id : {snapshot.config.get('configurable', {}).get('checkpoint_id')}")
    if snapshot.tasks:
        for task in snapshot.tasks:
            print(f"  pending task  : {task.name} (id={task.id})")
    return snapshot


def print_history(graph: Any, config: Mapping[str, Any], limit: int | None = None) -> list[Any]:
    """Print checkpoint history newest-first -- this is what time travel walks."""
    banner("checkpoint history")
    history = list(graph.get_state_history(config))
    if limit is not None:
        history = history[:limit]
    for i, snap in enumerate(history):
        meta = snap.metadata or {}
        writes = {k: v for k, v in snap.values.items() if k != "messages"}
        print(
            f"  [{i:>2}] step={meta.get('step'):<3} source={meta.get('source'):<9} "
            f"next={snap.next} checkpoint={snap.config.get('configurable', {}).get('checkpoint_id')}"
        )
        print(f"       values={_short(writes, 110)}")
    return history


def build_checkpointer(kind: Literal["memory", "sqlite"] = "memory", path: str = "checkpoints.sqlite") -> BaseCheckpointSaver:
    """Return a checkpointer.

    ``memory`` dies with the process (fine for lessons L1-L2), ``sqlite`` survives
    a process restart -- which is the whole point of L3's crash-recovery exercise.
    """
    if kind == "memory":
        return InMemorySaver()
    if kind == "sqlite":
        conn = sqlite3.connect(path, check_same_thread=False)
        from langgraph.checkpoint.sqlite import SqliteSaver

        saver = SqliteSaver(conn)
        saver.setup()
        return saver
    raise ValueError(f"unknown checkpointer kind: {kind}")


def thread(thread_id: str, **extra: Any) -> dict[str, Any]:
    """Build a RunnableConfig whose thread_id is the recovery key."""
    configurable = {"thread_id": thread_id}
    configurable.update(extra)
    return {"configurable": configurable}


def show_graph(graph: Any) -> None:
    """Print the compiled Mermaid source of a graph (paste into mermaid.live)."""
    banner("graph (mermaid)")
    print(graph.get_graph().draw_mermaid())


def summarize(events: Iterable[Any]) -> None:
    """Print stream_events output in a compact, per-node form."""
    banner("event stream")
    for event in events:
        kind = event.get("event", "?")
        name = event.get("name", "")
        print(f"  {kind:<28} {name}")
