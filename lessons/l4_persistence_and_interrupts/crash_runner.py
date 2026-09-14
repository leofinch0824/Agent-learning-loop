"""L4 crash fixture: a child process that runs a graph and can be SIGKILLed mid-node.

This is NOT a lesson demo -- it is the victim process for main.py DEMO 6/7 and
the subprocess tests. The parent spawns it, polls the ready-file, then SIGKILLs
it; a second invocation in "resume" mode reopens the SAME sqlite checkpointer
file and thread_id ("pcb-42"), proving recovery across a process boundary.

Graph (all effects land in the external log file):  propose -> apply -> report

Crash points (where the child writes the ready-file then sleeps):
    before_write      inside apply, BEFORE the WRITE line lands
    after_write       inside apply, AFTER the WRITE but BEFORE the node returns
                      -> no checkpoint for this super-step (the failure window)
    after_checkpoint  inside report, AFTER apply's checkpoint committed

Usage (positional argv):   run|resume SQLITE LOG READY [POINT] [LEDGER] [DURABILITY]
LEDGER: JSON file of "thread:task_id" keys; apply skips the WRITE when its key
is already marked done (the idempotency fix).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Annotated, TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import operator  # noqa: E402

from langgraph.graph import END, START, StateGraph  # noqa: E402
from langgraph.types import RunnableConfig  # noqa: E402

from lib import build_checkpointer, thread  # noqa: E402

MODE, SQLITE, LOG, READY = sys.argv[1:5]
POINT = sys.argv[5] if len(sys.argv) > 5 else "none"
LEDGER = sys.argv[6] if len(sys.argv) > 6 else ""
DURABILITY = (sys.argv[7] if len(sys.argv) > 7 else "default").lower()
THREAD = "pcb-42"  # the recovery key shared by run and resume processes


class S(TypedDict):
    log: Annotated[list[str], operator.add]


def external(line: str) -> None:
    # The "outside world": plain append, no transaction, no dedup.
    with open(LOG, "a") as f:
        f.write(line + "\n")


def ledger_keys() -> set[str]:
    """Load the done-marker set from the ledger JSON file."""
    return set(json.load(open(LEDGER))) if os.path.exists(LEDGER) else set()


def crash_point(name: str) -> None:
    # Announce the sleep point (the parent polls for this file), then wait to
    # be killed with SIGKILL (rc=-9).
    Path(READY).write_text(name)
    print(f"[child] sleeping at crash point: {name}", flush=True)
    time.sleep(60)


def propose(state: S) -> dict:
    external("node=propose")
    return {"log": ["propose: copper=2.0 oz"]}


def apply(state: S, config: RunnableConfig) -> dict:
    conf = config["configurable"]
    # Task ids are deterministic per (parent checkpoint, node): the re-run on
    # resume gets the SAME id, which is what makes the ledger key stable.
    key = f"{conf['thread_id']}:{conf['__pregel_task_id']}"
    external("node=apply")
    print(f"[child] node=apply task={conf['__pregel_task_id']}", flush=True)
    if LEDGER and key in ledger_keys():
        external(f"SKIP task={conf['__pregel_task_id']}")
        return {"log": ["apply: skipped (ledger)"]}
    if POINT == "before_write":
        crash_point("before_write")
    external("WRITE copper=2.0")  # THE side effect under test
    if LEDGER:
        json.dump(sorted(ledger_keys() | {key}), open(LEDGER, "w"))
    if POINT == "after_write":
        crash_point("after_write")
    return {"log": ["apply: written"]}


def report(state: S) -> dict:
    # Sleep BEFORE writing the node=report record, so a killed report leaves
    # no record and the resumed one is unambiguous.
    if POINT == "after_checkpoint":
        crash_point("after_checkpoint")
    external("node=report")
    return {"log": ["report: done"]}


builder = StateGraph(S)
builder.add_node("propose", propose)
builder.add_node("apply", apply)
builder.add_node("report", report)
builder.add_edge(START, "propose")
builder.add_edge("propose", "apply")
builder.add_edge("apply", "report")
builder.add_edge("report", END)
graph = builder.compile(checkpointer=build_checkpointer("sqlite", SQLITE))
cfg = thread(THREAD)

print(f"[child] mode={MODE} point={POINT} durability={DURABILITY}", flush=True)
if MODE == "run":
    graph.invoke({"log": []}, cfg, durability=None if DURABILITY == "default" else DURABILITY)
    print(f"RESULT {graph.get_state(cfg).values}", flush=True)
else:
    snap = graph.get_state(cfg)
    print(f"STATE next={snap.next} step={(snap.metadata or {}).get('step')} values={snap.values}", flush=True)
    if not snap.values and not snap.next:
        # Nothing was ever checkpointed for this thread (durability=exit + kill).
        print("NO-CHECKPOINT", flush=True)
    else:
        print(f"RESULT {graph.invoke(None, cfg)}", flush=True)
print(f"HISTORY n={len(list(graph.get_state_history(cfg)))}", flush=True)
print(f"LOG {open(LOG).read().splitlines() if os.path.exists(LOG) else []}", flush=True)
