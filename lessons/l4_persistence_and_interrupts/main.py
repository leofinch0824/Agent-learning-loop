"""L4 - Persistence, Interrupts, and Reliable Execution.

Scenario: a PCB process-parameter assistant proposes a spec change (copper
thickness) that a human must approve before it is written to the external
spec store (a plain append-only log file standing in for the real system).
Everything the framework gives us for reliability -- checkpoints, threads,
durability levels, interrupts, resume, time travel -- is exercised against
that one external write, because that is where reliability actually breaks.

The centerpiece is a REAL process crash (SIGKILL), not an exception:

    parent (main.py / test_main.py)            child (crash_runner.py)
    -------------------------------            ------------------------
    spawn `run` with sqlite path,      -->     build graph on sqlite
    log path, ready path, crash point          checkpointer, thread pcb-42
    poll for the ready file          <--       node hits the crash point,
                                                writes ready, sleeps 60s
    SIGKILL  -----------------------  x        process dies mid-node
    spawn `resume`, same sqlite path, -->      rebuild graph, get_state
    same thread_id                             shows next=('apply', ...)
                                               invoke(None) re-runs apply
                                               WRITE lands a 2nd time (!)

The failure window that causes the duplicate (why "after_write" is special):

    super-step N (node `apply`):
    |-- node starts --|-- WRITE lands --|-- node returns --|-- checkpoint --|
                         ^ SIGKILL here = external write WITHOUT checkpoint
                           -> resume cannot know it happened -> re-runs it

Run it:

    poetry run python lessons/l4_persistence_and_interrupts/main.py

Design promise: every demo prints raw runtime evidence (stream chunks with
__interrupt__, snapshot next/tasks, checkpoint history, external log lines)
and every claim about "how many times did it land" is answered by DIRECT
counting of log-file lines -- never inferred from final state. Probed API
facts on langgraph 1.2.11 are marked inline; anything not observable offline
is printed as such instead of being faked.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Annotated, TypedDict

import operator

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, RunnableConfig, interrupt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lib import (  # noqa: E402
    banner,
    build_checkpointer,
    print_history,
    print_state,
    show_graph,
    thread,
    trace,
)

CRASH_RUNNER = Path(__file__).resolve().parent / "crash_runner.py"


def say(node: str, message: str) -> None:
    """Node-side console output (flushed so subprocess interleaving stays readable)."""
    print(f"    [{node}] {message}", flush=True)


# ---------------------------------------------------------------------------
# 1. Shared definitions: state schema + the external "spec store"
# ---------------------------------------------------------------------------


class SpecState(TypedDict):
    """State of the spec-review flow.

    PITFALL (hit while developing this lesson): a node may only write channels
    that appear in the schema ANNOTATING THAT NODE'S `state` parameter. A node
    annotated with a narrower TypedDict has its writes to missing channels
    silently dropped -- no error, the key just never lands. Always annotate
    node params with the full state type unless you deliberately subset them.
    """

    log: Annotated[list[str], operator.add]  # reducer: append-only audit trail
    value: float  # proposed copper thickness (oz)
    decision: str  # "" while pending; accept | edit | reject
    status: str  # "" while pending; written | rejected


class MultiState(TypedDict):
    """State for the multiple-interrupt demo (DEMO 5)."""

    log: Annotated[list[str], operator.add]
    i: int  # loop counter; MUST be declared or writes to it are dropped


class CountState(TypedDict):
    """State for the thread-continuity demo (DEMO 1)."""

    log: Annotated[list[str], operator.add]
    n: int  # value to append; MUST be in the schema the node is annotated with


def spec_store_append(log_path: Path, line: str) -> None:
    """The external side effect: an append to the 'spec store' log file.

    Deliberately NOT transactional and NOT idempotent -- exactly like most real
    external systems (HTTP calls, DB rows outside our checkpointer, emails).
    """
    with open(log_path, "a") as f:
        f.write(line + "\n")


def spec_store_lines(log_path: Path) -> list[str]:
    return log_path.read_text().splitlines() if log_path.exists() else []


# ---------------------------------------------------------------------------
# 2. Graph builders (checkpointer is keyword-only: the point of this lesson)
# ---------------------------------------------------------------------------


def build_accumulator_graph(*, checkpointer=None):
    """One node that appends -- used to observe thread continuity (DEMO 1)."""

    def append(state: CountState) -> dict:
        return {"log": [f"n={state['n']}"]}

    builder = StateGraph(CountState)
    builder.add_node("append", append)
    builder.add_edge(START, "append")
    builder.add_edge("append", END)
    return builder.compile(checkpointer=checkpointer) if checkpointer else builder.compile()


def build_durability_graph(*, checkpointer, fail_once: list[bool]):
    """a -> boom -> c. `boom` raises while `fail_once` is non-empty (DEMO 2).

    `fail_once` is a mutable closure cell: in-process only, it lets us observe
    the state AFTER a mid-run exception, then RESUME the same thread and watch
    the unfinished node succeed. Closures do not survive a process boundary --
    for that we use crash_runner.py and the external log instead.
    """

    def a(state: SpecState) -> dict:
        return {"log": ["a"]}

    def boom(state: SpecState) -> dict:
        if fail_once:
            fail_once.pop()
            say("boom", "raising RuntimeError mid-run")
            raise RuntimeError("boom exploded")
        say("boom", "re-run after failure, succeeding this time")
        return {"log": ["boom"]}

    def c(state: SpecState) -> dict:
        return {"log": ["c"]}

    builder = StateGraph(SpecState)
    builder.add_node("a", a)
    builder.add_node("boom", boom)
    builder.add_node("c", c)
    builder.add_edge(START, "a")
    builder.add_edge("a", "boom")
    builder.add_edge("boom", "c")
    builder.add_edge("c", END)
    return builder.compile(checkpointer=checkpointer)


def build_review_graph(*, checkpointer=None, log_path: Path | None = None):
    """propose -> review(interrupt) -> apply | closed (DEMOS 3, 4, 8).

    `review` pauses via the DYNAMIC interrupt: the payload (the proposal) is
    computed by the node at runtime, and the resume value is the human's
    decision dict. `apply` performs the external write; `closed` is the reject
    path and must never touch the external store.
    """

    def propose(state: SpecState) -> dict:
        return {"log": [f"propose: copper={state['value']} oz"]}

    def review(state: SpecState) -> dict:
        decision = interrupt(
            {"proposal": state["value"], "question": "accept / edit / reject?"}
        )
        # The resume value IS the node's partial update: decision + edited value.
        return decision

    def apply(state: SpecState) -> dict:
        if log_path is not None:
            spec_store_append(log_path, f"WRITE copper={state['value']}")
        return {"log": [f"apply: wrote copper={state['value']}"], "status": "written"}

    def closed(state: SpecState) -> dict:
        return {"log": ["closed: proposal rejected, nothing written"], "status": "rejected"}

    def route(state: SpecState) -> str:
        return "apply" if state["decision"] in ("accept", "edit") else "closed"

    builder = StateGraph(SpecState)
    builder.add_node("propose", propose)
    builder.add_node("review", review)
    builder.add_node("apply", apply)
    builder.add_node("closed", closed)
    builder.add_edge(START, "propose")
    builder.add_edge("propose", "review")
    builder.add_conditional_edges("review", route, ["apply", "closed"])
    builder.add_edge("apply", END)
    builder.add_edge("closed", END)
    return builder.compile(checkpointer=checkpointer) if checkpointer else builder.compile()


def build_static_breakpoint_graph(*, checkpointer=None):
    """prepare -> apply compiled with interrupt_before=['apply'] (DEMO 3).

    NO interrupt() call anywhere: the pause is a property of the compiled
    graph, not of node logic. Contrast with build_review_graph.
    """

    def prepare(state: SpecState) -> dict:
        return {"log": ["prepare: spec change staged"]}

    def apply(state: SpecState) -> dict:
        return {"log": ["apply: applied without human payload"], "status": "written"}

    builder = StateGraph(SpecState)
    builder.add_node("prepare", prepare)
    builder.add_node("apply", apply)
    builder.add_edge(START, "prepare")
    builder.add_edge("prepare", "apply")
    builder.add_edge("apply", END)
    return builder.compile(checkpointer=checkpointer, interrupt_before=["apply"])


def build_multi_interrupt_graph(*, checkpointer=None):
    """pick -> confirm (loop x2) -> done (DEMO 5).

    TWO nodes carry interrupts, and `confirm` interrupts TWICE -- once per
    loop iteration -- which on langgraph 1.2.11 works fine: a node may host
    more than one interrupt call across executions. The design constraint is
    that the ORDER of interrupts across resumes must be stable, because resume
    answers are matched to interrupts by replay position.
    """

    def pick(state: MultiState) -> dict:
        material = interrupt({"ask": "material"})
        return {"log": [f"material={material}"]}

    def confirm(state: MultiState, config: RunnableConfig) -> dict:
        n = state["i"]
        answer = interrupt({"ask": f"confirm item {n}"})
        return {"log": [f"item{n}={answer}"], "i": n + 1}

    def done(state: MultiState) -> dict:
        return {"log": ["done"]}

    builder = StateGraph(MultiState)
    builder.add_node("pick", pick)
    builder.add_node("confirm", confirm)
    builder.add_node("done", done)
    builder.add_edge(START, "pick")
    builder.add_edge("pick", "confirm")
    builder.add_conditional_edges("confirm", lambda s: "confirm" if s["i"] < 2 else "done")
    builder.add_edge("done", END)
    return builder.compile(checkpointer=checkpointer) if checkpointer else builder.compile()


# ---------------------------------------------------------------------------
# 3. Subprocess crash harness (drives crash_runner.py)
# ---------------------------------------------------------------------------


def spawn_child(mode: str, workdir: Path, point: str = "none", ledger: str = "", durability: str = "default"):
    """Start a crash_runner.py child; paths are derived from workdir."""
    sqlite = workdir / "checkpoints.sqlite"
    log = workdir / "spec_store.log"
    ready = workdir / "ready.txt"
    proc = subprocess.Popen(
        [
            sys.executable,
            str(CRASH_RUNNER),
            mode,
            str(sqlite),
            str(log),
            str(ready),
            point,
            ledger,
            durability,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc, sqlite, log, ready


def wait_for_file(path: Path, timeout: float = 15.0) -> bool:
    """Poll until the child announces its crash point (robust, no sleeps)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return True
        time.sleep(0.02)
    return False


def crash_scenario(point: str, *, ledger: bool = False, durability: str = "default") -> dict:
    """Run one full kill+resume scenario in a throwaway directory.

    Returns the log lines observed between crash and resume (`pre_lines`),
    the resume child's stdout, and the final log lines (`final_lines`).
    """
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        ledger_path = str(workdir / "ledger.json") if ledger else ""
        proc, _sqlite, log, ready = spawn_child("run", workdir, point, ledger_path, durability)
        reached = wait_for_file(ready)
        proc.kill()  # SIGKILL: no cleanup, no finally blocks, no exit checkpoint
        proc.wait(timeout=10)
        pre_lines = spec_store_lines(log)
        resume = subprocess.run(
            [
                sys.executable,
                str(CRASH_RUNNER),
                "resume",
                str(workdir / "checkpoints.sqlite"),
                str(log),
                str(ready),
                "none",
                ledger_path,
                "default",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        return {
            "point": point,
            "reached": reached,
            "rc": proc.returncode,
            "pre_lines": pre_lines,
            "resume_stdout": resume.stdout,
            "final_lines": spec_store_lines(log),
        }


def count_writes(lines: list[str]) -> int:
    return sum(1 for line in lines if line.startswith("WRITE"))


def count_node(lines: list[str], node: str) -> int:
    return sum(1 for line in lines if line == f"node={node}")


def print_scenario(result: dict) -> None:
    print(f"  crash point      : {result['point']} (ready seen: {result['reached']}, rc={result['rc']})")
    print(f"  log after crash  : {result['pre_lines']}")
    for line in result["resume_stdout"].splitlines():
        print(f"  resume> {line}")
    print(f"  final log        : {result['final_lines']}")
    print(f"  WRITE count      : {count_writes(result['final_lines'])}")


# ---------------------------------------------------------------------------
# 4. Demos
# ---------------------------------------------------------------------------


def demo_checkpointer_and_thread() -> None:
    banner("DEMO 1 - checkpointer 与 thread：同线程续跑、换线程重开")
    initial = {"log": []}

    print("InMemorySaver -- lives and dies with the process object:")
    memory_graph = build_accumulator_graph(checkpointer=build_checkpointer("memory"))
    t1, t2 = thread("t1"), thread("t2")
    trace(memory_graph, {**initial, "n": 1}, t1, label="t1 first invoke")
    trace(memory_graph, {**initial, "n": 2}, t1, label="t1 second invoke (SAME thread continues)")
    trace(memory_graph, {**initial, "n": 3}, t2, label="t2 first invoke (NEW thread starts fresh)")
    print_state(memory_graph, t1)
    print_state(memory_graph, t2)

    print("\nSqliteSaver -- persistence lives in the FILE, not the object:")
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "checkpoints.sqlite"
        sqlite_graph = build_accumulator_graph(checkpointer=build_checkpointer("sqlite", str(db)))
        trace(sqlite_graph, {**initial, "n": 1}, thread("s1"), label="s1 first invoke")
        trace(sqlite_graph, {**initial, "n": 2}, thread("s1"), label="s1 second invoke")
        # Simulates "another process": a brand-new graph + checkpointer object.
        reopened = build_accumulator_graph(checkpointer=build_checkpointer("sqlite", str(db)))
        snapshot = reopened.get_state(thread("s1"))
        print(f"  reopened graph + fresh SqliteSaver on the same file sees: {snapshot.values['log']}")
        assert snapshot.values["log"] == ["n=1", "n=2"]
        print_history(reopened, thread("s1"))


def demo_durability() -> None:
    banner("DEMO 2 - durability 三档：exit / async / sync 落盘时机")
    print("Probed on 1.2.11: stream/invoke accept durability in {'sync','async','exit'};")
    print("default (None) behaves identically to 'sync'. A mid-run node exception shows")
    print("what is on disk BEFORE the process can react:\n")

    for durability in ("sync", "async", "exit"):
        graph = build_durability_graph(checkpointer=build_checkpointer("sqlite", ":memory:"), fail_once=[True])
        cfg = thread(f"dur-{durability}")
        try:
            graph.invoke({"log": [], "value": 2.0, "decision": "", "status": ""}, cfg, durability=durability)
        except RuntimeError as exc:
            print(f"  durability={durability!r}: invoke raised {exc}")
        history = list(graph.get_state_history(cfg))
        steps = [(snap.metadata or {}).get("step") for snap in history]
        print(f"    checkpoints on disk: {len(history)} (steps {steps}, newest first)")
        print(f"    resumable next      : {graph.get_state(cfg).next}")

    print("\n  Resuming the 'sync' thread -- the failed node simply re-runs:")
    graph = build_durability_graph(checkpointer=build_checkpointer("sqlite", ":memory:"), fail_once=[True])
    cfg = thread("dur-resume")
    try:
        graph.invoke({"log": [], "value": 2.0, "decision": "", "status": ""}, cfg)
    except RuntimeError:
        pass
    trace(graph, None, cfg, label="resume after exception: boom re-runs, then c")

    print("\n  Facts established offline:")
    print("  - sync/async (sync saver): a checkpoint per super-step; after a crash the")
    print("    process resumes from the last COMPLETED super-step.")
    print("  - exit: ONE checkpoint per invoke, written on the way out -- even when the")
    print("    run raises. Intermediate steps are never on disk (no time travel mid-run).")
    print("  - async vs sync are indistinguishable through a synchronous saver")
    print("    (SqliteSaver.aput is blocking); the true 'fire-and-forget' semantics of an")
    print("    async saver are NOT observable here -> 未实测 (see README).")
    print("  - exit + SIGKILL (no clean unwind) leaves NO checkpoint at all -- DEMO 6.")


def demo_dynamic_vs_static() -> None:
    banner("DEMO 3 - 动态 interrupt vs 静态断点")
    print("Dynamic: interrupt() INSIDE the node carries a runtime-computed payload.\n")
    graph = build_review_graph(checkpointer=build_checkpointer("memory"))
    show_graph(graph)
    cfg = thread("dyn")
    trace(graph, {"log": [], "value": 2.0, "decision": "", "status": ""}, cfg, label="run pauses inside review")
    snapshot = print_state(graph, cfg)
    print(f"  pending task interrupts: {[i.value for t in snapshot.tasks for i in t.interrupts]}")
    trace(graph, Command(resume={"decision": "accept", "value": 2.0}), cfg, label="resume via Command(resume=...)")

    print("\n  Counter-check (probed fact): plain invoke(None) does NOT resume a dynamic")
    print("  interrupt -- it just re-fires the same interrupt:")
    cfg2 = thread("dyn-none")
    list(graph.stream({"log": [], "value": 2.0, "decision": "", "status": ""}, cfg2))
    graph.invoke(None, cfg2)
    print(f"  after invoke(None): next={graph.get_state(cfg2).next} (still paused!)")

    print("\nStatic: compile(interrupt_before=['apply']) -- no interrupt() call at all.\n")
    static_graph = build_static_breakpoint_graph(checkpointer=build_checkpointer("memory"))
    cfg3 = thread("static")
    chunks = list(static_graph.stream({"log": [], "value": 2.0, "decision": "", "status": ""}, cfg3, stream_mode=["updates"]))
    print(f"  run chunks: {chunks}")
    snapshot = print_state(static_graph, cfg3)
    print(f"  task interrupts on a static pause: {[t.interrupts for t in snapshot.tasks]} (empty!)")
    trace(static_graph, None, cfg3, label="resume static pause with plain None input")
    cfg4 = thread("static-command")
    list(static_graph.stream({"log": [], "value": 2.0, "decision": "", "status": ""}, cfg4))
    trace(static_graph, Command(resume=True), cfg4, label="static pause also accepts Command(resume=True)")


def demo_review_paths() -> None:
    banner("DEMO 4 - 接受 / 修改 / 拒绝：三条审核路径")
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "spec_store.log"
        graph = build_review_graph(checkpointer=build_checkpointer("memory"), log_path=log_path)
        paths = {
            "accept": {"decision": "accept", "value": 2.0},
            "edit": {"decision": "edit", "value": 1.5},
            "reject": {"decision": "reject", "value": 2.0},
        }
        for name, answer in paths.items():
            cfg = thread(f"review-{name}")
            trace(graph, {"log": [], "value": 2.0, "decision": "", "status": ""}, cfg, label=f"{name}: run to interrupt")
            trace(graph, Command(resume=answer), cfg, label=f"{name}: resume {answer}")
            print(f"  -> thread review-{name} final: {graph.get_state(cfg).values}")
        lines = spec_store_lines(log_path)
        print(f"\n  external spec store: {lines}")
        print(f"  writes landed: accept=1, edit=1, reject=0 -> {count_writes(lines)} WRITE lines total")
        print("  reject skipped the apply node entirely (routed to `closed`).")


def demo_multi_interrupt_order() -> None:
    banner("DEMO 5 - 多个中断的顺序对照")
    graph = build_multi_interrupt_graph(checkpointer=build_checkpointer("memory"))
    cfg = thread("multi")
    answers = ["FR-4", "yes", "yes"]
    order: list[str] = []
    inp = {"log": [], "i": 0}
    rounds = 0
    while True:
        rounds += 1
        chunks = list(graph.stream(inp, cfg, stream_mode=["updates"]))
        for mode, chunk in chunks:
            if isinstance(chunk, dict) and "__interrupt__" in chunk:
                order.extend(i.value["ask"] for i in chunk["__interrupt__"])
        if not graph.get_state(cfg).next:
            break
        inp = Command(resume=answers[len(order) - 1])
    print(f"  stream calls: {rounds}; interrupt payload order: {order}")
    print(f"  final state: {graph.get_state(cfg).values}")
    print("  two different nodes host interrupts AND `confirm` interrupts twice across")
    print("  its loop -- the order is stable: answers are matched by replay position.")


def demo_crash_and_resume() -> None:
    banner("DEMO 6 - 子进程真崩溃 + 重启恢复：checkpoint ≠ 外部副作用恰好一次")
    print("Each scenario: spawn crash_runner (run), wait for its ready-file, SIGKILL it,\n"
          "then resume in a NEW process with the same sqlite file + thread_id.\n")

    for point in ("before_write", "after_write", "after_checkpoint"):
        result = crash_scenario(point)
        print(f"\n  --- crash point: {point} ---")
        print_scenario(result)

    print("\n  Direct counting of the external WRITE line:")
    print("  - before_write    : write had NOT landed -> resume re-runs apply -> 1 write (fine)")
    print("  - after_write     : write landed but NO checkpoint for that super-step ->")
    print("                      resume re-runs apply -> 2 writes. THE failure window.")
    print("  - after_checkpoint: apply's checkpoint committed -> resume starts at report -> 1 write")

    print("\n  The idempotency fix: an execution LEDGER keyed by thread_id + task_id")
    print("  (task ids are deterministic: the re-run after the crash gets the SAME id).")
    result = crash_scenario("after_write", ledger=True)
    print_scenario(result)
    print("  -> WRITE landed once, the replayed execution hit the ledger and skipped.")

    print("\n  durability='exit' + SIGKILL (no clean unwind -> no exit checkpoint):")
    result = crash_scenario("after_write", durability="exit")
    print_scenario(result)
    print("  -> resume process finds NO checkpoint for the thread (NO-CHECKPOINT);")
    print("     the external write is stranded with no state to resume from.")


def demo_replay_scope() -> None:
    banner("DEMO 7 - 恢复重放范围：只有未完成的工作会重跑")
    result = crash_scenario("after_write")
    print_scenario(result)
    pre, final = result["pre_lines"], result["final_lines"]
    print("\n  per-node execution records (node=... lines in the external log):")
    for node in ("propose", "apply", "report"):
        print(f"    {node:<8} before resume: {count_node(pre, node)}   after resume: {count_node(final, node)}")
    history_line = next(l for l in result["resume_stdout"].splitlines() if l.startswith("HISTORY"))
    print(f"  checkpoint history in the resume process: {history_line}")
    print("  -> propose never re-ran (its checkpoint was committed); apply re-ran once")
    print("     (its super-step was unfinished); report ran only after the resume.")
    print("     Replay scope == unfinished work, nothing more.")


def demo_history_and_fork() -> None:
    banner("DEMO 8 - 历史查询、分叉与外部撤销的区别")
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "spec_store.log"
        graph = build_review_graph(checkpointer=build_checkpointer("memory"), log_path=log_path)
        cfg = thread("main")
        list(graph.stream({"log": [], "value": 2.0, "decision": "", "status": ""}, cfg, stream_mode=["updates"]))
        list(graph.stream(Command(resume={"decision": "accept", "value": 2.0}), cfg, stream_mode=["updates"]))
        accept_result = graph.get_state(cfg).values
        print(f"  1. main thread accepted copper=2.0 -> external store: {spec_store_lines(log_path)}")

        print("\n  2. checkpoint history (newest first) -- this is what time travel walks:")
        history = print_history(graph, cfg)
        paused = next(h for h in history if h.tasks and any(t.interrupts for t in h.tasks))
        print(f"     forking from the paused-at-review checkpoint: step={paused.metadata.get('step')}")

        print("\n  3. fork on 1.2.11 = update_state on the old checkpoint, as_node='review':")
        fork_cfg = graph.update_state(paused.config, {"decision": "edit", "value": 1.5}, as_node="review")
        fork_result = graph.invoke(None, fork_cfg)
        print(f"     fork branch result: {fork_result}")
        print(f"     external store now: {spec_store_lines(log_path)}  <- TWO writes, no rollback")
        print(f"     thread head moved to the fork: {graph.get_state(cfg).values['log']}")
        print("     (probed: same-thread forking MOVES the head; the old branch survives in history)")

        print("\n  4. the pattern from older docs does NOT fork on 1.2.11 (probed):")
        raw = {"configurable": {"thread_id": "fork-raw", "checkpoint_id": paused.config["configurable"]["checkpoint_id"]}}
        print(f"     get_state(new thread + old checkpoint_id).values = {graph.get_state(raw).values}")
        raw_cfg = graph.update_state(raw, {"decision": "edit", "value": 9.9}, as_node="review")
        raw_result = graph.invoke(None, raw_cfg)
        print(f"     update_state + invoke on it 'succeeds' -- but from an EMPTY state: {raw_result['log']}")
        print(f"     external store: {spec_store_lines(log_path)}  <- a THIRD write, silently fresh-started")

        print("\n  5. restoring the head = replay the original paused checkpoint with the SAME")
        print("     resume command (probed: the cached answer is replayed):")
        list(graph.stream(Command(resume={"decision": "accept", "value": 2.0}), paused.config, stream_mode=["updates"]))
        print(f"     head after replay: {graph.get_state(cfg).values['log']}")
        print(f"     accept branch values back? {graph.get_state(cfg).values == accept_result}")
        print(f"     external store after the replay: {spec_store_lines(log_path)}")
        print("     note: the replay re-executed apply and DUPLICATED its write -- replays")
        print("     are re-executions, not cache reads.")

        print("\n  6. time travel rolled back NOTHING in the external store -- undoing is")
        print("     OUR compensating action, written by hand, not a framework feature:")
        kept: list[str] = []
        for line in spec_store_lines(log_path):
            if line in {"WRITE copper=1.5", "WRITE copper=9.9"}:
                continue  # writes of abandoned branches
            if line == "WRITE copper=2.0" and line in kept:
                continue  # duplicate produced by the replay
            kept.append(line)
        log_path.write_text("".join(line + "\n" for line in kept))
        print(f"     after compensation: {spec_store_lines(log_path)}")


if __name__ == "__main__":
    demo_checkpointer_and_thread()
    demo_durability()
    demo_dynamic_vs_static()
    demo_review_paths()
    demo_multi_interrupt_order()
    demo_crash_and_resume()
    demo_replay_scope()
    demo_history_and_fork()
    banner("done", width=72)
