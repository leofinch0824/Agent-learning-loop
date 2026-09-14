"""L3 - Tools and Execution Boundaries.

Scenario: a miniature agent harness around a KV "world" plus a subprocess code
sandbox. The model only ever PROPOSES tool calls; a ToolExecutor owns every
boundary that matters -- argument contracts, read/write classification,
concurrency control, timeout/retry/cancellation, the permission gate, and
sandboxing. Every decision lands in one structured execution log, so a failed
multi-step run can be attributed from the log alone.

    model (proposes)              executor (decides + executes)          world (effects)
    ---------------              -----------------------------         ----------------
    tool_call JSON  ───────────► PermissionGate ──► validate args ──► KVStore / subprocess
         ▲  feedback loop             │ reject            │ timeout / retry / cancel
         │                             ▼                   ▼
         └──────────── structured result envelope ◄── write lock
                                      │
                                      ▼
                        execution log (tool, args, decision, outcome, latency)

Design promise: envelope errors (bad args, denied, timeout) are OBSERVATIONS
fed back to the model; exceptions from tool code are PROGRAMMER BUGS that fail
fast and kill the run. Every demo below counts side effects directly instead
of inferring them from final state.

Run it (one process, one event loop -- the executor's asyncio.Lock must not
cross event loops, which is itself a lesson in why harnesses own their loop):

    poetry run python lessons/l3_tools_and_execution/main.py
"""

from __future__ import annotations

import asyncio
import inspect
import json
import operator
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Callable, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lib import banner, show_graph  # noqa: E402

from lessons.l3_tools_and_execution.mcp_server import LENGTHS, unit_convert  # noqa: E402

LESSON_DIR = Path(__file__).resolve().parent


def say(where: str, message: str) -> None:
    """Component-side console output (kept uniform with L1's node output)."""
    print(f"    [{where}] {message}", flush=True)


# ---------------------------------------------------------------------------
# 1. Tool contract: ToolSpec + result envelope
# ---------------------------------------------------------------------------

# Permission modes: the executor's answer to "who may run this".
ALLOW = "allow"
NEEDS_APPROVAL = "needs_approval"
FORBID = "forbid"


class ToolFailure(Exception):
    """An EXPECTED failure a tool raises on purpose so it is fed back to the
    model as data (an error envelope) instead of killing the run.

    This gives tool code two failure channels with different semantics:
    raise ToolFailure -> "this attempt failed, model/harness can react";
    raise anything else -> "this tool has a BUG", the executor re-raises and
    the run stops (DEMO 1B). Expected failures are data; bugs are not.
    """


@dataclass
class ToolSpec:
    """Declarative tool contract shown to the model AND enforced by the executor.

    ``kind`` (readonly/write) is not documentation -- it drives concurrency
    (write lock), retry policy (only idempotent reads), and permissions.
    If the classification is wrong, every boundary built on it is wrong too.
    """

    name: str
    description: str
    params_schema: dict[str, Any]  # JSON-schema subset, validated by hand below
    kind: Literal["readonly", "write"]
    fn: Callable[..., Any]


@dataclass
class ExecRecord:
    """One structured line of the execution log (DEMO 7 attributes failures from these)."""

    tool: str
    args: dict[str, Any]
    decision: str  # what the executor decided BEFORE running: execute | reject:<reason>
    outcome: str  # ok | denied | invalid_args | unknown_tool | timeout | tool_failure | raised | cancelled | retrying
    attempts: int = 1
    latency_ms: float = 0.0
    error_code: str | None = None
    detail: str | None = None


# JSON-schema type names -> python types. Deliberately tiny: enough to show a
# contract being CHECKED, not a validator library.
_TYPE_MAP: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
}


def validate_args(spec: ToolSpec, args: dict[str, Any]) -> str | None:
    """Return a human-readable violation, or None if args satisfy the schema.

    Runs at the execution layer, BEFORE the tool function is touched: the model
    gets a structured rejection it can act on instead of a crash.
    """
    props = spec.params_schema.get("properties", {})
    for key in spec.params_schema.get("required", []):
        if key not in args:
            return f"missing required argument {key!r}"
    for key, value in args.items():
        if key not in props:
            return f"unknown argument {key!r}"
        rule = props[key]
        expected = _TYPE_MAP.get(rule.get("type", "string"))
        if expected and not isinstance(value, expected):
            return f"argument {key!r} must be {rule['type']}, got {type(value).__name__}"
        if "enum" in rule and value not in rule["enum"]:
            return f"argument {key!r} must be one of {rule['enum']}, got {value!r}"
        if "minimum" in rule and value < rule["minimum"]:
            return f"argument {key!r} must be >= {rule['minimum']}, got {value!r}"
        if "maximum" in rule and value > rule["maximum"]:
            return f"argument {key!r} must be <= {rule['maximum']}, got {value!r}"
    return None


# ---------------------------------------------------------------------------
# 2. The world: a KV store whose every effect is countable
# ---------------------------------------------------------------------------


class KVStore:
    """The 'world' the tools act on. All mutations funnel through here so side
    effects can be COUNTED, not guessed from final state."""

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        self.data: dict[str, Any] = dict(data or {})
        self.reads = 0
        self.writes = 0

    def get(self, key: str) -> Any:
        self.reads += 1
        return self.data.get(key)

    def set(self, key: str, value: Any) -> Any:
        self.writes += 1
        self.data[key] = value
        return value

    def clear(self) -> int:
        cleared = len(self.data)
        self.writes += 1  # a wipe is an effect too
        self.data.clear()
        return cleared

    def snapshot(self) -> dict[str, Any]:
        return dict(self.data)


# ---------------------------------------------------------------------------
# 3. Tool catalogue (built around a store; factories for the failure props)
# ---------------------------------------------------------------------------


def default_specs(store: KVStore) -> dict[str, ToolSpec]:
    """The stable tool set: reads, writes, and the MCP twin of unit_convert."""

    def kv_get(key: str) -> Any:
        return store.get(key)

    def kv_set(key: str, value: float) -> float:
        return store.set(key, value)

    async def kv_add(key: str, delta: float) -> float:
        # read-modify-write with a DELIBERATE interleave window: without the
        # executor's write lock two concurrent adds lose an update (DEMO 2).
        current = store.get(key) or 0
        await asyncio.sleep(0.05)
        return store.set(key, current + delta)

    def delete_all() -> int:
        return store.clear()

    return {
        "kv_get": ToolSpec(
            name="kv_get",
            description="Read one key from the workbench store.",
            params_schema={
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"],
            },
            kind="readonly",
            fn=kv_get,
        ),
        "kv_set": ToolSpec(
            name="kv_set",
            description="Overwrite one key. Destructive: needs approval.",
            params_schema={
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                    "value": {"type": "number", "minimum": -1e9, "maximum": 1e9},
                },
                "required": ["key", "value"],
            },
            kind="write",
            fn=kv_set,
        ),
        "kv_add": ToolSpec(
            name="kv_add",
            description="Increment one key by delta (read-modify-write).",
            params_schema={
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                    "delta": {"type": "number"},
                },
                "required": ["key", "delta"],
            },
            kind="write",
            fn=kv_add,
        ),
        "delete_all": ToolSpec(
            name="delete_all",
            description="Wipe the whole store. Forbidden in this lesson.",
            params_schema={"type": "object", "properties": {}, "required": []},
            kind="write",
            fn=delete_all,
        ),
        # The SAME function object the MCP server in mcp_server.py exposes --
        # DEMO 6 calls it directly and across the protocol boundary.
        "unit_convert": ToolSpec(
            name="unit_convert",
            description="Convert a length between units (m, cm, mm, km, in, ft).",
            params_schema={
                "type": "object",
                "properties": {
                    "value": {"type": "number", "maximum": 1e9},
                    "from_unit": {"type": "string", "enum": sorted(LENGTHS)},
                    "to_unit": {"type": "string", "enum": sorted(LENGTHS)},
                },
                "required": ["value", "from_unit", "to_unit"],
            },
            kind="readonly",
            fn=unit_convert,
        ),
    }


def make_slow_read_spec(store: KVStore, delay_s: float = 0.3) -> ToolSpec:
    """A read that takes a while -- used to PROVE reads overlap in time."""

    async def slow_read(key: str) -> Any:
        await asyncio.sleep(delay_s)
        return store.get(key)

    return ToolSpec(
        name="slow_read",
        description="Slow read (used to observe read concurrency).",
        params_schema={
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
        },
        kind="readonly",
        fn=slow_read,
    )


def make_slow_write_spec(store: KVStore, hang_s: float = 5.0) -> ToolSpec:
    """A write whose effect lands only AFTER a long hang -- timeouts and
    cancellation must therefore leave zero side effects."""

    async def slow_write(key: str, value: float) -> float:
        await asyncio.sleep(hang_s)
        return store.set(key, value)

    return ToolSpec(
        name="slow_write",
        description="Write that hangs before the effect lands.",
        params_schema={
            "type": "object",
            "properties": {"key": {"type": "string"}, "value": {"type": "number"}},
            "required": ["key", "value"],
        },
        kind="write",
        fn=slow_write,
    )


def make_flaky_read_spec(store: KVStore, fail_times: int = 2) -> ToolSpec:
    """Fails N times then succeeds: safe to retry, because a read has no effect."""
    state = {"failures_left": fail_times}

    async def flaky_read(key: str) -> Any:
        if state["failures_left"] > 0:
            state["failures_left"] -= 1
            raise ToolFailure("transient read failure")
        return store.get(key)

    return ToolSpec(
        name="flaky_read",
        description="Flaky read: fails twice, then works.",
        params_schema={
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
        },
        kind="readonly",
        fn=flaky_read,
    )


def make_flaky_write_spec(store: KVStore) -> ToolSpec:
    """The dangerous twin: the write LANDS, then the response fails. Retrying a
    non-idempotent write would apply the effect twice -- so the policy never
    retries writes, and the effect counter is the proof."""

    async def flaky_write(key: str, value: float) -> float:
        store.set(key, value)
        raise ToolFailure("response lost AFTER the write landed")

    return ToolSpec(
        name="flaky_write",
        description="Write that lands, then fails to respond.",
        params_schema={
            "type": "object",
            "properties": {"key": {"type": "string"}, "value": {"type": "number"}},
            "required": ["key", "value"],
        },
        kind="write",
        fn=flaky_write,
    )


def make_boom_spec() -> ToolSpec:
    """Raises unconditionally: stands for a BUG in tool code, not an expected
    failure. The executor lets it propagate (fail fast) -- DEMO 1 contrasts
    this with the envelope path."""

    def boom() -> None:
        raise RuntimeError("bug inside tool code (not an expected failure)")

    return ToolSpec(
        name="boom",
        description="Always raises. Used to contrast raise-vs-envelope.",
        params_schema={"type": "object", "properties": {}, "required": []},
        kind="readonly",
        fn=boom,
    )


# ---------------------------------------------------------------------------
# 4. Sandbox: subprocess + pinned cwd + env whitelist + pre-exec path guard
# ---------------------------------------------------------------------------


class Sandbox:
    """A self-contained code-exec sandbox (no docker needed).

    Layers, each one observable:
      1. static path guard  -- reject absolute paths / ``..`` traversal /
         banned imports BEFORE spawning anything (a decision, not luck);
      2. subprocess         -- crashes and hangs cannot take down the harness;
      3. cwd pinned         -- relative paths can only address the sandbox dir;
      4. env whitelist      -- the snippet sees ONLY the whitelisted variables
         (plus `__CF_USER_TEXT_ENCODING`, which macOS injects into every
         subprocess behind your back -- visible in DEMO 5D).

    Honest limit (say it out loud in class): a static guard is a heuristic and
    can be obfuscated around (chr(47)+'etc', string concatenation). It is a
    first layer; real isolation needs OS-level mechanisms (seccomp, containers,
    VMs). The demo shows the layers it can actually prove.
    """

    BANNED_IMPORTS = ("subprocess", "socket", "shutil", "ctypes")
    _ABSOLUTE = re.compile(r"(?<![\w.])/[A-Za-z0-9_]")  # naive on purpose
    _TRAVERSAL = re.compile(r"\.\./|\.\.\\")
    _TILDE = re.compile(r"~")

    def __init__(self, root: Path) -> None:
        root.mkdir(parents=True, exist_ok=True)
        self.root = root.resolve()
        self.spawns = 0  # direct evidence of "the subprocess never even started"
        # -I = isolated python: ignores PYTHON* env vars and the user site dir.
        self.env_whitelist = {"PYTHONIOENCODING": "utf-8", "LANG": "C.UTF-8"}

    def guard(self, code: str) -> str | None:
        """Return a violation description, or None if the snippet passes."""
        if match := self._ABSOLUTE.search(code):
            return f"absolute path not allowed: {match.group(0)!r}"
        if self._TRAVERSAL.search(code):
            return "path traversal ('..') not allowed"
        if self._TILDE.search(code):
            return "home expansion ('~') not allowed"
        for module in self.BANNED_IMPORTS:
            if re.search(rf"\b(import|from)\s+{module}\b", code):
                return f"banned import: {module}"
        return None

    def run_py(self, code: str, timeout_s: float = 10.0) -> dict[str, Any]:
        """Guard first (no spawn on violation), then run in the subprocess jail."""
        violation = self.guard(code)
        if violation is not None:
            return {
                "executed": False,
                "violation": violation,
                "returncode": None,
                "stdout": "",
                "stderr": "",
            }
        self.spawns += 1
        proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [sys.executable, "-I", "-c", code],
            cwd=self.root,
            env=self.env_whitelist,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        return {
            "executed": True,
            "violation": None,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip()[:200],
        }


def make_py_exec_spec(sandbox: Sandbox) -> ToolSpec:
    """Expose the sandbox as a normal tool so guard refusals flow through the
    same result envelope as every other outcome."""

    def py_exec(code: str) -> dict[str, Any]:
        return sandbox.run_py(code)

    return ToolSpec(
        name="py_exec",
        description="Run a tiny python snippet inside the sandbox.",
        params_schema={
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
        },
        kind="write",  # code execution can mutate; treated as a write
        fn=py_exec,
    )


# ---------------------------------------------------------------------------
# 5. The executor: permissions, validation, timeout, retry, write lock, log
# ---------------------------------------------------------------------------


@dataclass
class RetryPolicy:
    """Retries are gated on IDEMPOTENCE: only readonly tools, never writes.

    A retried write whose first attempt already landed applies the effect
    twice (see flaky_write). Reads are safe to retry by construction.
    """

    max_attempts: int = 3
    retry_kinds: tuple[str, ...] = ("readonly",)


@dataclass
class ToolExecutor:
    """The runtime half of 'model proposes, executor decides'.

    Order of checks per call: unknown tool -> permission gate -> arg contract
    -> (retry loop: write lock -> timeout-bounded invoke). Every step records
    an ExecRecord; permission refusals additionally land in ``rejections``
    with the reason, and produce ZERO side effects because the tool function
    is never entered.

    NOTE: the write lock binds to the running event loop on first use, so one
    executor must live inside one loop (the harness owns its loop).
    """

    specs: dict[str, ToolSpec]
    permissions: dict[str, str] = field(default_factory=dict)
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    serialize_writes: bool = True  # hold one write lock across write tools
    log: list[ExecRecord] = field(default_factory=list)
    rejections: list[dict[str, Any]] = field(default_factory=list)
    _write_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    # -- public API ---------------------------------------------------------

    async def call(
        self,
        name: str,
        args: dict[str, Any] | None = None,
        *,
        approved: bool = False,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        """Execute one tool call and return the result envelope.

        ``approved=True`` is the human-approval flag (L4 turns this into a
        real interrupt; here it is a parameter so the gate can be tested).
        """
        args = dict(args or {})
        spec = self.specs.get(name)
        if spec is None:
            return self._record(name, args, "reject:unknown_tool", "unknown_tool",
                                error_code="unknown_tool",
                                message=f"no tool named {name!r}")

        # PermissionGate: the EXECUTION LAYER decides, whatever the model proposed.
        mode = self.permissions.get(name, ALLOW)
        if mode == FORBID:
            self.rejections.append({"tool": name, "args": args, "reason": "forbidden"})
            return self._record(name, args, "reject:forbidden", "denied",
                                error_code="denied",
                                message="tool is forbidden by the permission table")
        if mode == NEEDS_APPROVAL and not approved:
            self.rejections.append({"tool": name, "args": args, "reason": "needs_approval"})
            return self._record(name, args, "reject:needs_approval", "denied",
                                error_code="denied",
                                message="write tool requires human approval (approved=False)")

        problem = validate_args(spec, args)
        if problem is not None:
            return self._record(name, args, "reject:invalid_args", "invalid_args",
                                error_code="invalid_args", message=problem)

        attempts_allowed = (
            self.retry_policy.max_attempts
            if spec.kind in self.retry_policy.retry_kinds
            else 1
        )
        for attempt in range(1, attempts_allowed + 1):
            started = time.perf_counter()
            try:
                if spec.kind == "write" and self.serialize_writes:
                    # The lock spans the WHOLE read-modify-write tool call, so
                    # conflicting writes are serialized instead of interleaving.
                    async with self._write_lock:
                        value = await self._invoke(spec, args, timeout_s)
                else:
                    value = await self._invoke(spec, args, timeout_s)
            except TimeoutError:
                # wait_for() cancelled the inner call -> a hung write NEVER lands.
                return self._record(
                    name, args, "execute", "timeout", attempt=attempt,
                    latency_ms=(time.perf_counter() - started) * 1000,
                    error_code="timeout",
                    message=f"exceeded {timeout_s}s timeout; the side effect did not land",
                )
            except asyncio.CancelledError:
                # Cancellation must propagate -- but not before we log it.
                self.log.append(
                    ExecRecord(name, args, "execute", "cancelled", attempt,
                               (time.perf_counter() - started) * 1000,
                               "cancelled", "task cancelled before completion")
                )
                raise
            except ToolFailure as exc:
                # Channel 1: EXPECTED failure -> data. Retry if policy allows.
                if attempt < attempts_allowed:
                    self.log.append(
                        ExecRecord(name, args, "execute", "retrying", attempt,
                                   (time.perf_counter() - started) * 1000,
                                   "tool_failure", f"{exc}")
                    )
                    continue
                return self._record(
                    name, args, "execute", "tool_failure", attempt=attempt,
                    latency_ms=(time.perf_counter() - started) * 1000,
                    error_code="tool_failure", message=str(exc),
                )
            except Exception as exc:  # noqa: BLE001 - re-raised below, logged first
                # Channel 2: BUG in tool code -> fail fast, do NOT feed a
                # traceback to the model, do not continue the run.
                self.log.append(
                    ExecRecord(name, args, "execute", "raised", attempt,
                               (time.perf_counter() - started) * 1000,
                               "bug", f"{type(exc).__name__}: {exc}")
                )
                raise
            return self._record(name, args, "execute", "ok", attempt=attempt,
                                latency_ms=(time.perf_counter() - started) * 1000,
                                value=value)
        raise AssertionError("unreachable")  # pragma: no cover

    async def call_many(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Run several calls concurrently: reads overlap, writes serialize on
        the shared lock (asyncio.gather + the lock inside call())."""
        coros = [self.call(c["name"], c.get("args"), approved=c.get("approved", False),
                           timeout_s=c.get("timeout_s")) for c in calls]
        return await asyncio.gather(*coros)

    # -- internals ----------------------------------------------------------

    async def _invoke(self, spec: ToolSpec, args: dict[str, Any], timeout_s: float | None) -> Any:
        async def run() -> Any:
            result = spec.fn(**args)
            if inspect.isawaitable(result):
                result = await result
            return result

        if timeout_s is not None:
            return await asyncio.wait_for(run(), timeout_s)
        return await run()

    def _record(
        self,
        name: str,
        args: dict[str, Any],
        decision: str,
        outcome: str,
        *,
        attempt: int = 1,
        latency_ms: float = 0.0,
        error_code: str | None = None,
        message: str | None = None,
        value: Any = None,
    ) -> dict[str, Any]:
        self.log.append(ExecRecord(name, args, decision, outcome, attempt,
                                   latency_ms, error_code, message))
        envelope: dict[str, Any] = {"ok": outcome == "ok", "tool": name, "args": args}
        if outcome == "ok":
            envelope["value"] = value
        else:
            envelope["error"] = {"code": error_code, "message": message}
        return envelope


def print_log(records: list[ExecRecord], title: str = "execution log") -> None:
    """Render the execution log as a fixed-width table (DEMO 7's raw material)."""
    print(f"\n  {title}")
    header = f"    {'#':>2}  {'tool':<14} {'decision':<22} {'outcome':<12} {'att':>3}  {'ms':>8}  detail"
    print(header)
    print("    " + "-" * (len(header) - 4))
    for i, r in enumerate(records, 1):
        detail = r.detail or ""
        if r.error_code and r.error_code != "ok":
            detail = f"[{r.error_code}] {detail}".strip()
        print(f"    {i:>2}  {r.tool:<14} {r.decision:<22} {r.outcome:<12} "
              f"{r.attempts:>3}  {r.latency_ms:>8.2f}  {detail[:56]}")


def print_envelope(envelope: dict[str, Any]) -> None:
    """One-line view of a result envelope for demo output."""
    if envelope["ok"]:
        print(f"    ok    {envelope['tool']}({envelope['args']}) -> {envelope['value']!r}")
    else:
        error = envelope["error"]
        print(f"    FAIL  {envelope['tool']}({envelope['args']}) -> {error['code']}: {error['message']}")


# ---------------------------------------------------------------------------
# 6. A LangGraph tool loop THROUGH the gate (where a graph is the natural fit)
# ---------------------------------------------------------------------------


class LoopState(TypedDict):
    """Tool loop state: the model proposes, the executor observes back."""

    round: int
    pending: list[dict[str, Any]]  # LastValue: fully rewritten by each propose
    observations: Annotated[list[dict[str, Any]], operator.add]  # merges across the loop
    final: str  # empty until the model decides it is done


def build_gated_loop(executor: ToolExecutor, store: KVStore) -> Any:
    """propose -> execute -> (propose | END): rejections come back as
    observations the model has to deal with, instead of crashing the loop.

    The "model" here is scripted (offline determinism); DEMO 4's live twin is
    live_demo.py, where a real model plays the same propose role.
    """

    def propose(state: LoopState) -> dict:
        turn = state["round"]
        if turn == 0:
            plan = [
                {"name": "kv_get", "args": {"key": "hits"}},
                {"name": "kv_set", "args": {"key": "hits", "value": 99}},
                {"name": "delete_all", "args": {}},
            ]
            say("model", f"proposes {[c['name'] for c in plan]}")
            return {"round": 1, "pending": plan}
        denied = [r["tool"] for r in executor.rejections]
        say("model", "saw the denials in the observations; finishing with a report")
        return {"final": f"denied at the execution layer: {denied}; store still {store.snapshot()}"}

    async def execute(state: LoopState) -> dict:
        envelopes = await executor.call_many(state["pending"])
        for envelope in envelopes:
            status = "ok" if envelope["ok"] else f"DENIED ({envelope['error']['code']})"
            say("executor", f"observation: {status} {envelope['tool']}({envelope['args']})")
        return {"observations": envelopes, "pending": []}

    builder = StateGraph(LoopState)
    builder.add_node("propose", propose)
    builder.add_node("execute", execute)
    builder.add_edge(START, "propose")
    # Dynamic routing: keep looping while the model still proposes calls.
    builder.add_conditional_edges("propose", lambda state: END if state["final"] else "execute")
    builder.add_edge("execute", "propose")
    return builder.compile()


# ---------------------------------------------------------------------------
# Demo 1: tool contract -- envelope vs raise
# ---------------------------------------------------------------------------


async def demo_contract() -> None:
    banner("DEMO 1 - TOOL CONTRACT: schema in, envelope out (and when raising is right)")
    store = KVStore({"hits": 3})
    executor = ToolExecutor(default_specs(store))

    print("  A. out-of-range/unknown args come back as a STRUCTURED envelope the")
    print("     model can act on (feedback loop), and the run continues:\n")
    print_envelope(await executor.call("unit_convert", {"value": 5.0, "from_unit": "m", "to_unit": "mile"}))
    print_envelope(await executor.call("kv_set", {"key": "hits", "value": 1e12}))
    print()
    print_envelope(await executor.call("unit_convert", {"value": 5.0, "from_unit": "m", "to_unit": "ft"}))
    print("  -> the two failures above did NOT stop this call: an envelope error")
    print("     is an OBSERVATION, just like a successful result.")

    print("\n  B. the same sequence with a tool that RAISES a non-ToolFailure bug:")
    executor.specs["boom"] = make_boom_spec()
    later_ran = False
    try:
        for step_name, step_args in [
            ("kv_get", {"key": "hits"}),
            ("boom", {}),
            ("kv_set", {"key": "hits", "value": 7}),
        ]:
            await executor.call(step_name, step_args)
            if step_name == "boom":
                later_ran = True  # never reached: boom raises before returning
    except RuntimeError as exc:
        print(f"    raised as expected: RuntimeError: {exc}")
        print(f"    step after boom ran: {later_ran}; store.writes={store.writes} (kv_set never executed)")
        print(f"    log outcome for boom: {executor.log[-1].outcome} (recorded, then re-raised)")
    print("  -> the tool contract has TWO failure channels: ToolFailure and the")
    print("     executor's own checks produce ENVELOPES (data the model reacts to);")
    print("     any other exception is a BUG -> fail fast, never feed a traceback")
    print("     to the model, never continue the run on a broken harness.")


# ---------------------------------------------------------------------------
# Demo 2: reads parallelize; conflicting writes are controlled
# ---------------------------------------------------------------------------


async def demo_concurrency() -> None:
    banner("DEMO 2 - READS PARALLELIZE, CONFLICTING WRITES ARE CONTROLLED")
    store = KVStore({"hits": 0})

    # A. two slow reads concurrently: wall time proves overlap (0.3s each).
    executor = ToolExecutor(default_specs(store) | {"slow_read": make_slow_read_spec(store, delay_s=0.3)})
    started = time.perf_counter()
    results = await executor.call_many(
        [{"name": "slow_read", "args": {"key": "hits"}}, {"name": "slow_read", "args": {"key": "hits"}}]
    )
    elapsed = time.perf_counter() - started
    print(f"  A. two 0.3s reads concurrently: elapsed={elapsed:.2f}s (sequential would be ~0.60s)")
    print(f"     both ok: {all(r['ok'] for r in results)}; store.reads={store.reads}")

    # B. the bug: NO write lock -> two concurrent kv_add(+1) lose an update.
    store_b = KVStore({"hits": 0})
    unlocked = ToolExecutor(default_specs(store_b), serialize_writes=False)
    await unlocked.call_many(
        [{"name": "kv_add", "args": {"key": "hits", "delta": 1}},
         {"name": "kv_add", "args": {"key": "hits", "delta": 1}}]
    )
    print(f"\n  B. two kv_add(+1) WITHOUT the write lock: final hits={store_b.data['hits']}"
          f" (expected 2), store.writes={store_b.writes}")
    print("     both writes happened, one increment vanished: the classic lost update.")

    # C. the fix: the executor's write lock spans the whole read-modify-write.
    store_c = KVStore({"hits": 0})
    locked = ToolExecutor(default_specs(store_c), serialize_writes=True)
    await locked.call_many(
        [{"name": "kv_add", "args": {"key": "hits", "delta": 1}},
         {"name": "kv_add", "args": {"key": "hits", "delta": 1}}]
    )
    print(f"  C. same two kv_add WITH the write lock: final hits={store_c.data['hits']}"
          f" (expected 2), store.writes={store_c.writes}")
    print("     side effects counted directly: 2 writes, and the final state matches.")


# ---------------------------------------------------------------------------
# Demo 3: timeout / retry / cancellation
# ---------------------------------------------------------------------------


async def demo_fault_control() -> None:
    banner("DEMO 3 - TIMEOUT, RETRY (READS ONLY), CANCELLATION")
    store = KVStore({"hits": 1})
    executor = ToolExecutor(
        default_specs(store)
        | {
            "slow_write": make_slow_write_spec(store, hang_s=5.0),
            "flaky_read": make_flaky_read_spec(store, fail_times=2),
            "flaky_write": make_flaky_write_spec(store),
        },
        retry_policy=RetryPolicy(max_attempts=3),
    )

    print("  A. a hanging write under a 0.3s timeout:")
    print_envelope(await executor.call("slow_write", {"key": "hits", "value": 42}, timeout_s=0.3))
    print(f"     store.writes={store.writes} -> the hung write NEVER landed")

    print("\n  B. retry policy: readonly tools are retried, writes never are.")
    print_envelope(await executor.call("flaky_read", {"key": "hits"}))
    print(f"     flaky_read attempts recorded: {executor.log[-1].attempts} (failed twice, then ok)")
    print_envelope(await executor.call("flaky_write", {"key": "hits", "value": 5}))
    print(f"     flaky_write attempts recorded: {executor.log[-1].attempts}; "
          f"store.writes={store.writes} (that is the ONE landing from flaky_write)")
    print("     -> if the policy retried this write, the effect would be applied twice.")

    print("\n  C. cancellation of a hung write (asyncio task cancel):")
    writes_before_cancel = store.writes  # isolate the effect counter for this section
    task = asyncio.create_task(executor.call("slow_write", {"key": "hits", "value": 77}))
    await asyncio.sleep(0.15)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        print("     CancelledError propagated as expected")
    print(f"     store.writes delta across the cancel: {store.writes - writes_before_cancel} "
          f"-> the cancelled write did NOT land; last log outcome: {executor.log[-1].outcome}")


# ---------------------------------------------------------------------------
# Demo 4: PermissionGate -- model proposes, executor decides (LangGraph loop)
# ---------------------------------------------------------------------------


async def demo_permission_gate() -> None:
    banner("DEMO 4 - PERMISSIONGATE: MODEL PROPOSES, EXECUTOR DECIDES")
    store = KVStore({"hits": 3})
    permissions = {"kv_get": ALLOW, "unit_convert": ALLOW,
                   "kv_set": NEEDS_APPROVAL, "delete_all": FORBID}
    executor = ToolExecutor(
        {k: v for k, v in default_specs(store).items() if k in permissions},
        permissions=permissions,
    )
    graph = build_gated_loop(executor, store)
    show_graph(graph)
    final_state = await graph.ainvoke({"round": 0, "pending": [], "observations": [], "final": ""})

    print(f"\n  store after the run: {store.data} (kv_set denied -> hits NOT 99; delete_all denied -> intact)")
    print(f"  store.reads={store.reads} (only the allowed kv_get executed), store.writes={store.writes}")
    print(f"  rejection log: {[(r['tool'], r['reason']) for r in executor.rejections]}")
    print(f"  final message: {final_state['final']}")
    print(f"  observations fed back to the model: {len(final_state['observations'])}")
    print("  -> the model PROPOSED forbidden/unapproved actions; they were rejected")
    print("     AT THE EXECUTION LAYER with zero side effects, and the rejections")
    print("     came back as observations the loop had to handle.")


# ---------------------------------------------------------------------------
# Demo 5: sandbox isolation (subprocess, cwd, env, path guard)
# ---------------------------------------------------------------------------


async def demo_sandbox() -> None:
    banner("DEMO 5 - SANDBOX: SUBPROCESS, PINNED CWD, ENV WHITELIST, PATH GUARD")
    root = Path(tempfile.mkdtemp(prefix="l3_sandbox_"))
    try:
        sandbox = Sandbox(root)
        executor = ToolExecutor({"py_exec": make_py_exec_spec(sandbox)})

        print(f"  sandbox root: {root}")
        print("  A. in-bounds write (relative path, lands inside the sandbox):")
        envelope = await executor.call(
            "py_exec", {"code": "open('note.txt','w').write('inside')\nprint('wrote')"}
        )
        value = envelope["value"]
        print(f"     executed={value['executed']} returncode={value['returncode']} stdout={value['stdout']!r}")
        print(f"     note.txt exists inside sandbox: {(root / 'note.txt').exists()}")

        print("\n  B. out-of-bounds read (absolute path) -- rejected BEFORE any spawn:")
        spawns_before = sandbox.spawns
        envelope = await executor.call("py_exec", {"code": "print(open('/etc/hosts').read()[:20])"})
        value = envelope["value"]
        print(f"     violation: {value['violation']}")
        print(f"     executed={value['executed']}; spawns delta = {sandbox.spawns - spawns_before} (no process started)")

        print("\n  C. traversal write ('../../l3_escaped_probe.txt') -- also rejected:")
        escaped_target = (root / ".." / ".." / "l3_escaped_probe.txt").resolve()
        envelope = await executor.call(
            "py_exec", {"code": "open('../../l3_escaped_probe.txt','w').write('out')"}
        )
        value = envelope["value"]
        print(f"     violation: {value['violation']}; executed={value['executed']}")
        print(f"     escaped file created outside: {escaped_target.exists()}")

        print("\n  D. what the subprocess actually sees (cwd pinned, env whitelisted):")
        envelope = await executor.call(
            "py_exec", {"code": "import os\nprint(os.getcwd())\nprint(sorted(os.environ.keys()))"}
        )
        lines = envelope["value"]["stdout"].splitlines()
        print(f"     cwd  = {lines[0]}")
        print(f"     env  = {lines[1]}")
        print(f"     HOME present in sandbox env: {'HOME' in sandbox.env_whitelist} "
              f"(the parent process does have HOME)")
        print("  -> the guard is a first layer (a heuristic that can be obfuscated);")
        print("     real isolation adds OS-level mechanisms. What we CAN prove here:")
        print("     out-of-bounds access failed at the configured boundary, and the")
        print("     subprocess only ever saw the pinned cwd and the env whitelist.")
    finally:
        shutil.rmtree(root, ignore_errors=True)
        print(f"\n  cleaned up {root}")


# ---------------------------------------------------------------------------
# Demo 6: same tool, two boundaries -- direct call vs local MCP server
# ---------------------------------------------------------------------------


async def demo_mcp_contrast() -> None:
    banner("DEMO 6 - SAME TOOL, TWO BOUNDARIES: DIRECT CALL VS LOCAL MCP SERVER")
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    cases = [
        {"value": 3.0, "from_unit": "km", "to_unit": "m"},
        {"value": 12.0, "from_unit": "in", "to_unit": "cm"},
        {"value": 0.5, "from_unit": "ft", "to_unit": "mm"},
        {"value": 42.0, "from_unit": "cm", "to_unit": "m"},
        {"value": 1.0, "from_unit": "km", "to_unit": "ft"},
        {"value": 7.0, "from_unit": "mm", "to_unit": "cm"},
        {"value": 9.9, "from_unit": "m", "to_unit": "km"},
        {"value": 2.0, "from_unit": "m", "to_unit": "in"},
        {"value": 100.0, "from_unit": "mm", "to_unit": "m"},
        {"value": 5.0, "from_unit": "cm", "to_unit": "km"},
    ]

    # (a) direct: the very same function object, imported in-process.
    started = time.perf_counter()
    direct_results = [unit_convert(**case) for case in cases]
    direct_ms = (time.perf_counter() - started) * 1000
    print(f"  (a) direct in-process calls : {direct_ms:.3f} ms total "
          f"({direct_ms / len(cases) * 1000:.1f} us/call)")
    print(f"      sample: unit_convert(**{cases[0]}) -> {direct_results[0]}")

    # (b) via MCP: spawn subprocess, initialize handshake, discover, call.
    # errlog: route the SERVER's stderr into a temp file instead of letting it
    # spray into our console -- the error-contrast section below then shows
    # exactly where the lost error detail went. (errlog needs a real fileno,
    # so a StringIO does NOT work here -- probed the hard way.)
    params = StdioServerParameters(command=sys.executable, args=[str(LESSON_DIR / "mcp_server.py")])
    server_stderr = tempfile.TemporaryFile(mode="w+")
    t_spawn = time.perf_counter()
    async with stdio_client(params, errlog=server_stderr) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            startup_ms = (time.perf_counter() - t_spawn) * 1000
            print(f"\n  (b) MCP over stdio           : server={init.server_info.name} "
                  f"protocol={init.protocol_version}")
            print(f"      spawn + initialize       : {startup_ms:.1f} ms (paid once, not per call)")

            listed = (await session.list_tools()).tools
            for tool in listed:
                hints = tool.annotations
                props = list(tool.output_schema.get("properties", {})) if tool.output_schema else None
                print(f"      discovered via tools/list: {tool.name} "
                      f"read_only_hint={getattr(hints, 'read_only_hint', None)} "
                      f"output_schema_props={props}")
            print("      -> discovery and the read/write hint TRAVEL through the protocol.")

            started = time.perf_counter()
            mcp_results = []
            for case in cases:
                result = await session.call_tool("unit_convert", case)
                if result.is_error:
                    raise RuntimeError(f"MCP call failed: {result.content}")
                mcp_results.append(result.structured_content)
            mcp_ms = (time.perf_counter() - started) * 1000
            print(f"      {len(cases)} warm MCP calls   : {mcp_ms:.2f} ms total "
                  f"({mcp_ms / len(cases):.2f} ms/call)")

            print(f"\n  results identical across the boundary: {direct_results == mcp_results}")
            print(f"  per-call cost: MCP {mcp_ms / len(cases):.2f} ms vs direct "
                  f"{direct_ms / len(cases) * 1000:.1f} us "
                  f"(~{mcp_ms / max(direct_ms, 1e-9):.0f}x for this trivial tool)")
            print("  the boundary adds: process spawn (once), JSON-RPC serialization per")
            print("  call, the initialize handshake, and tool listing/discovery.")

            print("\n  error detail across the boundary (to_unit='furlong'):")
            try:
                unit_convert(value=1.0, from_unit="m", to_unit="furlong")
            except ValueError as exc:
                print(f"    direct: ValueError: {exc}")
            result = await session.call_tool(
                "unit_convert", {"value": 1.0, "from_unit": "m", "to_unit": "furlong"}
            )
            text = result.content[0].text if result.content else ""
            print(f"    MCP   : is_error={result.is_error}, content={text!r}")
            # Give the stderr pump a moment, then show where the detail went.
            await asyncio.sleep(0.2)
            server_stderr.seek(0)
            tail = server_stderr.read().strip().splitlines()
            print(f"    server stderr (captured via errlog, last line): {tail[-1] if tail else '(empty)'}")
            print("    -> the precise message stayed on the server (its stderr); the")
            print("       client sees a generic error envelope. Boundaries drop detail.")
    server_stderr.close()


# ---------------------------------------------------------------------------
# Demo 7: failure attribution from the execution log alone
# ---------------------------------------------------------------------------


async def demo_traceability() -> None:
    banner("DEMO 7 - FAILURE ATTRIBUTION FROM THE EXECUTION LOG ALONE")
    store = KVStore({"alpha": 1})
    executor = ToolExecutor(
        default_specs(store) | {"slow_write": make_slow_write_spec(store, hang_s=5.0)}
    )

    print("  a scripted multi-step run with two failures planted in it:")
    plan: list[dict[str, Any]] = [
        {"name": "kv_get", "args": {"key": "alpha"}},
        {"name": "kv_get", "args": {"key": "beta"}},
        {"name": "unit_convert", "args": {"value": 2.0, "from_unit": "m", "to_unit": "mile"}},  # invalid
        {"name": "kv_set", "args": {"key": "alpha", "value": 5}},
        {"name": "slow_write", "args": {"key": "alpha", "value": 9}, "timeout_s": 0.2},  # timeout
        {"name": "unit_convert", "args": {"value": 2.0, "from_unit": "m", "to_unit": "ft"}},
    ]
    for step in plan:
        await executor.call(step["name"], step["args"], timeout_s=step.get("timeout_s"))

    print_log(executor.log)

    # Attribution: answer these ONLY from the log + effect counters.
    failed = [r for r in executor.log if r.outcome not in {"ok", "retrying"}]
    print("\n  attribution from the log alone:")
    for record in failed:
        print(f"    - step {executor.log.index(record) + 1}: {record.tool}({json.dumps(record.args)}) "
              f"failed with {record.outcome}")
    print(f"    - side effects that ACTUALLY landed: store.writes={store.writes} "
          f"(the timed-out slow_write is NOT among them)")
    print(f"    - final state: {store.data}")
    print("  -> the log carries tool, args, decision, outcome, attempts and latency;")
    print("     combined with direct effect counters it explains the run end-to-end.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def run_all_demos() -> None:
    await demo_contract()
    await demo_concurrency()
    await demo_fault_control()
    await demo_permission_gate()
    await demo_sandbox()
    await demo_mcp_contrast()
    await demo_traceability()


if __name__ == "__main__":
    # One event loop for everything: the executor's asyncio.Lock binds to the
    # loop on first use, so reusing one executor across asyncio.run() calls
    # would raise "... bound to a different event loop".
    asyncio.run(run_all_demos())
    banner("done", width=72)
