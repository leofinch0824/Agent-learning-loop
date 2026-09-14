"""L0 - Why Agents: single shot, fixed workflow, agent loop.

The scenario is the course through-line project in miniature: read one
snippet of PCB process text, extract three parameters (board thickness,
copper weight, solder mask type) and validate their units / ranges. The SAME
task, the SAME tools and the SAME scoring run through three control regimes
-- the only variable under test is WHO decides the order of steps:

    1. single shot      one model call does extraction AND validation in place
                        (control flow: none; it is one prompt/response)

    2. fixed workflow   parse -> extract (model) -> validate (code) -> format
                        (control flow: written in Python; model paid once)

    3. agent loop       a plain while-loop runtime: the model emits tool_calls,
                        the harness executes tools and feeds observations back,
                        until a final answer / no-progress / budget stop
                        (control flow: decided by the model, enforced by the
                        harness)

    goal (task + stop conditions)
      |
      v
    +--------+  action: tool_call      +---------+  execute   +-----------+
    |  model | ---------------------> | runtime | ---------> |   tools   |
    |        |                        | harness |            | pure code |
    |        | <--------------------- +---------+ <--------- +-----------+
    +--------+  feedback: observation appended to the message history
      |
      v  loop until: final answer | no progress | budget exhausted

This lesson deliberately does NOT import langgraph: everything here predates
the framework. L1 re-expresses the same shapes with State / Node / Edge.

Run it:

    poetry run python lessons/l0_agent_foundations/main.py

Design promise: every demo prints raw runtime behaviour (each model call,
each tool dispatch, each stop decision) rather than a summary, and every
number in the comparison tables comes from a direct counter -- never
inferred from the final state.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lib import banner  # noqa: E402


def say(who: str, message: str, *, limit: int = 200) -> None:
    """Runtime-side console output, truncated so the tables stay readable.

    flush=True keeps ordering stable when stdout is piped, matching L1's say().
    """
    text = message if len(message) <= limit else message[: limit - 3] + "..."
    print(f"    [{who}] {text}", flush=True)


# ---------------------------------------------------------------------------
# 1. The task: extract + validate three PCB process parameters
# ---------------------------------------------------------------------------

# Free text: what a human writes into an order note. Position, spacing and
# filler words vary -- the case where a model's text understanding actually
# earns its latency.
FREE_TEXT_TASK = "订单备注：板材 FR-4；板厚 1.6 mm；外层铜厚 1 oz；阻焊类型 LPI（液态感光油墨）。"

# Deterministic variant: what a machine writes into a log line. Fixed grammar,
# zero ambiguity -- regex is exact here, so any model in the loop is overhead.
LOG_LINE_TASK = "THK=1.6mm|CU=1oz|MASK=LPI"

PARAM_NAMES = ("board_thickness", "copper_weight", "solder_mask")

# The spec a correct answer must satisfy. Units are part of the contract:
# "1.6" without a unit, or copper range-checked in mm, is WRONG -- exactly the
# check a single-shot model tends to fumble (see SINGLE_SHOT_SCRIPT below).
SPEC: dict[str, dict[str, Any]] = {
    "board_thickness": {"unit": "mm", "min": 0.2, "max": 4.0},
    "copper_weight": {"unit": "oz", "min": 0.25, "max": 5.0},
    "solder_mask": {"enum": ("LPI", "DRY_FILM", "NONE")},
}

# Ground truth for scoring: value AND unit AND verdict must all match.
EXPECTED: dict[str, dict[str, Any]] = {
    "board_thickness": {"value": 1.6, "unit": "mm", "verdict": "pass"},
    "copper_weight": {"value": 1.0, "unit": "oz", "verdict": "pass"},
    "solder_mask": {"value": "LPI", "unit": None, "verdict": "pass"},
}


# ---------------------------------------------------------------------------
# 2. Instrumentation: direct counters + simulated latency units
# ---------------------------------------------------------------------------

# Simulated latency in RELATIVE units, not milliseconds: one model round-trip
# dominates end-to-end latency (network + prompt processing + generation),
# while a tool execution is an in-process function call. The 5:1:1 ratio is
# the declared teaching assumption; L7 replaces it with measured telemetry.
MODEL_CALL_LATENCY = 5.0
TOOL_EXEC_LATENCY = 1.0
CODE_STAGE_LATENCY = 1.0


@dataclass
class Metrics:
    """What each regime actually did, counted where it happens.

    DESIGN.md rule: asserted behaviour must be counted directly (counter /
    closure), never reconstructed from the final state. These counters plus
    the latency sum are the entire observability story of L0 -- no MLflow by
    design; observability is layered in progressively from L1.
    """

    model_calls: int = 0
    tool_executions: int = 0
    loop_iterations: int = 0  # completed model turns inside a loop; 0 for pipelines
    latency_units: float = 0.0

    def model_call(self) -> None:
        self.model_calls += 1
        self.latency_units += MODEL_CALL_LATENCY

    def tool_exec(self) -> None:
        self.tool_executions += 1
        self.latency_units += TOOL_EXEC_LATENCY

    def code_stage(self, n: int = 1) -> None:
        self.latency_units += n * CODE_STAGE_LATENCY


@dataclass
class RunResult:
    """One completed run: the answer, how it was produced, and why it stopped."""

    approach: str
    answer: dict[str, Any] | None
    metrics: Metrics
    stop_reason: str  # "completed" | "final_answer" | "no_progress" | "budget_exhausted"


# ---------------------------------------------------------------------------
# 3. Shared tools: the same capabilities in every regime
# ---------------------------------------------------------------------------

# One regex per parameter, deliberately format-agnostic (it matches both the
# free-text note and the log line). Capture the value WITH its unit, because
# the unit is part of the contract.
RAW_PATTERNS: dict[str, str] = {
    "board_thickness": r"(\d+(?:\.\d+)?\s*mm)",
    "copper_weight": r"(\d+(?:\.\d+)?\s*oz)",
    "solder_mask": r"(LPI|DRY_FILM|NONE)",
}


def _split_value_unit(raw: str) -> tuple[Any, str | None]:
    """'1.6 mm' and '1.6mm' -> (1.6, 'mm'); 'LPI' -> ('LPI', None). Code, not model.

    Whitespace-tolerant on purpose: the free-text note writes '1.6 mm', the
    machine log line writes '1.6mm' -- the split must not care.
    """
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(mm|oz)?\s*", raw)
    if m:
        return float(m.group(1)), m.group(2)
    return raw, None


def make_tools(task_text: str) -> dict[str, Callable[[dict], str]]:
    """Build the tool registry for one task.

    The SAME three functions serve every regime: the fixed workflow calls them
    directly as library functions (control flow in code), the agent runtime
    dispatches them on the model's behalf (control flow in model output). What
    changes is the dispatch policy, never the capability -- that is the
    tools / runtime split this lesson wants to make visible.
    """

    def read_source(args: dict) -> str:
        # The model is stateless: in the agent regime it must LOOK before it
        # can act. This is what makes turn 2 depend on turn 1's observation --
        # the feedback in goal / action / observation / feedback.
        return task_text

    def extract_parameters(args: dict) -> str:
        # `args["text"]` lets the agent pass in what it just read (so the
        # observation genuinely flows into the next action); direct callers
        # omit it and fall back to the closure.
        text = args.get("text", task_text)
        found: dict[str, str | None] = {}
        for name, pattern in RAW_PATTERNS.items():
            m = re.search(pattern, text)
            found[name] = m.group(1) if m else None
        return json.dumps(found, ensure_ascii=False)

    def check_spec(args: dict) -> str:
        # Deterministic validation: unit + range, or enum membership. A model
        # CAN imitate this in one shot, but nothing forces it to -- code is
        # the only place where the check becomes a guarantee.
        name, value, unit = args["name"], args["value"], args.get("unit")
        spec = SPEC.get(name)
        if spec is None:
            return json.dumps({"verdict": "error", "detail": f"unknown parameter {name!r}"}, ensure_ascii=False)
        if "enum" in spec:
            ok = value in spec["enum"]
            return json.dumps({"verdict": "pass" if ok else "fail", "detail": f"{value!r} enum={spec['enum']}"}, ensure_ascii=False)
        if unit != spec["unit"]:
            return json.dumps(
                {"verdict": "fail", "detail": f"unit mismatch: got {unit!r}, spec wants {spec['unit']!r}"},
                ensure_ascii=False,
            )
        ok = spec["min"] <= value <= spec["max"]
        return json.dumps(
            {"verdict": "pass" if ok else "fail", "detail": f"{value} {unit} vs range [{spec['min']}, {spec['max']}] {spec['unit']}"},
            ensure_ascii=False,
        )

    return {"read_source": read_source, "extract_parameters": extract_parameters, "check_spec": check_spec}


# ---------------------------------------------------------------------------
# 4. The fake model: a scripted response queue
# ---------------------------------------------------------------------------


@dataclass
class ModelResponse:
    """One model turn: either tool_calls (act) or a final answer (answer)."""

    final: dict[str, Any] | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


def reply_tool(name: str, args: dict[str, Any]) -> ModelResponse:
    return ModelResponse(tool_calls=[{"name": name, "args": args}])


def reply_final(answer: dict[str, Any]) -> ModelResponse:
    return ModelResponse(final=answer)


class ScriptedModel:
    """Deterministic fake model: pops scripted responses in order.

    Fakes are per-lesson props and stay out of lib/ (DESIGN.md): they let the
    tests count model calls exactly and never touch the network. A leftover
    script is fine (budget stops consume fewer turns than scripted); running
    off the end of the script is a bug and raises.
    """

    def __init__(self, script: list[ModelResponse]) -> None:
        self.script = list(script)
        self.calls = 0  # the model's own count, used to cross-check Metrics
        self.message_log: list[list[dict[str, Any]]] = []  # messages seen per call

    def __call__(self, messages: list[dict[str, Any]]) -> ModelResponse:
        self.calls += 1
        self.message_log.append(list(messages))
        if not self.script:
            raise AssertionError(f"script exhausted at call {self.calls}")
        return self.script.pop(0)


# ---------------------------------------------------------------------------
# 5. Approach 1 -- single shot: one model call does everything
# ---------------------------------------------------------------------------

SINGLE_SHOT_INSTRUCTIONS = """\
你是 PCB 工艺参数审核员。从下面的订单文本中抽取三个参数并校验，只输出 JSON：
- board_thickness：单位 mm，合法范围 0.2 到 4.0
- copper_weight：单位 oz，合法范围 0.25 到 5.0（换算提示：1 oz 铜厚约 35 um）
- solder_mask：枚举值 LPI / DRY_FILM / NONE
输出格式：{"参数名": {"value": ..., "unit": ..., "verdict": "pass 或 fail", "note": "原因"}}

订单文本：
{task}"""


def build_single_shot_prompt(task_text: str) -> str:
    """The one prompt shared by the offline fake and the live L0 check.

    Plain replace, not str.format: the template contains literal JSON braces
    ({"参数名": ...}) that .format would treat as placeholders.
    """
    return SINGLE_SHOT_INSTRUCTIONS.replace("{task}", task_text)


# The scripted single-shot answer carries THE classic single-shot failure: the
# model did the unit conversion in its head (1 oz -> "35") but wrote mm instead
# of um, range-checked 35 against the 4.0 board-thickness limit and confidently
# reported a fail. Nothing downstream can catch this -- there is no downstream.
# Quality: 2/3.
SINGLE_SHOT_SCRIPT = [
    reply_final(
        {
            "board_thickness": {"value": 1.6, "unit": "mm", "verdict": "pass", "note": ""},
            "copper_weight": {"value": 35.0, "unit": "mm", "verdict": "fail", "note": "1 oz 约 35，超过 4.0 上限"},
            "solder_mask": {"value": "LPI", "unit": None, "verdict": "pass", "note": ""},
        }
    ),
]


def run_single_shot(task_text: str, model: ScriptedModel, *, verbose: bool = True) -> RunResult:
    """Everything (extraction + validation + formatting) inside one call.

    Cheapest and lowest-latency, perfectly fine when the task is one step and
    nobody needs a guarantee. The risk is silent: the correctness of the
    VALIDATION now depends on the model following instructions in the prompt.
    """
    metrics = Metrics()
    prompt = build_single_shot_prompt(task_text)
    if verbose:
        say("prompt", prompt.splitlines()[0] + " ...")
    metrics.model_call()
    metrics.loop_iterations = 1  # a single shot is an agent loop capped at one turn
    response = model([{"role": "user", "content": prompt}])
    if verbose:
        say("model", f"final answer in 1 call (copper verdict: {response.final['copper_weight']['verdict']})")
    return RunResult("single_shot", response.final, metrics, "completed")


# ---------------------------------------------------------------------------
# 6. Approach 2 -- fixed workflow: control flow in code, model paid once
# ---------------------------------------------------------------------------

WORKFLOW_EXTRACT_PROMPT = "抽取以下文本中 board_thickness / copper_weight / solder_mask 的原始值（带单位），只输出 JSON：\n{task}"

# The workflow spends its ONE model call on the step models are actually good
# at (reading free text) and returns the same raw shape the regex tool would.
WORKFLOW_EXTRACT_SCRIPT = [
    reply_final({"board_thickness": "1.6 mm", "copper_weight": "1 oz", "solder_mask": "LPI"}),
]


def run_fixed_workflow(task_text: str, extract_model: ScriptedModel, *, verbose: bool = True) -> RunResult:
    """parse -> extract (model, once) -> validate (code) -> format.

    The model is called ONLY where free-text understanding is needed. Splitting
    value/unit and checking the spec happen in the same check_spec tool the
    agent runtime dispatches to: quality stops depending on the model's
    arithmetic the moment validation moves into code. The price is pipeline
    structure -- more code to write, more stages to wait for.
    """
    metrics = Metrics()
    tools = make_tools(task_text)

    # Stage 1 -- parse (code): normalise CJK separators so downstream matching
    # is stable. Trivial here; in the real project this is the PDF/image
    # cleanup stage, and it is STILL the right place for deterministic work.
    normalized = task_text.replace("；", ";")
    metrics.code_stage()
    if verbose:
        say("workflow", "stage 1 parse (code): separators normalised")

    # Stage 2 -- extract (model): the single paid call.
    metrics.model_call()
    raw = extract_model(
        [{"role": "user", "content": WORKFLOW_EXTRACT_PROMPT.replace("{task}", normalized)}]
    )
    raw_values: dict[str, str] = raw.final or {}
    if verbose:
        say("workflow", f"stage 2 extract (model x1): {json.dumps(raw_values, ensure_ascii=False)}")

    # Stage 3 -- validate (code): the same tool the agent uses, called directly.
    answer: dict[str, Any] = {}
    for name in PARAM_NAMES:
        value, unit = _split_value_unit(raw_values[name])
        verdict = json.loads(tools["check_spec"]({"name": name, "value": value, "unit": unit}))
        metrics.tool_exec()
        answer[name] = {"value": value, "unit": unit, "verdict": verdict["verdict"], "note": verdict["detail"]}
        if verbose:
            say("workflow", f"stage 3 validate (code): {name} -> {verdict['verdict']} ({verdict['detail']})")

    # Stage 4 -- format (code).
    metrics.code_stage()
    if verbose:
        say("workflow", "stage 4 format (code): answer assembled")
    return RunResult("fixed_workflow", answer, metrics, "completed")


# ---------------------------------------------------------------------------
# 7. Approach 3 -- agent loop: model decides, harness enforces
# ---------------------------------------------------------------------------

AGENT_SYSTEM_PROMPT = (
    "你是 PCB 工艺参数审核 agent。可用工具：read_source（读订单原文）、"
    "extract_parameters（正则抽取）、check_spec（规格校验）。"
    "每轮先行动（一次 tool_call），观察结果后再决定下一步；信息齐全时输出最终 JSON 答案。"
    "禁止在没有观察的情况下臆造参数值。"
)

CORRECT_ANSWER: dict[str, Any] = {
    "board_thickness": {"value": 1.6, "unit": "mm", "verdict": "pass", "note": ""},
    "copper_weight": {"value": 1.0, "unit": "oz", "verdict": "pass", "note": ""},
    "solder_mask": {"value": "LPI", "unit": None, "verdict": "pass", "note": ""},
}


def agent_script_for(text: str) -> list[ModelResponse]:
    """A competent scripted agent: look, extract, validate each, answer.

    Note how the args of turn 2 embed the text observed in turn 1 -- that is
    the observation -> next-action dependency the loop exists to provide, and
    what test_main.py asserts directly on message_log.
    """
    return [
        reply_tool("read_source", {}),
        reply_tool("extract_parameters", {"text": text}),
        reply_tool("check_spec", {"name": "board_thickness", "value": 1.6, "unit": "mm"}),
        reply_tool("check_spec", {"name": "copper_weight", "value": 1.0, "unit": "oz"}),
        reply_tool("check_spec", {"name": "solder_mask", "value": "LPI"}),
        reply_final(CORRECT_ANSWER),
    ]


# On the deterministic task even the single shot is correct -- the point of
# DEMO 2 is pure cost, so the fake model is scripted competent here.
DETERMINISTIC_SINGLE_SHOT_SCRIPT = [reply_final(CORRECT_ANSWER)]


def run_agent_loop(
    task_text: str,
    model: ScriptedModel,
    tools: dict[str, Callable[[dict], str]],
    *,
    max_steps: int = 8,
    verbose: bool = True,
) -> RunResult:
    """A ~30-line runtime/harness: the whole 'agent framework' in miniature.

    The model only ever PROPOSES actions (tool_calls); the harness owns the
    loop, executes tools, appends observations to the message history, and
    enforces the three stop conditions every autonomous loop needs:

      1. success     -- the model returns a final answer
      2. no progress -- the same (tool, args) pair appears twice
      3. budget      -- max_steps model turns exceeded

    Autonomy without 2 and 3 is an unbounded do-while over a network service.
    NB: lib.step() is NOT used here because it prints "super-step", LangGraph
    vocabulary that belongs to L1; L0 prints its own loop-iteration markers.
    """
    metrics = Metrics()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": "任务：抽取并校验订单工艺参数；订单原文需先用 read_source 获取。"},
    ]
    seen: set[tuple[str, str]] = set()

    for iteration in range(1, max_steps + 1):
        metrics.loop_iterations = iteration
        if verbose:
            print(f"\n  -- iteration {iteration} --", flush=True)
        metrics.model_call()
        response = model(messages)

        if response.final is not None:  # stop 1: success
            messages.append({"role": "assistant", "content": json.dumps(response.final, ensure_ascii=False)})
            if verbose:
                say("model", "action: final answer")
                say("runtime", f"stop: final_answer after {iteration} iterations")
            return RunResult("agent_loop", response.final, metrics, "final_answer")

        messages.append({"role": "assistant", "tool_calls": response.tool_calls})
        if verbose:
            for call in response.tool_calls:
                say("model", f"action: {call['name']} {json.dumps(call['args'], ensure_ascii=False)}")
        for call in response.tool_calls:
            key = (call["name"], json.dumps(call["args"], sort_keys=True, ensure_ascii=False))
            if key in seen:  # stop 2: no progress
                if verbose:
                    say("runtime", f"stop: no_progress (repeat {call['name']} with identical args)")
                return RunResult("agent_loop", None, metrics, "no_progress")
            seen.add(key)
            fn = tools.get(call["name"])
            # Unknown tools become error observations instead of crashes: a bad
            # proposal is feedback the model can correct, not an exception.
            observation = fn(call["args"]) if fn else json.dumps({"error": f"unknown tool {call['name']!r}"})
            metrics.tool_exec()
            messages.append({"role": "tool", "content": observation})
            if verbose:
                say("runtime", f"execute {call['name']} -> observation: {observation}")

    if verbose:  # stop 3: budget
        say("runtime", f"stop: budget_exhausted (max_steps={max_steps})")
    return RunResult("agent_loop", None, metrics, "budget_exhausted")


# ---------------------------------------------------------------------------
# 8. Baselines: scoring, comparison table, pure-code counterexample
# ---------------------------------------------------------------------------


def score_answer(answer: dict[str, Any] | None) -> tuple[int, int]:
    """(correct, total): value AND unit AND verdict must all match EXPECTED."""
    correct = 0
    for name, exp in EXPECTED.items():
        got = (answer or {}).get(name)
        if (
            got
            and got.get("value") == exp["value"]
            and got.get("unit") == exp["unit"]
            and got.get("verdict") == exp["verdict"]
        ):
            correct += 1
    return correct, len(EXPECTED)


def print_comparison_table(title: str, rows: list[RunResult]) -> None:
    """The comparable minimal baseline: one GFM table, paste-able into notes.

    Quality and counts are printed side by side on purpose: an approach that
    wins on latency must also be judged on quality, and vice versa.
    """
    print(f"\n  {title}", flush=True)
    print("  | approach | model calls | tool execs | loop iters | latency units | quality | stop reason |", flush=True)
    print("  | --- | --- | --- | --- | --- | --- | --- |", flush=True)
    for r in rows:
        correct, total = score_answer(r.answer)
        print(
            f"  | {r.approach} | {r.metrics.model_calls} | {r.metrics.tool_executions} "
            f"| {r.metrics.loop_iterations} | {r.metrics.latency_units:.1f} "
            f"| {correct}/{total} | {r.stop_reason} |",
            flush=True,
        )


def parse_with_pure_code(task_text: str, *, verbose: bool = True) -> RunResult:
    """The 'do not build an agent here' baseline: zero model calls.

    On a fixed-grammar input the regex tool is exact, so the cheapest correct
    answer is: call the capabilities as plain functions and go home. This is
    the numbers-first rebuttal to 'add an agent just in case'.
    """
    metrics = Metrics()
    tools = make_tools(task_text)
    raw_values = json.loads(tools["extract_parameters"]({}))
    metrics.tool_exec()
    answer: dict[str, Any] = {}
    for name in PARAM_NAMES:
        value, unit = _split_value_unit(raw_values[name])
        verdict = json.loads(tools["check_spec"]({"name": name, "value": value, "unit": unit}))
        metrics.tool_exec()
        answer[name] = {"value": value, "unit": unit, "verdict": verdict["verdict"], "note": verdict["detail"]}
    if verbose:
        say("pure_code", f"0 model calls, {metrics.tool_executions} tool executions, answer assembled")
    return RunResult("pure_code_parser", answer, metrics, "completed")


# ---------------------------------------------------------------------------
# 9. Demos
# ---------------------------------------------------------------------------


def demo_free_text() -> None:
    banner("DEMO 1 - one free-text task, three control regimes")
    print(f"  task     : {FREE_TEXT_TASK}", flush=True)
    print("  variable : WHO decides the order of steps (nothing else changes)\n", flush=True)

    single = run_single_shot(FREE_TEXT_TASK, ScriptedModel(SINGLE_SHOT_SCRIPT))
    print(flush=True)
    fixed = run_fixed_workflow(FREE_TEXT_TASK, ScriptedModel(WORKFLOW_EXTRACT_SCRIPT))
    print(flush=True)
    agent = run_agent_loop(
        FREE_TEXT_TASK, ScriptedModel(agent_script_for(FREE_TEXT_TASK)), make_tools(FREE_TEXT_TASK)
    )

    print_comparison_table("baseline: free-text task", [single, fixed, agent])
    print(
        "\n  reading: single shot is 7x cheaper than the loop, but its copper verdict is\n"
        "  wrong (unit hallucination -> 2/3); the workflow fixes quality by moving\n"
        "  validation into code, paying 2 code stages + 3 tool calls; the agent loop\n"
        "  also fixes it -- at 6 model calls. Freedom bought quality at 7x the latency.",
        flush=True,
    )


def demo_deterministic() -> None:
    banner("DEMO 2 - counterexample: fixed grammar, deterministic task")
    print(f"  task     : {LOG_LINE_TASK}", flush=True)
    print("  question : does ANY model in the loop buy quality here?\n", flush=True)

    parser = parse_with_pure_code(LOG_LINE_TASK)
    print(flush=True)
    single = run_single_shot(LOG_LINE_TASK, ScriptedModel(DETERMINISTIC_SINGLE_SHOT_SCRIPT))
    print(flush=True)
    agent = run_agent_loop(
        LOG_LINE_TASK, ScriptedModel(agent_script_for(LOG_LINE_TASK)), make_tools(LOG_LINE_TASK)
    )

    print_comparison_table("baseline: deterministic task", [parser, single, agent])
    print(
        "\n  reading: all three are 3/3 correct. The agent burns 6 model calls and 5 tool\n"
        "  executions to reach the answer pure code reaches with 0 model calls -- and it\n"
        "  is the only regime that can additionally hit the no_progress / budget failures\n"
        "  of DEMO 3. This is the scenario where adding agent complexity is strictly negative.",
        flush=True,
    )


# A stuck model: after reading the source it re-issues the SAME extract call.
# Without the seen-set guard this loop would spin until the budget guard fires
# (or forever, if the budget were removed too -- break-it exercise 1).
NO_PROGRESS_SCRIPT = [
    reply_tool("read_source", {}),
    reply_tool("extract_parameters", {"text": FREE_TEXT_TASK}),
    reply_tool("extract_parameters", {"text": FREE_TEXT_TASK}),
]

# A thrashing model: every turn re-validates copper with slightly different
# arguments, so the no-progress guard never fires and only the budget stops it.
BUDGET_SCRIPT = [
    reply_tool("check_spec", {"name": "copper_weight", "value": 6.0, "unit": "oz"}),
    reply_tool("check_spec", {"name": "copper_weight", "value": 6.1, "unit": "oz"}),
    reply_tool("check_spec", {"name": "copper_weight", "value": 6.2, "unit": "oz"}),
    reply_tool("check_spec", {"name": "copper_weight", "value": 6.3, "unit": "oz"}),
]


def demo_failure_modes() -> None:
    banner("DEMO 3 - failure modes only the agent has")
    print("  autonomy means the RUNTIME must own termination: success / no-progress / budget\n", flush=True)
    tools = make_tools(FREE_TEXT_TASK)

    stuck = run_agent_loop(FREE_TEXT_TASK, ScriptedModel(NO_PROGRESS_SCRIPT), tools)
    correct, total = score_answer(stuck.answer)
    print(
        f"\n  stuck model  -> stop={stuck.stop_reason}, model calls={stuck.metrics.model_calls}, "
        f"tool execs={stuck.metrics.tool_executions}, quality={correct}/{total}",
        flush=True,
    )

    thrash = run_agent_loop(FREE_TEXT_TASK, ScriptedModel(BUDGET_SCRIPT), tools, max_steps=3)
    correct, total = score_answer(thrash.answer)
    print(
        f"  thrash model -> stop={thrash.stop_reason}, model calls={thrash.metrics.model_calls}, "
        f"tool execs={thrash.metrics.tool_executions}, quality={correct}/{total}",
        flush=True,
    )
    print(
        "\n  reading: both runs end with NO answer (0/3). The fixed workflow cannot spin\n"
        "  because it does not loop; the single shot cannot either. The freedom the agent\n"
        "  gained in DEMO 1 is exactly this new failure surface -- termination design is\n"
        "  part of the agent, not an afterthought.",
        flush=True,
    )


if __name__ == "__main__":
    demo_free_text()
    demo_deterministic()
    demo_failure_modes()
    banner("done", width=72)
