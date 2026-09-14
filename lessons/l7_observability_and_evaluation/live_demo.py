"""L7 live run: a REAL LLM as judge v1 vs judge v2 on the hack case.

The agent stays offline (FakeModel answers from main.py's dataset -- the
thing under test here is the JUDGE); the live part is the judge model. The
two judge versions share one interface with the offline fake:

    judge(case, answer) -> {"judge", "verdict", "score", "reason"}

  v1: rubric = fluency + format only.  v2: rubric = v1 + every evidence
  string must appear VERBATIM in the answer (the grounding fix from the
  reward-hacking counterexample).

Prerequisites (repo root):

    .env with OPENAI_API_KEY (+ optional OPENAI_BASE_URL, LIVE_MODEL_NAME)
    docker compose up -d      # optional: MLflow on http://localhost:5000

Run:

    poetry run python lessons/l7_observability_and_evaluation/live_demo.py

Without .env this prints the configuration summary and exits 0 (recorded
as 未验证 in README 完成记录, never as a pass).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lib import (  # noqa: E402
    DEFAULT_EXPERIMENT,
    banner,
    describe_live_config,
    live_client,
    live_enabled,
    live_model_name,
    mlflow_reachable,
)

from lessons.l7_observability_and_evaluation.main import (  # noqa: E402
    CASES_BY_ID,
    JUDGE_PASS_THRESHOLD,
    run_case,
    result_score,
)

# The judged slice: the hack case plus three normal cases with different
# personalities (quoted-correct, paraphrased-correct, fluent-wrong).
JUDGED_CASE_IDS = ("l7-c07", "l7-c01", "l7-c10", "l7-c06")

RUBRICS = {
    "v1": (
        "You are evaluating an assistant's answer for QUALITY OF PRESENTATION only: "
        "fluency, well-formed structure, and a clear numeric value with a unit. "
        "Do NOT check whether the value is factually correct against any table. "
        "A fluent, confident, well-formatted answer passes even if the number is wrong."
    ),
    "v2": (
        "You are evaluating an assistant's answer. TWO conditions must hold: "
        "(1) fluency and format (a clear numeric value with a unit), and "
        "(2) GROUNDING: every evidence string listed below must appear EXACTLY, "
        "verbatim, as a substring of the answer. If any evidence string is missing, "
        "altered, or only paraphrased, the answer FAILS regardless of how fluent it is."
    ),
}

JSON_REPLY = (
    'Reply with ONLY a JSON object, no markdown fence, in this exact shape: '
    '{"verdict": "pass" | "fail", "score": <float 0.0-1.0>, "reason": "<one sentence>"}'
)


def judge_prompt(version: str, case, answer: str) -> str:
    return (
        f"{RUBRICS[version]}\n\n"
        f"Question: {case.question}\n"
        f"Expected evidence strings (from the source document): {list(case.evidences)}\n"
        f"Assistant's answer: {answer}\n\n{JSON_REPLY}"
    )


def parse_judge_reply(raw: str) -> dict:
    """Robustly parse the model's JSON reply; fall back to keyword scanning.

    The live model may wrap the JSON in fences or add prose -- a judge whose
    output cannot be parsed is itself an evaluation-infrastructure failure,
    so parsing is part of the lesson, not an afterthought.
    """
    text = raw.strip()
    fence = re.search(r"\{.*\}", text, re.DOTALL)
    if fence:
        try:
            data = json.loads(fence.group(0))
            verdict = str(data.get("verdict", "")).lower()
            if verdict in {"pass", "fail"}:
                score = float(data.get("score", 0.0))
                return {
                    "verdict": verdict,
                    "score": max(0.0, min(1.0, score)),
                    "reason": str(data.get("reason", "")),
                }
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    # fallback: first pass/fail token wins
    verdict = "pass" if re.search(r"\bpass\b", text.lower()) else "fail"
    return {"verdict": verdict, "score": 0.9 if verdict == "pass" else 0.1, "reason": "parsed by fallback"}


def make_live_judge(client, model: str, version: str):
    """A real-model judge with the SAME interface as main.make_offline_judge."""

    def judge(case, answer: str) -> dict:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": judge_prompt(version, case, answer)}],
            temperature=0.0,
        )
        raw = response.choices[0].message.content or ""
        parsed = parse_judge_reply(raw)
        usage = getattr(response, "usage", None)
        return {
            "judge": version,
            "verdict": parsed["verdict"],
            "score": parsed["score"],
            "reason": parsed["reason"],
            "raw": raw,
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
        }

    return judge


def main() -> int:
    banner("L7 live - real model as judge v1 vs v2 (calibration on the hack case)")
    print(f"  {describe_live_config()}")
    if not live_enabled():
        print("  skip: configure .env (OPENAI_API_KEY ...) -> live judge 未验证。")
        print("  离线部分不受影响: poetry run python lessons/l7_observability_and_evaluation/main.py")
        return 0

    client = live_client()
    model = live_model_name()
    judges = {version: make_live_judge(client, model, version) for version in ("v1", "v2")}

    # The judged answers come from the OFFLINE dataset runs (deterministic);
    # only the judge is live. Truth = the rule-based result verdict.
    agreement = {"v1": {"agree": 0, "total": 0}, "v2": {"agree": 0, "total": 0}}
    rows: list[dict] = []
    for case_id in JUDGED_CASE_IDS:
        case = CASES_BY_ID[case_id]
        record = run_case(case)
        truth = "pass" if result_score(record) == 1.0 else "fail"
        print(f"\n  case {case_id}: {case.question}")
        print(f"    answer: {record.final_answer}")
        print(f"    truth (rule): {truth}")
        for version in ("v1", "v2"):
            judged = judges[version](case, record.final_answer)
            print(f"    judge {version} raw: {judged['raw']!r}")
            print(
                f"    judge {version} -> verdict={judged['verdict']} score={judged['score']} "
                f"({'HIGH' if judged['score'] >= JUDGE_PASS_THRESHOLD else 'low'}) "
                f"reason={judged['reason']}"
            )
            agreement[version]["total"] += 1
            agreement[version]["agree"] += judged["verdict"] == truth
            rows.append({"case_id": case_id, "version": version, **judged})

    print("\n  agreement with rule-based truth:")
    for version in ("v1", "v2"):
        stats = agreement[version]
        print(f"    {version}: {stats['agree']}/{stats['total']}")

    if mlflow_reachable():
        import mlflow  # noqa: PLC0415 - only needed when a run actually happens

        from lib import setup_mlflow  # noqa: PLC0415

        _, experiment_id = setup_mlflow(DEFAULT_EXPERIMENT)
        with mlflow.start_run(run_name="l7-live-judge") as run:
            mlflow.log_params({"model": model, "cases": ",".join(JUDGED_CASE_IDS)})
            metrics = {
                f"judge_{row['version']}_{row['case_id'].replace('-', '_')}": row["score"]
                for row in rows
            }
            metrics.update(
                {
                    "v1_agreement": agreement["v1"]["agree"],
                    "v2_agreement": agreement["v2"]["agree"],
                    "judge_prompt_tokens": sum(r["prompt_tokens"] for r in rows),
                    "judge_completion_tokens": sum(r["completion_tokens"] for r in rows),
                }
            )
            mlflow.log_metrics(metrics)
            print(f"\n  mlflow: l7-live-judge run {run.info.run_id[:8]}... logged in {DEFAULT_EXPERIMENT}")
    else:
        print("\n  mlflow: server unreachable -> judge scores 未记录（服务器不可达）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
