"""Live tests for L7: the REAL model as judge v1 vs v2.

Skipped entirely when .env is not configured (live_enabled() False); a
skip is recorded in README 完成记录 as 未验证, never counted as passed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lib import live_client, live_enabled, live_model_name  # noqa: E402
from lessons.l7_observability_and_evaluation.live_demo import parse_judge_reply  # noqa: E402
from lessons.l7_observability_and_evaluation.main import (  # noqa: E402
    CASES_BY_ID,
    JUDGE_PASS_THRESHOLD,
    run_case,
)

# Probed once at import time: without .env every test below skips.
_LIVE = live_enabled()
pytestmark = pytest.mark.skipif(not _LIVE, reason="live model not configured (.env: OPENAI_API_KEY)")


def _judge(version: str, case_id: str) -> dict:
    from lessons.l7_observability_and_evaluation.live_demo import make_live_judge  # noqa: PLC0415

    record = run_case(CASES_BY_ID[case_id])
    judge = make_live_judge(live_client(), live_model_name(), version)
    return judge(CASES_BY_ID[case_id], record.final_answer)


def test_live_judge_output_parses():
    """The real judge's reply parses into the judge interface (verdict/score/reason)."""
    judged = _judge("v1", "l7-c01")
    assert judged["verdict"] in {"pass", "fail"}
    assert 0.0 <= judged["score"] <= 1.0
    assert isinstance(judged["reason"], str)


def test_parse_judge_reply_tolerates_fences_and_prose():
    """Parsing survives markdown fences, prose, and broken JSON (offline check)."""
    fenced = 'Sure!\n```json\n{"verdict": "fail", "score": 0.2, "reason": "no evidence"}\n```'
    assert parse_judge_reply(fenced)["verdict"] == "fail"
    assert parse_judge_reply(fenced)["score"] == 0.2
    garbage = "I think the answer fails the grounding check."
    assert parse_judge_reply(garbage)["verdict"] == "fail"


def test_live_judge_v2_catches_the_hack_case():
    """The grounded live judge fails the fluent-but-wrong answer (v2 catch)."""
    judged = _judge("v2", "l7-c07")
    assert judged["verdict"] == "fail"
    assert judged["score"] < JUDGE_PASS_THRESHOLD
