"""Live tests for L5: real summarization quality and usage telemetry.

Skipped entirely when .env is not configured (live_enabled() False). A skip
is recorded in README 完成记录 as 未验证, never counted as passed. These are
deliberately LIGHT: non-determinism of a real model is reported, not
asserted away.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lib import live_client, live_enabled, live_model_name  # noqa: E402

from lessons.l5_context_and_memory.live_demo import (  # noqa: E402
    PREFIX_PROBE_SUFFIXES,
    _chat,
    cached_token_fields,
    live_summarize,
)
from lessons.l5_context_and_memory.main import (  # noqa: E402
    build_long_history,
    messages_tokens,
    trim_messages,
)

_LIVE = live_enabled()

pytestmark = pytest.mark.skipif(not _LIVE, reason="live model not configured (.env: OPENAI_API_KEY)")


def _dropped_messages() -> list[dict]:
    history = build_long_history()
    budget = messages_tokens([history[0], history[-2], history[-1]]) + 5
    _, dropped = trim_messages(history, budget)
    return dropped


def test_live_summary_is_non_empty_and_reports_usage():
    """The real model returns a non-empty summary of the dropped middle + usage."""
    dropped = _dropped_messages()
    summary, usage = live_summarize(live_client(), live_model_name(), dropped)
    assert isinstance(summary, str) and summary.strip(), "real summarizer returned nothing"
    assert usage.get("prompt_tokens", 0) > 0 and usage.get("completion_tokens", 0) > 0
    # Observation only: whether the number->parameter binding survives is the
    # quality question demo'd in live_demo.py, not something to assert on.
    print(f"    summary: {summary[:200]}")
    print(f"    binding survived (观察): 0.1={'0.1' in summary}, 外层={'外层' in summary}")


def test_live_usage_fields_and_cache_telemetry_reported():
    """A prefix-probe call returns usage with token counts; cache fields are printed, not asserted."""
    usage = _chat(live_client(), live_model_name(), PREFIX_PROBE_SUFFIXES[0])
    assert usage.get("prompt_tokens", 0) >= len(PREFIX_PROBE_SUFFIXES), "usage must count the prompt"
    assert usage.get("completion_tokens", 0) > 0
    # If the provider exposes cache telemetry we print it; absence is a valid
    # observation (-> 未实测), never a failure: prefix equality alone proves
    # nothing about hits (DESIGN §7).
    print(f"    usage       : {usage}")
    print(f"    cache fields: {cached_token_fields(usage) or '(absent)'}")
