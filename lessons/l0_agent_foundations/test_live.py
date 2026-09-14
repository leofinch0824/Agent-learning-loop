"""Live tests for L0 - one real single-shot call.

These skip cleanly when no endpoint is configured (no .env / OPENAI_API_KEY).
A skip is recorded as 未验证 in the lesson README, never counted as passed
(same convention as L1's test_mlflow.py, but for the model endpoint).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lib import live_enabled  # noqa: E402

pytestmark = pytest.mark.skipif(not live_enabled(), reason="live model not configured (no .env)")


def test_live_single_shot_returns_content_and_usage():
    """One real call on the L0 task returns non-empty content AND a usage block."""
    from lessons.l0_agent_foundations.live_demo import run_live_single_shot

    content, usage = run_live_single_shot()
    assert isinstance(content, str) and content.strip(), "response content must be non-empty"
    assert usage is not None, "endpoint returned no usage block"
    for field_name in ("prompt_tokens", "completion_tokens", "total_tokens"):
        assert getattr(usage, field_name) is not None, f"usage.{field_name} missing"
