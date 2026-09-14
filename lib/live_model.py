"""Live-model scaffolding shared by all lessons.

The curriculum separates two kinds of experiments:

- **offline mechanism experiments** -- deterministic, driven by scripted fake
  models defined *inside each lesson's* ``main.py``. They never touch this
  module and never need network access.
- **live model experiments** -- real calls through an OpenAI-compatible
  endpoint (OpenAI itself, or Qwen/vLLM/any compatible gateway). Configure
  them via a ``.env`` file in the repo root:

      OPENAI_API_KEY=sk-...
      OPENAI_BASE_URL=https://api.openai.com/v1   # or a Qwen-compatible URL
      LIVE_MODEL_NAME=gpt-4o-mini                  # or e.g. qwen3-max

Conventions (mirrored from DESIGN.md):

1. A live experiment without configuration **skips** -- a skip is recorded in
   the lesson README as "未验证", never counted as passed.
2. From L2 on, a live run should also leave an MLflow trace + cost metrics.
   The cheap recipe: ``setup_mlflow()`` + ``enable_graph_tracing()`` before
   the run (server reachable), ``mlflow.start_run()`` around it, and read
   traces back with ``flush=True`` -- see ``lib/mlflow_utils.py``.
3. Fakes are pedagogical props per lesson and stay in the lesson folder;
   this module only carries the *real* endpoint plumbing.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# Lessons are run from anywhere; pin the .env to the repo root next to pyproject.
_REPO_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(_REPO_ROOT / ".env")

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_LIVE_MODEL = "gpt-4o-mini"


def live_enabled() -> bool:
    """True if a real endpoint is configured.

    Used by ``live_demo.py`` scripts to bail out early and by tests to skip
    cleanly -- the offline equivalent of ``mlflow_reachable``.
    """
    return bool(os.environ.get("OPENAI_API_KEY"))


def live_base_url() -> str:
    return os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL)


def live_model_name() -> str:
    return os.environ.get("LIVE_MODEL_NAME", DEFAULT_LIVE_MODEL)


def live_client() -> OpenAI:
    """A ready OpenAI SDK client aimed at the configured endpoint."""
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=live_base_url())


def describe_live_config() -> str:
    """One-line summary for demo banners; the key itself is never printed."""
    if not live_enabled():
        return "live model: NOT configured (set OPENAI_API_KEY in .env) -> skip / 未验证"
    key = os.environ["OPENAI_API_KEY"]
    shown = key[:6] + "..." if len(key) > 6 else "***"
    return f"live model: {live_model_name()} @ {live_base_url()} (key {shown})"
