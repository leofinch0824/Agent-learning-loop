"""L0 live check: ONE real single-shot call on the very same task.

L0 has no MLflow by design (observability is layered in progressively from
L1), so the only live evidence this lesson collects is the raw request /
response / usage triple of the regime every other regime is measured against.

If no endpoint is configured (.env absent, no OPENAI_API_KEY), the script
prints the skip notice from ``lib.describe_live_config()`` and exits 0 -- a
skip, never a failure, and recorded as 未验证 in the lesson README.

Run it (repo root):

    poetry run python lessons/l0_agent_foundations/live_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lib import banner, describe_live_config, live_client, live_enabled, live_model_name  # noqa: E402

from lessons.l0_agent_foundations.main import (  # noqa: E402
    FREE_TEXT_TASK,
    build_single_shot_prompt,
)


def run_live_single_shot() -> tuple[str, Any]:
    """Make exactly ONE real chat completion call; return (content, usage).

    The prompt is byte-for-byte the one the offline fake answered in DEMO 1,
    so whatever comes back can be compared against SINGLE_SHOT_SCRIPT's
    scripted unit-conversion slip.
    """
    client = live_client()
    response = client.chat.completions.create(
        model=live_model_name(),
        messages=[{"role": "user", "content": build_single_shot_prompt(FREE_TEXT_TASK)}],
        temperature=0,  # deterministic-ish; still a real model, so no offline assertion
    )
    return response.choices[0].message.content or "", response.usage


def main() -> int:
    banner("L0 LIVE - one real single-shot call on the same task")
    print(f"  {describe_live_config()}", flush=True)
    if not live_enabled():
        print("  -> live run skipped; recorded as 未验证（未配置 .env） in README 完成记录.", flush=True)
        print("     offline demos still work: poetry run python lessons/l0_agent_foundations/main.py", flush=True)
        return 0

    print(f"  model    : {live_model_name()}", flush=True)
    print(f"  task     : {FREE_TEXT_TASK}", flush=True)
    content, usage = run_live_single_shot()

    print("  response :", flush=True)
    for line in content.splitlines():
        print(f"    {line}", flush=True)

    # DESIGN.md evidence rule for live runs: provider/model + response + usage.
    # usage can be None on some OpenAI-compatible gateways -- print, do not crash.
    if usage is None:
        print("  usage    : None (endpoint returned no usage block)", flush=True)
    else:
        print(
            f"  usage    : prompt_tokens={usage.prompt_tokens} "
            f"completion_tokens={usage.completion_tokens} total_tokens={usage.total_tokens}",
            flush=True,
        )
    print(
        "\n  compare with DEMO 1: this is the single_shot regime -- 1 model call,\n"
        "  5 latency units in the simulated model. Check the copper verdict yourself:\n"
        "  did the real model keep 1 oz in oz, or slip into mm like the scripted one?",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
