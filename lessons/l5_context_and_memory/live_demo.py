"""L5 live run: a REAL summarizer under the offline contract, and a prefix-cache probe.

Two small live experiments (no MLflow in this lesson -- progressive design,
observation via lib helpers only):

1. Real mid-history summarization: the SAME dropped-message list demo 4
   collapses offline is summarized by the real model under the same contract
   as ``fake_summarizer`` (system instruction + serialized dropped messages).
   The observation of interest: does the binding "外层 -> 0.1 mm" survive,
   or only the bare number like the fake?
2. Prefix-cache probe: two calls sharing the identical stable prefix, then
   print whatever usage fields the endpoint returns. Per DESIGN §7 a shared
   prefix is only a NECESSARY condition; a cache HIT can be claimed solely
   from provider telemetry. If no cached-token field is present, we print
   that and claim nothing.

Prerequisites (repo root): ``.env`` with OPENAI_API_KEY (+ optional
OPENAI_BASE_URL, LIVE_MODEL_NAME).

Run:

    poetry run python lessons/l5_context_and_memory/live_demo.py

Without .env this prints the configuration summary and exits 0.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lib import banner, describe_live_config, live_client, live_enabled, live_model_name  # noqa: E402

from lessons.l5_context_and_memory.main import (  # noqa: E402
    STABLE_PREFIX,
    build_long_history,
    collapse_with_summary,
    messages_tokens,
    trim_messages,
)

# Same instruction SHAPE as the offline contract in main.fake_summarizer:
# a system prompt + the serialized dropped messages -> one summary string.
SUMMARIZER_SYSTEM_PROMPT = (
    "你在为一次 PCB 工艺参数对话做中段摘要。保留全部数值事实及其绑定"
    "（哪个参数、哪个数值、哪个单位），不要引入新事实。用中文，不超过 3 句话。"
)

PREFIX_PROBE_SUFFIXES = [
    "3 mm 是多少 mil？",
    "0.127 mm 是多少 mil？",
]


def live_summarize(client, model: str, dropped: list[dict]) -> tuple[str, dict]:
    """Real-model drop-in for fake_summarizer; returns (summary, usage_dict)."""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SUMMARIZER_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(dropped, ensure_ascii=False)},
        ],
        temperature=0,
    )
    usage = resp.usage.model_dump() if getattr(resp, "usage", None) else {}
    return resp.choices[0].message.content or "", usage


def cached_token_fields(usage: dict) -> dict:
    """Harvest whatever cache-telemetry fields this provider's usage carries.

    Known shapes: OpenAI ``prompt_tokens_details.cached_tokens``; Anthropic-
    style ``cache_read_input_tokens`` / ``cache_creation_input_tokens``;
    other gateways use ``cached_input_tokens`` or ``prompt_cache_hit_tokens``.
    Absent fields mean NO evidence -- never a negative claim about caching.
    """
    found: dict = {}
    if not isinstance(usage, dict):
        return found
    details = usage.get("prompt_tokens_details") or {}
    if isinstance(details, dict) and details.get("cached_tokens") is not None:
        found["prompt_tokens_details.cached_tokens"] = details["cached_tokens"]
    for key in ("cache_read_input_tokens", "cached_input_tokens", "prompt_cache_hit_tokens"):
        if usage.get(key) is not None:
            found[key] = usage[key]
    return found


def _chat(client, model: str, suffix: str) -> dict:
    """One prefix-probe call; returns the raw usage dict."""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": STABLE_PREFIX},
            {"role": "user", "content": suffix},
        ],
        temperature=0,
    )
    return resp.usage.model_dump() if getattr(resp, "usage", None) else {}


def main() -> int:
    banner("L5 live - real summarizer + prefix-cache telemetry probe")
    print(f"  {describe_live_config()}")
    if not live_enabled():
        print("  skip: configure .env (OPENAI_API_KEY ...) -> live 未验证。")
        print("  offline 机制实验不受影响: poetry run python lessons/l5_context_and_memory/main.py")
        return 0

    client = live_client()
    model = live_model_name()

    # -- 1. real mid-history summarization under the offline contract --------
    banner("LIVE 1 - 真实模型做中段摘要（与 DEMO 4 同一裁剪结果）")
    history = build_long_history()
    budget = messages_tokens([history[0], history[-2], history[-1]]) + 5
    kept, dropped = trim_messages(history, budget)
    print(f"  dropped {len(dropped)} message(s) (~{messages_tokens(dropped)} tok by chars/4 estimate)")
    summary, usage = live_summarize(client, model, dropped)
    print(f"  usage: {usage}")
    print(f"  summary: {summary}")
    # Observation only (not an assertion): does the binding survive?
    print(f"  观察: 数字 0.1 存活: {'0.1' in summary} | 绑定 '外层' 存活: {'外层' in summary}")
    offline = collapse_with_summary(history, budget, lambda d: summary)
    print(f"  折叠后上下文 (~{messages_tokens(offline)} tok by estimate), 首条摘要消息:")
    print(f"    {[m['content'] for m in offline if m['content'].startswith('[summary')][0][:200]}")

    # -- 2. prefix-cache telemetry probe --------------------------------------
    banner("LIVE 2 - 前缀缓存遥测: 两次同前缀调用, 只读 usage, 不做无据宣称")
    for i, suffix in enumerate(PREFIX_PROBE_SUFFIXES, 1):
        usage_i = _chat(client, model, suffix)
        cache_fields = cached_token_fields(usage_i)
        print(f"  call {i} suffix={suffix!r}")
        print(f"    usage       : {usage_i}")
        print(f"    cache fields: {cache_fields if cache_fields else '(absent -> no hit claim) 未实测'}")
    print("  DESIGN §7: 前缀相同只是必要条件; 只有 cached_tokens 之类的遥测字段才能证明命中。")
    print("  字段缺失时本 demo 不宣称任何缓存收益 -- 记录为 未验证（无 provider 遥测字段）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
