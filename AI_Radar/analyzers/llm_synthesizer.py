"""LLM-driven synthesis: TL;DR + narrative + Deep Dive suggestions.

Reads env: ANTHROPIC_AUTH_TOKEN / ANTHROPIC_API_KEY, ANTHROPIC_BASE_URL,
ANTHROPIC_MODEL. Falls back to a rules-based summary if disabled or fails.
"""
from __future__ import annotations
import json
import logging
from typing import Any

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是一位 AI 行业观察分析师。用户每天会给你 9 个渠道当天收集到的原始条目 (论文、GitHub 活动、厂商博客、行业新闻等)。

你的任务:输出一段结构化的中文简报,帮助用户 (1) 5 分钟看完当天的 AI 行业动向,(2) 挑出值得当天深入研究的 1-3 个方向。

严格按以下 JSON 结构返回,不要任何 markdown 代码块围栏,不要多余的解释:
{
  "tldr": ["<第 1 条今日要事>", "<第 2 条>", "<第 3-5 条>"],
  "narrative": {
    "research": "<研究前沿的一段叙事,提炼跨源信号,例如“MoE + 长上下文 + Agentic RL 的组合在本轮同时出现”>",
    "inference_stack": "<vLLM/SGLang/TRT-LLM 等推理栈的方向变化>",
    "vendor": "<NVIDIA/AMD/DeepMind 等厂商的信号>",
    "industry": "<Reuters/TheInformation/SemiAnalysis 的产业与资金动向>",
    "usage": "<OpenRouter 折射出的真实使用趋势>"
  },
  "deep_dive": [
    {"topic": "<方向名>", "why": "<为什么值得今天深入>", "starting_points": ["<url1>", "<url2>"], "search_prompt": "<用户可以复制到搜索引擎/AI 里的探索性 prompt>"}
  ]
}

要求:
- tldr 每条不超过 40 汉字,写清楚 "谁 / 做了什么 / 意味着什么"。
- narrative 每段 3-5 句,重点突出跨源交叉信号,不要罗列条目。
- deep_dive 数量按用户 config (默认 3)。starting_points 用给到的条目里已有的真实 url,不要编造。
- 用户是资深工程师,不要科普基础概念,直接讲判断和差异。"""


def _try_import_sdk():
    try:
        import anthropic  # type: ignore
        return anthropic
    except ImportError:
        return None


def _build_user_content(payloads: dict[str, dict], keyword_trend: dict, deep_dive_count: int) -> str:
    """Compact the raw items into a token-friendly digest for the LLM."""
    lines: list[str] = [f"# 今日采集摘要  (需要 deep_dive 数量: {deep_dive_count})", ""]

    # keyword trend first — it's compact and high-signal
    lines.append("## 关键词跨源热度 (top 15)")
    for kw, hits, src_cnt in keyword_trend.get("top_keywords", []):
        lines.append(f"- {kw}: {hits} 次,{src_cnt} 个源")
    lines.append("")

    # per source
    for source_name, pl in payloads.items():
        if not pl.get("ok"):
            lines.append(f"## [{source_name}]  ⚠ FAILED: {pl.get('error')}")
            continue
        items = pl.get("items", [])
        if not items:
            lines.append(f"## [{source_name}]  (0 items)")
            continue
        lines.append(f"## [{source_name}]  ({len(items)} items, showing up to 25)")
        for it in items[:25]:
            title = (it.get("title") or "").replace("\n", " ")[:180]
            url = it.get("url") or ""
            summary = (it.get("summary") or "").replace("\n", " ")[:220]
            lines.append(f"- {title}\n  {url}\n  {summary}")
        lines.append("")

    return "\n".join(lines)


def _rules_fallback(payloads: dict[str, dict], keyword_trend: dict, deep_dive_count: int) -> dict:
    """Deterministic fallback if LLM is disabled or fails."""
    # TL;DR: pick top item from each of the most active sources
    tldr = []
    for src in ["huggingface", "github", "vendor_blogs", "arxiv", "reuters_tech"]:
        pl = payloads.get(src)
        if pl and pl.get("ok") and pl.get("items"):
            it = pl["items"][0]
            tldr.append(f"[{src}] {(it.get('title') or '')[:80]}")
        if len(tldr) >= 5:
            break

    top_kw = ", ".join(f"{kw}({h})" for kw, h, _ in keyword_trend.get("top_keywords", [])[:5])
    narrative = {
        "research": f"关键词跨源热度前列: {top_kw or '无'}。请查看 §1/§5 明细。",
        "inference_stack": "见 §2 详细列表 (GitHub 活动)。",
        "vendor": "见 §3 厂商博客明细。",
        "industry": "见 §4 产业渠道。",
        "usage": "见 §5 OpenRouter 明细。",
    }

    # Deep Dive: pick top-2 keywords with most source diversity
    deep_dive = []
    for kw, hits, src_cnt in keyword_trend.get("top_keywords", [])[:deep_dive_count]:
        related_items = keyword_trend.get("keyword_items", {}).get(kw, [])
        urls = [it.get("url") for it in related_items[:3] if it.get("url")]
        deep_dive.append({
            "topic": kw,
            "why": f"在 {src_cnt} 个源中出现,共 {hits} 次,跨源信号较强。",
            "starting_points": urls,
            "search_prompt": f"从推理系统 / 论文 / 产业博客三个角度总结 {kw} 最近 2 周的进展与开源代码。",
        })

    return {
        "tldr": tldr or ["(规则版) 今日无高置信要事,请翻阅明细"],
        "narrative": narrative,
        "deep_dive": deep_dive,
        "_generated_by": "rules_fallback",
    }


def synthesize(payloads: dict[str, dict], keyword_trend: dict, cfg: dict) -> dict:
    llm_cfg = cfg.get("llm", {}) or {}
    if not llm_cfg.get("enabled", True):
        return _rules_fallback(payloads, keyword_trend, cfg.get("deep_dive_count", 3))

    env = cfg.get("_env", {}) or {}
    token = env.get("anthropic_token")
    if not token:
        log.warning("ANTHROPIC_AUTH_TOKEN not set — using rules fallback")
        return _rules_fallback(payloads, keyword_trend, cfg.get("deep_dive_count", 3))

    sdk = _try_import_sdk()
    if sdk is None:
        log.warning("anthropic sdk not installed — using rules fallback")
        return _rules_fallback(payloads, keyword_trend, cfg.get("deep_dive_count", 3))

    client_kwargs: dict[str, Any] = {"api_key": token}
    if env.get("anthropic_base_url"):
        client_kwargs["base_url"] = env["anthropic_base_url"]
    client = sdk.Anthropic(**client_kwargs)

    model = llm_cfg.get("model", "claude-sonnet-4-6")
    max_tokens = llm_cfg.get("max_tokens", 4096)
    deep_dive_count = cfg.get("deep_dive_count", 3)

    user_content = _build_user_content(payloads, keyword_trend, deep_dive_count)

    try:
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )
    except Exception as e:  # noqa: BLE001
        log.warning("LLM call failed (%s) — %s", type(e).__name__, e)
        if llm_cfg.get("fallback_to_rules", True):
            out = _rules_fallback(payloads, keyword_trend, deep_dive_count)
            out["_llm_error"] = f"{type(e).__name__}: {e}"
            return out
        raise

    text = "".join(getattr(b, "text", "") for b in resp.content).strip()
    stop_reason = getattr(resp, "stop_reason", None)
    usage = getattr(resp, "usage", None)
    log.info("LLM returned %d chars, stop_reason=%s, usage=%s", len(text), stop_reason, usage)
    parsed_obj = _extract_json(text)
    if parsed_obj is not None:
        parsed_obj.setdefault("_generated_by", f"llm:{model}")
        return parsed_obj

    log.warning("LLM returned unparseable text (first 200 chars): %s", text[:200])
    out = _rules_fallback(payloads, keyword_trend, deep_dive_count)
    out["_llm_raw"] = text[:2000]
    out["_llm_error"] = "json parse failed"
    return out


def _extract_json(text: str) -> dict | None:
    """Try several strategies to pull a JSON object out of the LLM response."""
    import re
    if not text:
        return None
    candidates = []
    # 1. as-is
    candidates.append(text.strip())
    # 2. strip ```json ... ``` or ``` ... ``` fences
    fence = re.match(r"^\s*```(?:json)?\s*\n?(.*?)\n?```\s*$", text, re.DOTALL)
    if fence:
        candidates.append(fence.group(1).strip())
    # 3. first { ... last } substring
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last > first:
        candidates.append(text[first:last + 1])
    for c in candidates:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None
