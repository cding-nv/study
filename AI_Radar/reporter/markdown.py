"""Assemble the final Markdown daily report."""
from __future__ import annotations
from datetime import date
from pathlib import Path
from .. import config as _cfg


def _fmt_item(it: dict, show_source: bool = False) -> str:
    title = (it.get("title") or "(untitled)").replace("|", "\\|")
    url = it.get("url") or ""
    src = f" _(*{it['source']}*)_" if show_source and it.get("source") else ""
    line = f"- [{title}]({url}){src}" if url else f"- {title}{src}"
    summary = (it.get("summary") or "").strip().replace("\n", " ")
    if summary:
        line += f"\n  - {summary[:300]}"
    return line


def _fmt_empty(payload: dict | None) -> str | None:
    """Return a placeholder line if the source has no data or wasn't run; else None."""
    if payload is None:
        return "_(本次未运行此源)_\n"
    if not payload.get("ok"):
        return f"_采集失败: {payload.get('error')}_\n"
    return None


def _section_hf(payload: dict) -> str:
    placeholder = _fmt_empty(payload)
    if placeholder:
        return placeholder
    items = payload["items"]
    papers = [i for i in items if i["source"] == "huggingface_papers"]
    models = [i for i in items if i["source"] == "huggingface_models"]
    parts = []
    parts.append(f"### 1.1 HuggingFace Trending Papers ({len(papers)})")
    for p in papers[:15]:
        aid = p.get("meta", {}).get("arxiv_id", "")
        parts.append(f"- [{p['title']}]({p['url']}) — `arXiv:{aid}`")
    parts.append("")
    parts.append(f"### 1.2 HuggingFace Trending Models ({len(models)})")
    for m in models[:12]:
        parts.append(f"- [{m['title']}]({m['url']})")
    return "\n".join(parts) + "\n"


def _section_arxiv(payload: dict) -> str:
    placeholder = _fmt_empty(payload)
    if placeholder:
        return placeholder
    items = payload["items"]
    parts = [f"### 1.3 arXiv 最新提交 ({len(items)})"]
    for it in items[:20]:
        parts.append(_fmt_item(it))
        m = it.get("meta") or {}
        if m.get("authors"):
            parts.append(f"  - 作者: {m['authors']}  |  分类: {m.get('category', '')}")
    return "\n".join(parts) + "\n"


def _section_github(payload: dict) -> str:
    placeholder = _fmt_empty(payload)
    if placeholder:
        return placeholder
    items = payload["items"]
    by_repo: dict[str, list[dict]] = {}
    for it in items:
        repo = it.get("meta", {}).get("repo", "unknown")
        by_repo.setdefault(repo, []).append(it)
    parts = []
    for i, (repo, its) in enumerate(sorted(by_repo.items()), 1):
        releases = [x for x in its if x["source"] == "github_release"]
        prs = [x for x in its if x["source"] == "github_pr"]
        issues = [x for x in its if x["source"] == "github_issue"]
        parts.append(f"### 2.{i} {repo}  — {len(releases)} releases · {len(prs)} PRs · {len(issues)} issues")
        if releases:
            parts.append("**Releases**")
            for r in releases[:5]:
                parts.append(_fmt_item(r))
        if prs:
            parts.append("**Merged PRs**")
            for p in prs[:10]:
                parts.append(_fmt_item(p))
        if issues:
            parts.append("**Notable Issues**")
            for iss in issues[:5]:
                r = iss.get("meta", {}).get("reactions", 0)
                parts.append(_fmt_item(iss) + f"  · reactions: {r}")
        parts.append("")
    return "\n".join(parts)


def _section_vendor(payload: dict) -> str:
    placeholder = _fmt_empty(payload)
    if placeholder:
        return placeholder
    items = payload["items"]
    by_src: dict[str, list[dict]] = {}
    for it in items:
        by_src.setdefault(it["source"], []).append(it)
    label = {
        "nvidia_dev_blog": "NVIDIA Developer Blog",
        "amd_rocm_blog": "AMD ROCm Blog",
        "deepmind_blog": "DeepMind Blog",
    }
    parts = []
    for i, src in enumerate(["nvidia_dev_blog", "amd_rocm_blog", "deepmind_blog"], 1):
        its = by_src.get(src, [])
        parts.append(f"### 3.{i} {label[src]} ({len(its)})")
        if not its:
            parts.append("- _(无新内容 / 拉取失败)_")
        for it in its[:8]:
            parts.append(_fmt_item(it))
        parts.append("")
    return "\n".join(parts)


def _section_industry(reuters: dict, info: dict, semi: dict) -> str:
    parts = []
    for i, (name, payload) in enumerate([
        ("Reuters Technology", reuters),
        ("The Information (AI)", info),
        ("SemiAnalysis", semi),
    ], 1):
        parts.append(f"### 4.{i} {name}")
        placeholder = _fmt_empty(payload)
        if placeholder:
            parts.append(placeholder)
            continue
        items = payload["items"]
        if not items:
            parts.append("- _(无新内容)_")
        for it in items[:12]:
            parts.append(_fmt_item(it))
        parts.append("")
    return "\n".join(parts)


def _section_openrouter(payload: dict) -> str:
    placeholder = _fmt_empty(payload)
    if placeholder:
        return placeholder
    items = payload["items"]
    models = [i for i in items if i["source"] == "openrouter_models"]
    rank = [i for i in items if i["source"] == "openrouter_rankings"]
    parts = [f"### 5.1 最新 / 最近上架模型 ({len(models)})"]
    for m in models[:12]:
        meta = m.get("meta") or {}
        pp = meta.get("pricing_prompt") or "-"
        pc = meta.get("pricing_completion") or "-"
        ctx = meta.get("context_length") or "-"
        parts.append(f"- [{m['title']}]({m['url']})  |  ctx {ctx}  |  in ${pp} / out ${pc}")
    parts.append("")
    parts.append(f"### 5.2 Rankings (usage) — {len(rank)} 条")
    if not rank:
        parts.append("- _(未能解析出结构化排名,可手动查看 https://openrouter.ai/rankings)_")
    for r in rank[:15]:
        parts.append(_fmt_item(r))
    return "\n".join(parts) + "\n"


def _section_trend(kw: dict) -> str:
    if not kw:
        return "_no keyword data_\n"
    parts = ["### 关键词跨源热度 (按源多样度排序)", ""]
    parts.append("| 关键词 | 命中 | 源数量 |")
    parts.append("|---|---|---|")
    for k, hits, srcn in kw.get("top_keywords", [])[:15]:
        parts.append(f"| **{k}** | {hits} | {srcn} |")
    parts.append("")
    parts.append("### 分类别命中")
    for cat, info in kw.get("categories", {}).items():
        if info["total_hits"] == 0:
            continue
        kws = ", ".join(f"{k} ({v['hits']})" for k, v in info["keywords"].items())
        parts.append(f"- **{cat}** — 总命中 {info['total_hits']} — {kws}")
    return "\n".join(parts) + "\n"


def _section_synthesis(syn: dict) -> str:
    parts = []
    parts.append("## ⚡ TL;DR (今日要事)\n")
    for i, line in enumerate(syn.get("tldr", []), 1):
        parts.append(f"{i}. {line}")
    parts.append("")
    n = syn.get("narrative", {}) or {}
    if n:
        parts.append("## 📝 跨源叙事")
        for key, label in [
            ("research", "研究前沿"),
            ("inference_stack", "推理栈"),
            ("vendor", "厂商信号"),
            ("industry", "产业与资金"),
            ("usage", "使用趋势"),
        ]:
            if n.get(key):
                parts.append(f"**{label}** — {n[key]}\n")
    gen = syn.get("_generated_by", "")
    if gen:
        parts.append(f"_综合生成: {gen}_")
    if syn.get("_llm_error"):
        parts.append(f"_(⚠ LLM 出错回落到规则版: {syn['_llm_error']})_")
    if syn.get("_llm_raw"):
        parts.append("<details><summary>LLM 原始返回 (前 2000 字符)</summary>\n")
        parts.append("```\n" + syn["_llm_raw"] + "\n```")
        parts.append("</details>")
    parts.append("")
    return "\n".join(parts)


def _section_deep_dive(syn: dict) -> str:
    dd = syn.get("deep_dive", [])
    if not dd:
        return "_(无 deep dive 建议)_\n"
    parts = []
    for i, d in enumerate(dd, 1):
        parts.append(f"### §6.{i} {d.get('topic', '?')}")
        parts.append(f"**为什么**: {d.get('why', '')}\n")
        sp = d.get("starting_points") or []
        if sp:
            parts.append("**入口链接**")
            for u in sp:
                parts.append(f"- {u}")
            parts.append("")
        if d.get("search_prompt"):
            parts.append("**可复制的探索 prompt**")
            parts.append(f"> {d['search_prompt']}")
            parts.append("")
    return "\n".join(parts)


def build_report(day: date, payloads: dict[str, dict], kw_trend: dict, synthesis: dict) -> str:
    header = f"# AI Radar · {day.isoformat()}\n"
    counts = ", ".join(f"{k}={v['count']}" for k, v in payloads.items())
    failed = [k for k, v in payloads.items() if not v.get("ok")]
    header += f"\n> 采集统计: {counts}\n"
    if failed:
        header += f"> ⚠ 失败源: {', '.join(failed)}\n"
    header += "\n---\n"

    body_parts = [
        _section_synthesis(synthesis),
        "---\n## 🔬 §1 Research Frontier\n",
        _section_hf(payloads.get("huggingface", {})),
        _section_arxiv(payloads.get("arxiv", {})),
        "\n---\n## 🏗️ §2 Inference Stack (GitHub)\n",
        _section_github(payloads.get("github", {})),
        "\n---\n## 🏢 §3 Vendor Blogs\n",
        _section_vendor(payloads.get("vendor_blogs", {})),
        "\n---\n## 💰 §4 Industry Intel\n",
        _section_industry(
            payloads.get("reuters_tech", {}),
            payloads.get("theinformation", {}),
            payloads.get("semianalysis", {}),
        ),
        "\n---\n## 📊 §5 OpenRouter Usage\n",
        _section_openrouter(payloads.get("openrouter", {})),
        "\n---\n## 🔎 §6 今日 Deep Dive 建议\n",
        _section_deep_dive(synthesis),
        "\n---\n## 🧠 附录 · 关键词跨源热度\n",
        _section_trend(kw_trend),
    ]
    return header + "\n".join(body_parts)


def write_report(day: date, content: str) -> Path:
    fp = _cfg.REPORTS_DIR / f"{day.isoformat()}.md"
    fp.write_text(content, encoding="utf-8")
    return fp
