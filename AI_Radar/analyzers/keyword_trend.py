"""Keyword frequency + cross-source signal detection.

Given all collected items and a keyword taxonomy from config, compute:
  - per-category frequency
  - per-keyword source diversity (how many distinct sources mention it)
  - items containing the keyword (for drill-down)
"""
from __future__ import annotations
import re
from collections import defaultdict


def _mk_pattern(kw: str) -> re.Pattern:
    # word-boundary, case-insensitive, allow hyphens and spaces literally
    esc = re.escape(kw)
    # allow "MoE" to also match "MoEs"
    return re.compile(rf"(?<![A-Za-z0-9]){esc}(?![A-Za-z0-9])", re.IGNORECASE)


def analyze(items: list[dict], taxonomy: dict[str, list[str]]) -> dict:
    """Returns:
    {
      "categories": { "architecture": {"total_hits": N, "keywords": {"MoE": {"hits": n, "sources": [..]}}}},
      "top_keywords": [(kw, hits, source_count), ...],
      "keyword_items": {kw: [item, ...]},   # for drill-down (deduped)
    }
    """
    compiled = {kw: _mk_pattern(kw) for kws in taxonomy.values() for kw in kws}
    per_kw_hits: dict[str, int] = defaultdict(int)
    per_kw_sources: dict[str, set[str]] = defaultdict(set)
    per_kw_items: dict[str, list[dict]] = defaultdict(list)

    for it in items:
        text = " ".join([
            str(it.get("title") or ""),
            str(it.get("summary") or ""),
        ])
        if not text.strip():
            continue
        matched_here: set[str] = set()
        for kw, pat in compiled.items():
            if pat.search(text):
                matched_here.add(kw)
        for kw in matched_here:
            per_kw_hits[kw] += 1
            per_kw_sources[kw].add(it.get("source", "unknown"))
            per_kw_items[kw].append(it)

    categories: dict[str, dict] = {}
    for cat, kws in taxonomy.items():
        cat_total = 0
        kw_stats = {}
        for kw in kws:
            h = per_kw_hits.get(kw, 0)
            if h == 0:
                continue
            kw_stats[kw] = {"hits": h, "sources": sorted(per_kw_sources.get(kw, set()))}
            cat_total += h
        categories[cat] = {"total_hits": cat_total, "keywords": kw_stats}

    top = sorted(
        [(kw, per_kw_hits[kw], len(per_kw_sources[kw])) for kw in per_kw_hits],
        key=lambda x: (x[2], x[1]),   # source-diversity first, then raw hits
        reverse=True,
    )[:15]

    return {
        "categories": categories,
        "top_keywords": top,
        "keyword_items": {kw: its for kw, its in per_kw_items.items()},
    }
