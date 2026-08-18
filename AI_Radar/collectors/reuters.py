"""Reuters technology news.

Reuters killed their public RSS. Direct HTML is behind a 401/challenge.
We fetch via Google News RSS filtered to site:reuters.com + AI/chip/GPU
keywords — public, stable, and already filtered to what we care about.
"""
from __future__ import annotations
from .base import BaseCollector
from . import _google_news
from ..utils.dedupe import dedupe_items


class ReutersCollector(BaseCollector):
    name = "reuters_tech"

    def collect(self) -> list[dict]:
        # Reuters posts ~dozens of AI-tagged pieces a day; 48h keeps it fresh
        raw = _google_news.fetch(
            self.client,
            query="site:reuters.com (AI OR chip OR GPU OR datacenter OR Nvidia OR OpenAI OR Anthropic OR TSMC)",
            lookback_hours=48,
            max_items=30,
        )
        items = [{**it, "source": "reuters_tech"} for it in raw]
        return dedupe_items(items)
