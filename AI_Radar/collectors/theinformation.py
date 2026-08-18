"""The Information (paywalled + Cloudflare-protected).

Direct HTML returns Cloudflare "Attention Required" 403.
We surface headlines via Google News RSS filtered to site:theinformation.com.
"""
from __future__ import annotations
from .base import BaseCollector
from . import _google_news
from ..utils.dedupe import dedupe_items


class TheInformationCollector(BaseCollector):
    name = "theinformation"

    def collect(self) -> list[dict]:
        # TheInformation is lower-volume; give it a wider window
        raw = _google_news.fetch(
            self.client,
            query="site:theinformation.com (AI OR OpenAI OR Anthropic OR Nvidia OR agent OR chip OR datacenter)",
            lookback_hours=96,
            max_items=25,
        )
        items = []
        for it in raw:
            items.append({
                **it,
                "source": "theinformation",
                "summary": "🔒 Paywalled — headline only.",
            })
        if not items:
            items.append({
                "source": "theinformation",
                "title": "[The Information — no fresh matches via Google News]",
                "url": "https://www.theinformation.com/features/artificial-intelligence",
                "published": None,
                "summary": "Direct site is Cloudflare-gated; Google News returned 0 recent AI results.",
                "meta": {"paywall": True, "empty": True},
            })
        return dedupe_items(items)
