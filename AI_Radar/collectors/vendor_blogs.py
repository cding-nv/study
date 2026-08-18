"""Vendor blogs via RSS: NVIDIA Developer, AMD ROCm, DeepMind."""
from __future__ import annotations
import feedparser
from datetime import datetime, timedelta, timezone
from dateutil import parser as dtp
from .base import BaseCollector
from ..utils.http import get
from ..utils.dedupe import dedupe_items

FEEDS = [
    ("nvidia_dev_blog",     "https://developer.nvidia.com/blog/feed"),
    ("amd_rocm_blog",       "https://rocm.blogs.amd.com/blog/atom.xml"),
    ("deepmind_blog",       "https://deepmind.google/blog/rss.xml"),
]

FALLBACK_HTML = {
    "amd_rocm_blog": "https://rocm.blogs.amd.com/",
    "deepmind_blog": "https://deepmind.google/discover/blog/",
    "nvidia_dev_blog": "https://developer.nvidia.com/blog/",
}


class VendorBlogsCollector(BaseCollector):
    name = "vendor_blogs"

    def collect(self) -> list[dict]:
        # Give vendors a wider window (7 days) — most post <daily
        since = datetime.now(timezone.utc) - timedelta(days=7)
        items: list[dict] = []
        for source_name, url in FEEDS:
            try:
                items.extend(self._one_feed(source_name, url, since))
            except Exception as e:  # noqa: BLE001
                items.append({
                    "source": source_name,
                    "title": f"[error fetching {source_name}]",
                    "url": FALLBACK_HTML.get(source_name, url),
                    "published": None,
                    "summary": f"{type(e).__name__}: {e}",
                    "meta": {"error": True},
                })
        return dedupe_items(items)

    def _one_feed(self, source_name: str, url: str, since: datetime) -> list[dict]:
        # Use our proxy-aware client, then hand bytes to feedparser
        r = get(self.client, url)
        parsed = feedparser.parse(r.content)
        out: list[dict] = []
        for e in parsed.entries[:30]:
            published_raw = e.get("published") or e.get("updated") or ""
            dt = None
            if published_raw:
                try:
                    dt = dtp.parse(published_raw)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    if dt < since:
                        continue
                except Exception:
                    pass
            summary = (e.get("summary") or e.get("description") or "")
            # strip HTML in summary
            if "<" in summary:
                from bs4 import BeautifulSoup
                summary = BeautifulSoup(summary, "lxml").get_text(" ", strip=True)
            summary = summary[:600]
            out.append({
                "source": source_name,
                "title": e.get("title", "(untitled)"),
                "url": e.get("link"),
                "published": dt.isoformat() if dt else published_raw or None,
                "summary": summary,
                "meta": {"author": e.get("author")},
            })
        return out
