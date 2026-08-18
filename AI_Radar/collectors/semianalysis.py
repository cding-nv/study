"""SemiAnalysis (Substack): free-tier RSS gives title + teaser reliably."""
from __future__ import annotations
import feedparser
from datetime import datetime, timedelta, timezone
from dateutil import parser as dtp
from bs4 import BeautifulSoup
from .base import BaseCollector
from ..utils.http import get

FEED = "https://semianalysis.com/feed"


class SemiAnalysisCollector(BaseCollector):
    name = "semianalysis"

    def collect(self) -> list[dict]:
        # SemiAnalysis posts weekly-ish; a 60-day window keeps some content in the daily digest.
        # (Their raw feed returns 10 items; we time-filter after parse.)
        since = datetime.now(timezone.utc) - timedelta(days=60)
        try:
            r = get(self.client, FEED)
        except Exception as e:  # noqa: BLE001
            return [{
                "source": "semianalysis",
                "title": "[SemiAnalysis — fetch failed]",
                "url": "https://semianalysis.com",
                "published": None,
                "summary": f"{type(e).__name__}: {e}",
                "meta": {"error": True},
            }]
        parsed = feedparser.parse(r.content)
        out: list[dict] = []
        for e in parsed.entries[:25]:
            published_raw = e.get("published") or e.get("updated") or ""
            dt = None
            try:
                if published_raw:
                    dt = dtp.parse(published_raw)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    if dt < since:
                        continue
            except Exception:
                pass
            summary = e.get("summary") or e.get("description") or ""
            if "<" in summary:
                summary = BeautifulSoup(summary, "lxml").get_text(" ", strip=True)
            out.append({
                "source": "semianalysis",
                "title": e.get("title", "(untitled)"),
                "url": e.get("link"),
                "published": dt.isoformat() if dt else published_raw or None,
                "summary": summary[:600],
                "meta": {"author": e.get("author"), "note": "Full article likely paywalled"},
            })
        return out
