"""arXiv listing via the official Atom API."""
from __future__ import annotations
import feedparser
from datetime import datetime, timedelta, timezone
from dateutil import parser as dtp
from .base import BaseCollector
from ..utils.http import get

ARXIV_QUERY = "http://export.arxiv.org/api/query"


class ArxivCollector(BaseCollector):
    name = "arxiv"

    def collect(self) -> list[dict]:
        cfg = self.cfg.get("arxiv", {})
        cats = cfg.get("categories", ["cs.AI"])
        n = cfg.get("max_results_per_cat", 30)
        since = datetime.now(timezone.utc) - timedelta(hours=self.cfg.get("lookback_hours", 36))
        out: list[dict] = []
        seen: set[str] = set()
        for cat in cats:
            for it in self._one_cat(cat, n, since):
                aid = it["meta"]["arxiv_id"]
                if aid in seen:
                    continue
                seen.add(aid)
                out.append(it)
        return out

    def _one_cat(self, cat: str, n: int, since: datetime) -> list[dict]:
        params = {
            "search_query": f"cat:{cat}",
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": n,
        }
        r = get(self.client, ARXIV_QUERY, params=params)
        parsed = feedparser.parse(r.content)
        out: list[dict] = []
        for e in parsed.entries:
            published = e.get("published", "")
            try:
                dt = dtp.parse(published) if published else None
                if dt and dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
            except Exception:
                dt = None
            if dt and dt < since:
                continue
            aid = e.get("id", "").rsplit("/", 1)[-1].split("v")[0]
            authors = ", ".join(a.get("name", "") for a in e.get("authors", [])[:5])
            summary = (e.get("summary") or "").strip().replace("\n", " ")[:600]
            out.append({
                "source": "arxiv",
                "title": e.get("title", "(untitled)").strip().replace("\n", " "),
                "url": e.get("link"),
                "published": dt.isoformat() if dt else published or None,
                "summary": summary,
                "meta": {"arxiv_id": aid, "category": cat, "authors": authors},
            })
        return out
