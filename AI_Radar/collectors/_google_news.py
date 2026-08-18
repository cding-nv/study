"""Google News RSS helper.

For sites behind Cloudflare/paywall/anti-bot (TheInformation, Reuters),
Google News's public RSS endpoint gives us titles + real article URLs.

URL form:
  https://news.google.com/rss/search?q=<QUERY>&hl=en-US&gl=US&ceid=US:en
"""
from __future__ import annotations
import urllib.parse
import feedparser
from datetime import datetime, timedelta, timezone
from dateutil import parser as dtp
from ..utils.http import get


def fetch(client, query: str, lookback_hours: int = 48, max_items: int = 30) -> list[dict]:
    """Return a list of items {title, url, published, summary, meta}.

    The Google News <link> is a google.com redirector; we keep it as-is —
    clicking works, and article-body scraping is out of scope. The publisher
    is exposed via meta.source_name.
    """
    q = urllib.parse.quote(query)
    url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
    r = get(client, url)
    parsed = feedparser.parse(r.content)
    since = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    out: list[dict] = []
    for e in parsed.entries[:max_items * 2]:  # over-fetch, then time-filter
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
        title = e.get("title", "").strip()
        # Google News suffixes titles with "- Publisher"; extract it
        source_name = None
        if " - " in title:
            head, sep, tail = title.rpartition(" - ")
            if 2 <= len(tail) <= 40 and not tail.endswith("."):
                title, source_name = head, tail
        out.append({
            "title": title,
            "url": e.get("link"),
            "published": dt.isoformat() if dt else published_raw or None,
            "summary": "",
            "meta": {"source_name": source_name, "query": query},
        })
        if len(out) >= max_items:
            break
    return out
