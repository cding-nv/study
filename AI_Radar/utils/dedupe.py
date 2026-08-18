"""URL / title dedupe helpers."""
from __future__ import annotations
import re
from urllib.parse import urlparse, urlunparse


def canonical_url(url: str) -> str:
    """Strip query params, fragments, trailing slash, and www."""
    if not url:
        return url
    p = urlparse(url.strip().lower())
    netloc = p.netloc.removeprefix("www.")
    path = p.path.rstrip("/")
    return urlunparse((p.scheme or "https", netloc, path, "", "", ""))


_ws = re.compile(r"\s+")


def norm_title(t: str) -> str:
    return _ws.sub(" ", (t or "").strip().lower())


def dedupe_items(items: list[dict], key: str = "url", title_key: str = "title") -> list[dict]:
    """Dedupe by canonical URL, fall back to normalized title."""
    seen_urls: set[str] = set()
    seen_titles: set[str] = set()
    out: list[dict] = []
    for it in items:
        u = canonical_url(it.get(key, "") or "")
        t = norm_title(it.get(title_key, "") or "")
        if u and u in seen_urls:
            continue
        if t and t in seen_titles:
            continue
        if u:
            seen_urls.add(u)
        if t:
            seen_titles.add(t)
        out.append(it)
    return out
