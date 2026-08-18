"""HuggingFace: trending papers + trending models.

Papers page (https://huggingface.co/papers/trending) ships a JSON payload
in a <script id="__NUXT__"> / Nuxt block. Simplest robust approach:
- try the public paper listing HTML
- parse links matching /papers/<arxiv_id>
- for each, fetch the paper page for title + upvotes + abstract snippet
Fall back to arxiv id extraction only if the detail fetch fails.
"""
from __future__ import annotations
import re
from bs4 import BeautifulSoup
from .base import BaseCollector
from ..utils.http import get
from ..utils.dedupe import dedupe_items

PAPER_LINK = re.compile(r"^/papers/(\d{4}\.\d{4,6})(?:/|$)")
MODEL_LINK = re.compile(r"^/([^/?#]+)/([^/?#]+)$")

# Owners/paths under huggingface.co that are NOT models
_MODEL_BLOCKLIST_OWNERS = {
    "inference", "settings", "spaces", "datasets", "papers", "docs", "blog",
    "join", "login", "pricing", "enterprise", "posts", "collections", "chat",
    "organizations", "new", "notifications", "hub", "learn", "api",
}
_MODEL_BLOCKLIST_PATHS = {
    "models", "hardware", "billing", "tokens", "keys", "profile",
}


class HuggingFaceCollector(BaseCollector):
    name = "huggingface"

    def extra_headers(self):
        return {"Accept": "text/html,application/xhtml+xml"}

    def collect(self) -> list[dict]:
        hf_cfg = self.cfg.get("huggingface", {})
        papers = self._papers(top_n=hf_cfg.get("papers_top_n", 20))
        models = self._models(top_n=hf_cfg.get("models_top_n", 15))
        return dedupe_items(papers + models)

    # ---- Trending Papers ----
    def _papers(self, top_n: int) -> list[dict]:
        r = get(self.client, "https://huggingface.co/papers/trending")
        soup = BeautifulSoup(r.text, "lxml")
        # First pass: build arxiv_id -> title from h3-anchored links, which HF uses for paper titles
        id_to_title: dict[str, str] = {}
        for h in soup.find_all(["h3", "h2", "h4"]):
            a = h.find("a", href=True)
            if not a:
                continue
            m = PAPER_LINK.match(a["href"])
            if not m:
                continue
            t = a.get_text(" ", strip=True) or h.get_text(" ", strip=True)
            if t and len(t) >= 5:
                id_to_title.setdefault(m.group(1), t)

        seen: set[str] = set()
        items: list[dict] = []
        for a in soup.find_all("a", href=True):
            m = PAPER_LINK.match(a["href"])
            if not m:
                continue
            arxiv_id = m.group(1)
            if arxiv_id in seen:
                continue
            seen.add(arxiv_id)
            title = id_to_title.get(arxiv_id) or a.get_text(" ", strip=True) or f"arXiv:{arxiv_id}"
            items.append({
                "source": "huggingface_papers",
                "title": title,
                "url": f"https://huggingface.co/papers/{arxiv_id}",
                "published": None,
                "summary": "",
                "meta": {"arxiv_id": arxiv_id, "arxiv_url": f"https://arxiv.org/abs/{arxiv_id}"},
            })
            if len(items) >= top_n:
                break
        return items

    # ---- Trending Models ----
    def _models(self, top_n: int) -> list[dict]:
        # sort=trending is the trending tab
        r = get(self.client, "https://huggingface.co/models?sort=trending")
        soup = BeautifulSoup(r.text, "lxml")
        items: list[dict] = []
        seen: set[str] = set()
        for a in soup.find_all("a", href=True):
            href = a["href"]
            m = MODEL_LINK.match(href)
            if not m:
                continue
            owner, name = m.group(1), m.group(2)
            if owner in _MODEL_BLOCKLIST_OWNERS:
                continue
            if name in _MODEL_BLOCKLIST_PATHS:
                continue
            # HF model ids are lowercase-flexible, but nav links to language pages etc.
            # look like /zh/models — skip 2-char owners that are language codes
            if len(owner) <= 2:
                continue
            model_id = f"{owner}/{name}"
            if model_id in seen:
                continue
            seen.add(model_id)
            text = a.get_text(" ", strip=True)
            items.append({
                "source": "huggingface_models",
                "title": model_id,
                "url": f"https://huggingface.co/{model_id}",
                "published": None,
                "summary": text if text and text != model_id else "",
                "meta": {"model_id": model_id},
            })
            if len(items) >= top_n:
                break
        return items
