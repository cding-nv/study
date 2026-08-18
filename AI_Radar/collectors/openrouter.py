"""OpenRouter model usage.

Approach:
1) Try https://openrouter.ai/api/frontend/stats/app  (public, used by their UI)
   -- not guaranteed stable; wrapped in try.
2) Fall back to scraping the rankings HTML at /rankings.
3) Also fetch the models catalog for pricing context.
"""
from __future__ import annotations
import json
import logging
from bs4 import BeautifulSoup
from .base import BaseCollector
from ..utils.http import get

log = logging.getLogger(__name__)


class OpenRouterCollector(BaseCollector):
    name = "openrouter"

    def collect(self) -> list[dict]:
        top_n = self.cfg.get("openrouter", {}).get("top_n", 20)
        items: list[dict] = []

        # 1. Public models catalog (stable)
        try:
            r = get(self.client, "https://openrouter.ai/api/v1/models")
            models = r.json().get("data", [])
            # Pick recently promoted / newest by created ts
            models_sorted = sorted(models, key=lambda m: m.get("created", 0), reverse=True)[:top_n]
            for m in models_sorted:
                pricing = m.get("pricing", {}) or {}
                items.append({
                    "source": "openrouter_models",
                    "title": m.get("name") or m.get("id"),
                    "url": f"https://openrouter.ai/{m.get('id', '')}",
                    "published": None,
                    "summary": (m.get("description") or "")[:400],
                    "meta": {
                        "id": m.get("id"),
                        "context_length": m.get("context_length"),
                        "pricing_prompt": pricing.get("prompt"),
                        "pricing_completion": pricing.get("completion"),
                        "created": m.get("created"),
                    },
                })
        except Exception as e:  # noqa: BLE001
            log.warning("openrouter models api failed: %s", e)

        # 2. Rankings page (usage trends)
        try:
            r = get(self.client, "https://openrouter.ai/rankings")
            soup = BeautifulSoup(r.text, "lxml")
            # Try to find embedded __NEXT_DATA__ for structured info
            nxt = soup.find("script", id="__NEXT_DATA__")
            if nxt and nxt.string:
                try:
                    data = json.loads(nxt.string)
                    # Depth-first look for a "rankings"/"models" list w/ tokens
                    def walk(o):
                        if isinstance(o, dict):
                            if "tokens" in o and ("name" in o or "model" in o or "slug" in o):
                                yield o
                            for v in o.values():
                                yield from walk(v)
                        elif isinstance(o, list):
                            for v in o:
                                yield from walk(v)

                    ranked = list(walk(data))[:top_n]
                    for r_ in ranked:
                        title = r_.get("name") or r_.get("model") or r_.get("slug") or "(unknown)"
                        items.append({
                            "source": "openrouter_rankings",
                            "title": str(title),
                            "url": "https://openrouter.ai/rankings",
                            "published": None,
                            "summary": f"tokens={r_.get('tokens')}",
                            "meta": {k: r_.get(k) for k in ("tokens", "requests", "share")},
                        })
                except Exception as e:  # noqa: BLE001
                    log.debug("openrouter rankings parse failed: %s", e)
        except Exception as e:  # noqa: BLE001
            log.warning("openrouter rankings fetch failed: %s", e)

        return items
