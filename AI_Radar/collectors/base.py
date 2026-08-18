"""Base class for all collectors.

Collector output shape (list of dict) - each item should have:
  - source: str  (e.g., "huggingface_papers")
  - title: str
  - url: str
  - published: str | None   (ISO8601)
  - summary: str            (short, human-readable)
  - meta: dict              (source-specific extras)
"""
from __future__ import annotations
import logging
from abc import ABC, abstractmethod
from typing import Any
import httpx
from ..utils.http import make_client
from ..utils.cache import save as cache_save

log = logging.getLogger(__name__)


class BaseCollector(ABC):
    name: str = "base"

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.client: httpx.Client | None = None

    # --- lifecycle ---
    def __enter__(self):
        self.client = make_client(
            proxy=self.cfg.get("proxy"),
            timeout=self.cfg.get("request_timeout", 30),
            extra_headers=self.extra_headers(),
        )
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.client:
            self.client.close()

    def extra_headers(self) -> dict:
        return {}

    # --- interface ---
    @abstractmethod
    def collect(self) -> list[dict]:
        ...

    def run(self) -> dict:
        """Wraps collect() with error isolation + cache."""
        try:
            with self:
                items = self.collect()
            log.info("[%s] collected %d items", self.name, len(items))
            payload = {"ok": True, "count": len(items), "items": items, "error": None}
        except Exception as e:  # noqa: BLE001 - want to catch anything
            log.exception("[%s] failed", self.name)
            payload = {"ok": False, "count": 0, "items": [], "error": f"{type(e).__name__}: {e}"}
        cache_save(self.name, payload)
        return payload
