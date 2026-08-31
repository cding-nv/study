"""
Anthropic Client for HIP Debugger

Interface mirrors utils.llm_client.LLMClient:
  is_available()
  generate(prompt, system_prompt="")      → Optional[Dict]     (JSON-enforced)
  generate_code(prompt, system_prompt="") → Optional[str]      (raw text)

Configuration via env: ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN /
                        ANTHROPIC_BASE_URL / ANTHROPIC_MODEL

Also exposes DCU_SKILL_PROMPT: contents of hipc_kernel_opt_skills/SKILL.md,
which fix / reviewer agents append to their system prompt.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from utils.logger import get_logger

logger = get_logger(__name__)


ANTHROPIC_MODEL     = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")
ANTHROPIC_MAX_TOKS  = int(os.environ.get("ANTHROPIC_MAX_TOKENS", "4096"))
ANTHROPIC_TIMEOUT_S = int(os.environ.get("ANTHROPIC_TIMEOUT", "120"))
ANTHROPIC_TEMP      = float(os.environ.get("ANTHROPIC_TEMPERATURE", "0.1"))


def _load_dcu_skill() -> str:
    """Load the hipc_kernel_opt_skills SKILL.md as a domain-knowledge prompt."""
    candidates = [
        Path(os.environ.get("HIP_SKILL_PATH", "")) if os.environ.get("HIP_SKILL_PATH") else None,
        # Container view: /public/home/dingf is mounted at /workspace, so the host path
        # /public/home/dingf/workspace/.claude/... becomes /workspace/workspace/.claude/...
        Path("/workspace/workspace/.claude/skills/hipc_kernel_opt_skills/SKILL.md"),
        Path("/workspace/.claude/skills/hipc_kernel_opt_skills/SKILL.md"),
        Path("/public/home/dingf/workspace/.claude/skills/hipc_kernel_opt_skills/SKILL.md"),
    ]
    for c in candidates:
        if c and c.is_file():
            try:
                return c.read_text(encoding="utf-8")
            except Exception as e:
                logger.warning(f"[anthropic_client] Failed to read {c}: {e}")
    logger.warning("[anthropic_client] SKILL.md not found — fix/reviewer will lack DCU domain knowledge")
    return ""


DCU_SKILL_PROMPT = _load_dcu_skill()


class AnthropicLLMClient:
    def __init__(self, model: str = ANTHROPIC_MODEL, timeout: int = ANTHROPIC_TIMEOUT_S):
        self.model = model
        self.timeout = timeout
        self._client = None
        self._available: Optional[bool] = None

    def _lazy_init(self):
        if self._client is not None:
            return
        try:
            from anthropic import Anthropic
            self._client = Anthropic(timeout=self.timeout)
        except Exception as e:
            logger.warning(f"[anthropic_client] SDK init failed: {e}")
            self._available = False

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            logger.warning("[anthropic_client] No API key or auth token set")
            self._available = False
            return False
        self._lazy_init()
        self._available = self._client is not None
        return self._available

    # ─────────────────────────────────────────────────────────────────────────
    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = ANTHROPIC_MAX_TOKS,
        retries: int = 2,
    ) -> Optional[Dict[str, Any]]:
        """Enforce a JSON dict response. Returns None on complete failure."""
        if not self.is_available():
            return None

        full_system = (
            "You are a senior HIP / hygon DCU kernel debugger. "
            "You MUST respond with a SINGLE valid JSON object only. "
            "No markdown fences, no prose outside the JSON. "
            "If uncertain about a field, set it to null. "
            "Do not invent file paths or line numbers not present in the input."
        )
        if system_prompt:
            full_system += "\n\n" + system_prompt

        for attempt in range(1, retries + 2):
            try:
                resp = self._client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=full_system,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = "".join(
                    b.text for b in resp.content if getattr(b, "type", "") == "text"
                )
                parsed = _safe_json_parse(raw)
                if parsed is not None:
                    return parsed
                logger.warning(
                    f"[anthropic_client] attempt {attempt}: non-JSON response ({len(raw)} chars)"
                )
            except Exception as e:
                logger.warning(f"[anthropic_client] attempt {attempt} error: {e}")
                if attempt <= retries:
                    time.sleep(2 ** attempt)

        return None

    def generate_code(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = ANTHROPIC_MAX_TOKS,
    ) -> Optional[str]:
        """Return free-form text (no JSON enforcement)."""
        if not self.is_available():
            return None

        full_system = (
            "You are an expert HIP / hygon DCU kernel author. "
            "Output ONLY the requested code with no markdown fences and no commentary."
        )
        if system_prompt:
            full_system += "\n\n" + system_prompt

        try:
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=full_system,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
            return _strip_code_fences(text) if text else None
        except Exception as e:
            logger.warning(f"[anthropic_client] generate_code error: {e}")
            return None


# ─── helpers ────────────────────────────────────────────────────────────────

def _strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        lines = t.split("\n")
        # drop first fence line
        lines = lines[1:]
        # drop trailing fence
        while lines and lines[-1].strip().startswith("```"):
            lines.pop()
        t = "\n".join(lines)
    return t.strip()


def _safe_json_parse(text: str) -> Optional[Dict]:
    if not text or not text.strip():
        return None
    t = _strip_code_fences(text)
    try:
        r = json.loads(t)
        if isinstance(r, dict):
            return r
    except json.JSONDecodeError:
        pass
    # Fallback: pull first {...} block
    s = t.find("{")
    e = t.rfind("}")
    if s != -1 and e > s:
        try:
            r = json.loads(t[s:e + 1])
            if isinstance(r, dict):
                return r
        except json.JSONDecodeError:
            pass
    return None
