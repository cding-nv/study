"""
HIP Triage Agent
Extract a bug class + hypotheses from bug_report + kernel source preview.
LLM (Anthropic) first; keyword-based heuristic fallback.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

from hip_orchestrator.state import HIPDebugState
from hip_tools.anthropic_client import AnthropicLLMClient
from utils.logger import get_logger

logger = get_logger(__name__)

BUG_CLASSES = [
    "out_of_bounds",
    "reduction_race",
    "bank_conflict",
    "uncoalesced_access",
    "launch_config_error",
    "precision_error",
    "atomic_contention",
    "missing_sync",
    "unknown",
]


class HIPTriageAgent:

    def __init__(self):
        self._llm = AnthropicLLMClient()

    def run(self, state: HIPDebugState) -> HIPDebugState:
        br = state.get("bug_report") or {}
        src = state.get("kernel_source") or ""

        if state.get("llm_available"):
            result = self._llm_triage(br, src)
            if isinstance(result, dict) and result:
                state["triage"] = self._sanitize(result)
                logger.info(f"[triage] LLM ok: bug_class={state['triage'].get('bug_class')}")
                return state

        state["triage"] = self._heuristic(br, src)
        logger.info(f"[triage] heuristic: bug_class={state['triage'].get('bug_class')}")
        return state

    def _llm_triage(self, br: Dict, src: str) -> Dict:
        prompt = f"""Analyze this HIP kernel bug on hygon DCU BW151.

Return ONLY a JSON object with this exact shape:
{{
  "title": "short title",
  "severity": "critical|high|medium|low",
  "bug_class": "one of: {' | '.join(BUG_CLASSES)}",
  "summary": "one paragraph explaining the likely defect",
  "hypotheses": [
    {{"id":"H1","description":"...","confidence":0.85,"reasoning":"..."}}
  ],
  "expected_symptoms": ["FAIL_NUMERIC" | "FAIL_RUNTIME" | "hipErrorIllegalAddress" | "slow" | ...]
}}

BUG_REPORT:
{json.dumps(br, indent=2)[:2000]}

KERNEL_SOURCE (first 3000 chars):
{src[:3000]}
"""
        return self._llm.generate(prompt) or {}

    def _sanitize(self, d: Dict) -> Dict:
        out = dict(d)
        # confidence normalization
        clean_h = []
        for h in d.get("hypotheses") or []:
            if not isinstance(h, dict):
                continue
            try:
                h["confidence"] = max(0.0, min(1.0, float(h.get("confidence", 0.5))))
            except (TypeError, ValueError):
                h["confidence"] = 0.5
            h.setdefault("id", f"H{len(clean_h) + 1}")
            h.setdefault("description", "")
            h.setdefault("reasoning", "")
            clean_h.append(h)
        out["hypotheses"] = clean_h
        if out.get("bug_class") not in BUG_CLASSES:
            out["bug_class"] = "unknown"
        return out

    def _heuristic(self, br: Dict, src: str) -> Dict:
        text = (
            (br.get("title", "") + " " + br.get("description", "") + " " +
             " ".join(br.get("reproduction_hints", [])))
        ).lower() + src.lower()
        hypotheses: List[Dict] = []
        bug_class = "unknown"

        checks = [
            ("out_of_bounds",       ["<=", "out of bound", "illegal address", "oob", "vmfault"]),
            ("reduction_race",      ["__syncthreads", "reduc", "race", "run spread", "shared memory race"]),
            ("bank_conflict",       ["bank conflict", "tile[", "lds", "shared memory tile"]),
            ("uncoalesced_access",  ["uncoalesced", "coalesc", "stride"]),
            ("launch_config_error", ["launch bounds", "block size", "wavefront"]),
            ("precision_error",     ["precision", "fp16", "half"]),
            ("missing_sync",        ["syncthreads", "barrier", "sync"]),
        ]
        for cls, kws in checks:
            hits = sum(1 for kw in kws if kw in text)
            if hits >= 1 and bug_class == "unknown":
                bug_class = cls
            if hits > 0:
                hypotheses.append({
                    "id": f"H{len(hypotheses) + 1}",
                    "description": f"Potential {cls.replace('_', ' ')} — {hits} keyword match(es)",
                    "confidence": min(0.85, 0.4 + 0.15 * hits),
                    "reasoning": f"Keyword-based match on: {[k for k in kws if k in text][:3]}",
                })
        if not hypotheses:
            hypotheses.append({
                "id": "H1",
                "description": "Unclassified HIP kernel bug",
                "confidence": 0.25,
                "reasoning": "No triage keywords matched",
            })

        return {
            "title": br.get("title", "Unknown HIP bug"),
            "severity": (br.get("severity") or "medium").lower(),
            "bug_class": bug_class,
            "summary": br.get("description", "")[:400],
            "hypotheses": hypotheses,
            "expected_symptoms": [],
            "_source": "heuristic",
        }
