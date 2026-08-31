"""
HIP Reviewer Agent

Independent critique of the fix planner's proposal:
  - Are the changes minimal and localized?
  - Does the patched source still contain the STATUS/GPU_RESULT protocol
    lines so downstream verification stays functional?
  - Any obvious anti-patterns re-introduced?
  - Any contradictions between root cause and evidence?
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

from hip_orchestrator.state import HIPDebugState
from hip_tools.anthropic_client import AnthropicLLMClient, DCU_SKILL_PROMPT
from utils.logger import get_logger

logger = get_logger(__name__)


def _sl(v): return v if isinstance(v, list) else []
def _sd(v): return v if isinstance(v, dict) else {}


class HIPReviewerAgent:

    def __init__(self):
        self._llm = AnthropicLLMClient()

    def run(self, state: HIPDebugState) -> HIPDebugState:
        original = state.get("kernel_source", "") or ""
        patched  = state.get("patched_source", "") or ""
        rc       = _sd(state.get("root_cause"))
        pp       = _sd(state.get("patch_plan"))
        runtime  = _sd(state.get("runtime_result"))
        verify   = _sd(state.get("verify_result"))

        # ── Structural checks ────────────────────────────────────────────────
        issues: List[str] = []
        contradictions: List[str] = []

        if not patched or patched == original:
            issues.append("ERROR: patched_source is empty or identical to original")
        if patched and "STATUS:" not in patched:
            issues.append("WARN: patched source dropped the STATUS: printf protocol — verify may fail to classify")
        if patched and "GPU_RESULT" not in patched:
            issues.append("WARN: patched source dropped GPU_RESULT: printf — verify cannot compare samples")
        if len(patched) > 3 * len(original) if original else False:
            issues.append("WARN: patched source is >3× larger than original — not a minimal fix")

        # Contradiction between claimed fix and verify outcome
        if verify:
            if not verify.get("compiled"):
                contradictions.append("Fix did not compile — root cause claim not verifiable")
            elif not verify.get("numerically_correct") and verify.get("ran"):
                contradictions.append(
                    f"Fix compiled and ran but STATUS={verify.get('status')}, correct={verify.get('numerically_correct')}"
                    " — the proposed patch does not resolve the bug"
                )

        # LLM independent review (optional)
        llm_review = None
        if state.get("llm_available"):
            llm_review = self._llm_review(rc, pp, runtime, verify)

        state["review"] = {
            "issues": issues,
            "contradictions": contradictions,
            "llm_review": llm_review,
            "recommendation": self._recommendation(issues, contradictions, verify),
        }
        logger.info(
            f"[reviewer] issues={len(issues)} contradictions={len(contradictions)} "
            f"rec={state['review']['recommendation']}"
        )
        return state

    def _llm_review(self, rc: Dict, pp: Dict, runtime: Dict, verify: Dict) -> Dict:
        prompt = f"""You are a senior HIP kernel engineer independently reviewing a proposed
kernel patch. Be critical — challenge the diagnosis and the proposed fix.

Return JSON only:
{{
  "strengths": ["..."],
  "weaknesses": ["..."],
  "missed_edge_cases": ["..."],
  "safety_concerns": ["..."]
}}

Root cause: {json.dumps(rc, indent=2)[:800]}
Patch plan: {json.dumps(pp, indent=2)[:800]}
Runtime status: {runtime.get('status')} exit={runtime.get('exit_code')}
Verify: {json.dumps(verify, indent=2)[:400]}
"""
        return self._llm.generate(prompt, system_prompt=DCU_SKILL_PROMPT, max_tokens=1024) or {}

    @staticmethod
    def _recommendation(issues, contradictions, verify) -> str:
        if any(i.startswith("ERROR") for i in issues):
            return "DO NOT SHIP — critical structural issue in patch"
        if contradictions:
            return "REVISIT — contradictions between diagnosis and verify outcome"
        if verify and verify.get("compiled") and verify.get("numerically_correct"):
            spd = verify.get("speedup")
            if isinstance(spd, (int, float)) and spd > 1.2:
                return f"SHIP — patch verified correct, {spd:.2f}× speedup"
            return "SHIP — patch verified correct"
        return "MANUAL REVIEW — no ship signal"
