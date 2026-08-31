"""
HIP Fix Planner Agent

Aggregates all evidence (triage / build warnings / runtime status /
kernel_parse suspicious patterns / probe non-determinism) and asks
Claude to produce:

    - root_cause: {hypothesis, mechanism, confidence, bug_type, affected_lines}
    - patch_plan: {summary, patched_kernel_source, diff, risks, ...}

SKILL.md is injected into the system prompt so Claude reasons with hygon
DCU domain knowledge (wave=64, LDS 32 banks, +1 tile padding, ...).

Heuristic fallback uses hip_repro_templates.py for the 4 canonical bug
classes when the LLM is unavailable or returns invalid output.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

from hip_agents import hip_repro_templates
from hip_orchestrator.state import HIPDebugState
from hip_tools.anthropic_client import AnthropicLLMClient, DCU_SKILL_PROMPT
from utils.logger import get_logger

logger = get_logger(__name__)


def _sf(v, d=0.0):
    if v is None:
        return d
    try:
        r = float(v)
        return d if r != r else max(0.0, min(1.0, r))
    except (TypeError, ValueError):
        return d


class HIPFixPlannerAgent:

    def __init__(self):
        self._llm = AnthropicLLMClient()

    def run(self, state: HIPDebugState) -> HIPDebugState:
        evidence = self._collect_evidence(state)

        # ── Try Anthropic first ─────────────────────────────────────────────
        if state.get("llm_available"):
            result = self._llm_plan(state, evidence)
            if result and result.get("root_cause") and result.get("patched_kernel_source"):
                state["root_cause"]     = self._sanitize_rc(result.get("root_cause"))
                state["patch_plan"]     = self._sanitize_pp(result.get("patch_plan") or {})
                state["patched_source"] = result["patched_kernel_source"]
                logger.info(
                    f"[fix] LLM plan ok: bug_type={state['root_cause'].get('bug_type')} "
                    f"conf={state['root_cause'].get('confidence')}"
                )
                return state
            logger.warning("[fix] LLM plan invalid or missing patched_kernel_source — falling back")

        # ── Heuristic fallback ──────────────────────────────────────────────
        bug_class = (state.get("triage") or {}).get("bug_class", "unknown")
        original = state.get("kernel_source", "")
        rc, pp, patched = hip_repro_templates.build_fix(bug_class, original)
        state["root_cause"]     = rc
        state["patch_plan"]     = pp
        state["patched_source"] = patched
        logger.info(f"[fix] heuristic plan for bug_class={bug_class}")
        return state

    def _collect_evidence(self, state: HIPDebugState) -> Dict[str, Any]:
        return {
            "bug_report":     state.get("bug_report", {}),
            "triage":         state.get("triage", {}),
            "build_result": {
                "status": state.get("build_result", {}).get("status"),
                "warnings": state.get("build_result", {}).get("warnings", []),
                "errors": state.get("build_result", {}).get("errors", []),
            },
            "runtime_result": {
                "exit_code": state.get("runtime_result", {}).get("exit_code"),
                "status":    state.get("runtime_result", {}).get("status"),
                "stdout_tail": (state.get("runtime_result", {}).get("stdout") or "")[-800:],
                "stderr_tail": (state.get("runtime_result", {}).get("stderr") or "")[-800:],
            },
            "numeric_result": state.get("numeric_result", {}),
            "kernel_parse":   {
                "kernels": [
                    {k: v for k, v in k_.items() if k != "body_snippet"}
                    for k_ in (state.get("kernel_parse", {}).get("kernels") or [])
                ],
                "suspicious_patterns": state.get("kernel_parse", {}).get("suspicious_patterns", []),
                "launches": state.get("kernel_parse", {}).get("launches", []),
            },
            "probe_evidence": state.get("probe_evidence", {}),
        }

    def _llm_plan(self, state: HIPDebugState, evidence: Dict) -> Optional[Dict]:
        src = state.get("kernel_source", "")
        prompt = f"""You are debugging a HIP kernel on a hygon DCU BW151. Below is:
  1. The full kernel source.
  2. Structured evidence from a multi-agent pipeline (build/runtime/static-analysis/probe).

Your task: identify the ROOT CAUSE and produce a MINIMAL FIX.

Return ONLY this JSON, no markdown:
{{
  "root_cause": {{
    "hypothesis": "one precise sentence",
    "mechanism": "how it produces the observed symptom, step by step",
    "confidence": 0.0-1.0,
    "bug_type": "out_of_bounds | reduction_race | bank_conflict | uncoalesced_access | launch_config_error | precision_error | atomic_contention | missing_sync | other",
    "affected_lines": [42, 43]
  }},
  "patch_plan": {{
    "summary": "one line",
    "diff": "unified-diff-style hunk showing the change (or a small before→after snippet)",
    "risks": ["..."],
    "expected_effect": "e.g. correct output on all elements, or 2x speedup",
    "validation_steps": ["recompile", "rerun", "check STATUS: PASS"]
  }},
  "patched_kernel_source": "<the FULL patched .hip file — every line, ready to save and compile>"
}}

Rules for `patched_kernel_source`:
  * It MUST be the entire file (do NOT emit just the changed lines).
  * Preserve the existing STATUS: / MAX_ABS_ERR: / GPU_RESULT: printf protocol used by main().
  * Do NOT add explanatory comments outside the changed region.
  * Do NOT rename functions.
  * The file must compile with `hipcc -O2 -std=c++17` on hygon DCU (BW151).

KERNEL SOURCE:
```cpp
{src}
```

EVIDENCE:
{json.dumps(evidence, indent=2, default=str)[:6000]}
"""
        return self._llm.generate(prompt, system_prompt=DCU_SKILL_PROMPT, max_tokens=8192)

    def _sanitize_rc(self, rc: Any) -> Dict:
        if not isinstance(rc, dict):
            return {}
        rc = dict(rc)
        rc["confidence"] = _sf(rc.get("confidence"), 0.5)
        for f in ("hypothesis", "mechanism", "bug_type"):
            if not isinstance(rc.get(f), str):
                rc[f] = ""
        if not isinstance(rc.get("affected_lines"), list):
            rc["affected_lines"] = []
        return rc

    def _sanitize_pp(self, pp: Any) -> Dict:
        if not isinstance(pp, dict):
            return {}
        pp = dict(pp)
        for f in ("summary", "diff", "expected_effect"):
            if not isinstance(pp.get(f), str):
                pp[f] = ""
        for lf in ("risks", "validation_steps"):
            if not isinstance(pp.get(lf), list):
                pp[lf] = []
        return pp
