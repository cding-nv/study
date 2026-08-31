"""
HIP debugger LangGraph pipeline.

Node flow:
  preprocess → triage → build ─┬─ build_ok  ─→ runtime → kernel_analyst → probe → fix → verify → reviewer → finalize
                               └─ build_err ────────────────────────→ fix → verify → reviewer → finalize

Every node is wrapped in _safe_node (mirrors the pattern from the original
Python pipeline) so an agent crash never kills the run.
"""
from __future__ import annotations

import time
import traceback
from typing import Any, Dict

from langgraph.graph import StateGraph, END

from hip_orchestrator.state import HIPDebugState
from hip_agents.triage_agent   import HIPTriageAgent
from hip_agents.build_agent    import BuildAgent
from hip_agents.runtime_agent  import RuntimeAgent
from hip_agents.kernel_analyst import KernelAnalystAgent
from hip_agents.probe_agent    import ProbeAgent
from hip_agents.fix_agent      import HIPFixPlannerAgent
from hip_agents.reviewer_agent import HIPReviewerAgent
from hip_agents.verify_agent   import VerifyAgent
from utils.logger import get_logger

logger = get_logger(__name__)


def _safe_node(name: str, callable_fn):
    def node(state: HIPDebugState) -> HIPDebugState:
        start = time.time()
        entry = {"agent": name, "status": "running"}
        try:
            logger.info(f"[hip-orchestrator] ▶ {name}")
            result = callable_fn(state)
            dur = time.time() - start
            entry.update({"status": "success", "duration_s": round(dur, 3)})
            logger.info(f"[hip-orchestrator] ✔ {name} ({dur:.2f}s)")
            result["agent_trace"] = list(state.get("agent_trace", [])) + [entry]
            return result
        except Exception as e:
            dur = time.time() - start
            msg = f"{type(e).__name__}: {e}"
            tb = traceback.format_exc()
            logger.error(f"[hip-orchestrator] ✘ {name} failed: {msg}\n{tb}")
            entry.update({"status": "error", "duration_s": round(dur, 3), "error": msg})
            return {
                **state,
                "errors": list(state.get("errors", [])) + [f"{name}: {msg}"],
                "agent_trace": list(state.get("agent_trace", [])) + [entry],
            }
    node.__name__ = name
    return node


# ── Instances shared across nodes ────────────────────────────────────────────
_triage   = HIPTriageAgent()
_build    = BuildAgent()
_runtime  = RuntimeAgent()
_analyst  = KernelAnalystAgent()
_probe    = ProbeAgent()
_fix      = HIPFixPlannerAgent()
_review   = HIPReviewerAgent()
_verify   = VerifyAgent()


def _preprocess(state: HIPDebugState) -> HIPDebugState:
    from hip_tools.anthropic_client import AnthropicLLMClient
    state["llm_available"] = AnthropicLLMClient().is_available()
    if not state["llm_available"]:
        state.setdefault("errors", []).append(
            "WARN: Anthropic API unavailable — heuristic-only mode"
        )
    if not state.get("kernel_source"):
        state.setdefault("errors", []).append(
            "ERROR: kernel_source is empty"
        )
    return state


def _finalize(state: HIPDebugState) -> HIPDebugState:
    """Compute a final confidence score from the accumulated evidence."""
    score = 0.0
    build     = state.get("build_result") or {}
    runtime   = state.get("runtime_result") or {}
    numeric   = state.get("numeric_result") or {}
    parse     = state.get("kernel_parse") or {}
    root      = state.get("root_cause") or {}
    review    = state.get("review") or {}
    verify    = state.get("verify_result") or {}

    # Bug is confirmed via runtime failure or numeric fail — evidence quality
    if runtime.get("status") in ("error", "crash", "timeout"):
        score += 0.30
    if numeric.get("status") in ("FAIL_NUMERIC", "FAIL_RUNTIME", "FAIL_LAUNCH"):
        score += 0.20
    if build.get("warning_count", 0) > 0:
        score += 0.05
    # Static analysis catches something
    if parse.get("suspicious_patterns"):
        score += 0.10
    # Root cause quality
    rc_conf = float(root.get("confidence") or 0.0)
    if root.get("hypothesis"):
        score += 0.10 + rc_conf * 0.05
    # Verify closes the loop
    v_ran = verify.get("ran", False)
    v_correct = verify.get("numerically_correct", False)
    if v_ran and v_correct:
        score += 0.30
    elif v_ran and not v_correct:
        score -= 0.15
    # Speedup evidence (performance bugs)
    if isinstance(verify.get("speedup"), (int, float)) and verify["speedup"] > 1.2:
        score += 0.05
    # Contradictions penalty
    score -= 0.10 * len(review.get("contradictions") or [])
    # Cap
    state["confidence"] = round(max(0.0, min(1.0, score)), 3)
    logger.info(
        f"[hip-orchestrator] final confidence={state['confidence']} "
        f"verify_correct={v_correct} runtime={runtime.get('status')}"
    )
    return state


# ── Conditional edges ──────────────────────────────────────────────────────
def _route_after_build(state: HIPDebugState) -> str:
    if state.get("build_result", {}).get("status") == "ok":
        return "runtime"
    return "fix"    # skip runtime; go straight to fix


def _route_after_runtime(state: HIPDebugState) -> str:
    # Whether or not runtime succeeded, we still want the analyst then probe → fix
    return "kernel_analyst"


def build_graph():
    g = StateGraph(HIPDebugState)
    g.add_node("preprocess",     _safe_node("preprocess",     _preprocess))
    g.add_node("triage",         _safe_node("triage",         _triage.run))
    g.add_node("build",          _safe_node("build",          _build.run))
    g.add_node("runtime",        _safe_node("runtime",        _runtime.run))
    g.add_node("kernel_analyst", _safe_node("kernel_analyst", _analyst.run))
    g.add_node("probe",          _safe_node("probe",          _probe.run))
    g.add_node("fix",            _safe_node("fix",            _fix.run))
    g.add_node("verify",         _safe_node("verify",         _verify.run))
    g.add_node("reviewer",       _safe_node("reviewer",       _review.run))
    g.add_node("finalize",       _safe_node("finalize",       _finalize))

    g.set_entry_point("preprocess")
    g.add_edge("preprocess", "triage")
    g.add_edge("triage",     "build")
    g.add_conditional_edges("build", _route_after_build,
                            {"runtime": "runtime", "fix": "fix"})
    g.add_edge("runtime", "kernel_analyst")
    g.add_edge("kernel_analyst", "probe")
    g.add_edge("probe",   "fix")
    g.add_edge("fix",     "verify")
    g.add_edge("verify",  "reviewer")
    g.add_edge("reviewer","finalize")
    g.add_edge("finalize", END)
    return g.compile()


def run_pipeline(
    bug_report: Dict,
    kernel_source: str,
    kernel_file_host: str,
    kernel_file_container: str,
    output_dir_host: str,
    output_dir_container: str,
) -> HIPDebugState:
    from hip_orchestrator.state import initial_state
    logger.info("=" * 60)
    logger.info(f"[hip-orchestrator] Starting pipeline for {bug_report.get('id','UNKNOWN')}")
    logger.info("=" * 60)
    state = initial_state(
        bug_report, kernel_source, kernel_file_host, kernel_file_container,
        output_dir_host, output_dir_container,
    )
    graph = build_graph()
    try:
        final = graph.invoke(state)
        logger.info("[hip-orchestrator] Pipeline completed")
        return final
    except Exception as e:
        logger.error(f"[hip-orchestrator] Pipeline crashed: {e}\n{traceback.format_exc()}")
        return {**state, "errors": state.get("errors", []) + [f"pipeline crash: {e}"], "confidence": 0.0}
