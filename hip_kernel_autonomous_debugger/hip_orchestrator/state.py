"""
Shared state for the HIP kernel debugging pipeline.
Each agent reads and writes this TypedDict.
"""
from typing import Any, Dict, List, Optional, TypedDict


class HIPDebugState(TypedDict, total=False):
    # ── Inputs ──────────────────────────────────────────────────────────────
    bug_report: Dict[str, Any]        # parsed bug_report.json
    kernel_source: str                # kernel .hip file contents (host-side)
    kernel_file_host: str             # host path to the .hip file
    kernel_file_container: str        # container-visible path
    output_dir_host: str
    output_dir_container: str

    # ── Runtime configuration ───────────────────────────────────────────────
    llm_available: bool

    # ── Triage output ───────────────────────────────────────────────────────
    triage: Dict[str, Any]            # {title, severity, hypotheses, bug_class, ...}

    # ── Build output ────────────────────────────────────────────────────────
    build_result: Dict[str, Any]      # BuildResult.to_dict()
    binary_container_path: str

    # ── Runtime output ──────────────────────────────────────────────────────
    runtime_result: Dict[str, Any]    # ExecResult.to_dict() + parsed NumericResult
    numeric_result: Dict[str, Any]    # NumericResult.to_dict()

    # ── Static analysis ─────────────────────────────────────────────────────
    kernel_parse: Dict[str, Any]      # kernel_parser.KernelParseResult.to_dict()

    # ── Probe / dynamic evidence ────────────────────────────────────────────
    probe_evidence: Dict[str, Any]    # optional — findings from instrumented rerun

    # ── Fix planner output ──────────────────────────────────────────────────
    root_cause: Dict[str, Any]        # {hypothesis, mechanism, confidence, ...}
    patch_plan: Dict[str, Any]        # {summary, patched_source, diff, risks, ...}
    patched_source: str               # full patched kernel source

    # ── Review output ───────────────────────────────────────────────────────
    review: Dict[str, Any]

    # ── Verification (re-compile + re-run patched version) ──────────────────
    verify_result: Dict[str, Any]     # {compiled, ran, correct, speedup?, ...}

    # ── Meta ────────────────────────────────────────────────────────────────
    confidence: float
    errors: List[str]
    agent_trace: List[Dict]


def initial_state(
    bug_report: Dict,
    kernel_source: str,
    kernel_file_host: str,
    kernel_file_container: str,
    output_dir_host: str,
    output_dir_container: str,
) -> HIPDebugState:
    return HIPDebugState(
        bug_report=bug_report,
        kernel_source=kernel_source,
        kernel_file_host=kernel_file_host,
        kernel_file_container=kernel_file_container,
        output_dir_host=output_dir_host,
        output_dir_container=output_dir_container,
        llm_available=False,
        triage={},
        build_result={},
        binary_container_path="",
        runtime_result={},
        numeric_result={},
        kernel_parse={},
        probe_evidence={},
        root_cause={},
        patch_plan={},
        patched_source="",
        review={},
        verify_result={},
        confidence=0.0,
        errors=[],
        agent_trace=[],
    )
