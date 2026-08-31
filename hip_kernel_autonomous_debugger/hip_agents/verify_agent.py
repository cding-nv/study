"""
Verify Agent
Applies the patched kernel source, re-compiles, re-runs, and compares
numeric output & timing against the baseline.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from hip_orchestrator.state import HIPDebugState
from hip_tools.docker_runner import run_in_container, host_to_container, container_to_host
from hip_tools.hipcc_wrapper import compile_hip
from hip_tools.numeric_validator import parse_program_output, compare_speedup
from utils.logger import get_logger

logger = get_logger(__name__)


class VerifyAgent:

    def run(self, state: HIPDebugState) -> HIPDebugState:
        patched = state.get("patched_source", "") or ""
        if not patched:
            state["verify_result"] = {"ran": False, "compiled": False, "reason": "no patched_source"}
            return state

        # ── 1. Write patched source to disk (host side, then compile in container) ──
        kernel_container = state["kernel_file_container"]
        kernel_host      = state["kernel_file_host"]
        out_dir_host      = state["output_dir_host"]
        out_dir_container = state["output_dir_container"]

        name = Path(kernel_container).stem
        patched_host      = os.path.join(out_dir_host,      f"{name}_patched.hip")
        patched_container = os.path.join(out_dir_container, f"{name}_patched.hip")
        Path(patched_host).write_text(patched, encoding="utf-8")

        # ── 2. Compile patched version ──────────────────────────────────────
        binary_container = os.path.join(out_dir_container, f"{name}_patched")
        br = compile_hip(patched_container, binary_container, timeout=90)
        if br.status != "ok":
            state["verify_result"] = {
                "ran": False,
                "compiled": False,
                "compile_errors": [e.__dict__ for e in br.errors],
                "compile_stderr_tail": br.stderr[-800:],
                "patched_source_file": patched_host,
            }
            logger.warning(f"[verify] patched compile failed: {len(br.errors)} errors")
            return state

        # ── 3. Run patched binary ────────────────────────────────────────────
        r = run_in_container(binary_container, timeout=120)
        combined = r.stdout + "\n" + r.stderr
        numeric = parse_program_output(combined)

        # Compare vs baseline
        baseline_numeric = state.get("numeric_result") or {}
        baseline_ms = baseline_numeric.get("elapsed_ms")
        patched_ms  = numeric.elapsed_ms
        speedup_info = compare_speedup(baseline_ms, patched_ms) if (baseline_ms and patched_ms) else {}

        state["verify_result"] = {
            "ran": True,
            "compiled": True,
            "exit_code": r.exit_code,
            "status": numeric.status,
            "hip_err": numeric.hip_err,
            "numerically_correct": bool(numeric.correct),
            "max_abs_err": numeric.max_abs_err,
            "gpu_sample": numeric.gpu_sample[:8],
            "cpu_sample": numeric.cpu_sample[:8],
            "patched_elapsed_ms": patched_ms,
            "baseline_elapsed_ms": baseline_ms,
            "speedup": speedup_info.get("speedup"),
            "patched_source_file": patched_host,
            "patched_binary_container": binary_container,
            "stdout_tail": r.stdout[-600:],
            "stderr_tail": r.stderr[-600:],
        }
        logger.info(
            f"[verify] compiled=True ran=True status={numeric.status} "
            f"correct={numeric.correct} speedup={speedup_info.get('speedup')}"
        )
        return state
