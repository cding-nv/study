"""
Probe Agent

Reruns the buggy binary a few times to catch **non-determinism** — for HIP
race conditions the same binary produces different results on repeated
invocations. If runtime already crashed we skip probing (evidence is
sufficient). If numeric already failed with a single deterministic value we
still want the multi-run signal to strengthen the race hypothesis.

This is the HIP equivalent of the original ReproAgent, but instead of
generating a synthetic script, we exploit the fact that our buggy binaries
ARE self-contained repros — we just need to observe them enough times.
"""
from __future__ import annotations

from typing import Any, Dict, List

from hip_orchestrator.state import HIPDebugState
from hip_tools.docker_runner import run_in_container
from hip_tools.numeric_validator import parse_program_output
from utils.logger import get_logger

logger = get_logger(__name__)


class ProbeAgent:

    RUNS = 5

    def run(self, state: HIPDebugState) -> HIPDebugState:
        binary = state.get("binary_container_path", "")
        runtime = state.get("runtime_result", {}) or {}
        # Skip if runtime crashed hard — evidence already strong
        if runtime.get("status") in ("timeout",) or not binary:
            state["probe_evidence"] = {"skipped": True, "reason": "runtime crash or no binary"}
            return state

        samples: List[Dict[str, Any]] = []
        for i in range(self.RUNS):
            r = run_in_container(binary, timeout=60)
            n = parse_program_output(r.stdout + "\n" + r.stderr)
            samples.append({
                "run": i,
                "status": n.status,
                "gpu_sample": n.gpu_sample[:4],
                "max_abs_err": n.max_abs_err,
                "elapsed_ms": n.elapsed_ms,
                "hip_err": n.hip_err,
                "exit_code": r.exit_code,
            })

        # Detect non-determinism (race)
        statuses = {s["status"] for s in samples}
        first_gpu = tuple(samples[0].get("gpu_sample") or ())
        varied = sum(1 for s in samples if tuple(s.get("gpu_sample") or ()) != first_gpu)

        elapsed_vals = [s["elapsed_ms"] for s in samples if isinstance(s["elapsed_ms"], (int, float))]
        elapsed_stats = {}
        if elapsed_vals:
            elapsed_stats = {
                "min": round(min(elapsed_vals), 3),
                "max": round(max(elapsed_vals), 3),
                "mean": round(sum(elapsed_vals) / len(elapsed_vals), 3),
            }

        evidence = {
            "runs": self.RUNS,
            "samples": samples,
            "distinct_statuses": sorted(statuses),
            "run_to_run_variation": varied,
            "nondeterministic": varied > 0,
            "elapsed_ms_stats": elapsed_stats,
        }
        state["probe_evidence"] = evidence
        logger.info(
            f"[probe] {self.RUNS} runs → varied={varied}/{self.RUNS} "
            f"statuses={statuses} elapsed={elapsed_stats}"
        )
        return state
