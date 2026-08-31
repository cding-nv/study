"""
Runtime Agent
Execute the compiled HIP binary inside the container and parse the
strict-format stdout into a NumericResult.
"""
from __future__ import annotations

from hip_orchestrator.state import HIPDebugState
from hip_tools.docker_runner import run_in_container
from hip_tools.numeric_validator import parse_program_output
from utils.logger import get_logger

logger = get_logger(__name__)


class RuntimeAgent:

    def run(self, state: HIPDebugState) -> HIPDebugState:
        binary = state.get("binary_container_path", "")
        if not binary:
            logger.warning("[runtime] no binary path — skipping")
            state["runtime_result"] = {"status": "error", "stdout": "", "stderr": "no binary", "exit_code": -1}
            state["numeric_result"] = {"status": "SKIPPED", "correct": False}
            return state

        r = run_in_container(binary, timeout=120)
        # Merge stdout + stderr for STATUS line detection (VMFault might dump on stderr)
        combined = r.stdout + "\n" + r.stderr
        numeric = parse_program_output(combined)

        # Post-processing: if the process died without printing STATUS, infer FAIL_RUNTIME
        if numeric.status in ("UNKNOWN", ""):
            if "vmfault" in combined.lower() or "illegal address" in combined.lower():
                numeric.status = "FAIL_RUNTIME"
                numeric.hip_err = "hipErrorIllegalAddress (VMFault detected)"
                numeric.correct = False
            elif r.exit_code != 0:
                numeric.status = "FAIL_RUNTIME"
                numeric.correct = False

        state["runtime_result"] = r.to_dict()
        state["numeric_result"] = numeric.to_dict()
        logger.info(
            f"[runtime] exit={r.exit_code} numeric_status={numeric.status} "
            f"correct={numeric.correct} elapsed_ms={numeric.elapsed_ms}"
        )
        return state
