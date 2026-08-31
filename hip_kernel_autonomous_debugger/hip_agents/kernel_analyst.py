"""
Kernel Analyst Agent
Runs kernel_parser.parse_kernel_source() over the .hip file to surface
suspicious patterns (loop_le_bound, shared_no_sync, tile_bank_conflict, ...)
without any LLM call. These findings feed Fix Planner's evidence bundle.
"""
from __future__ import annotations

from hip_orchestrator.state import HIPDebugState
from hip_tools.kernel_parser import parse_kernel_source
from utils.logger import get_logger

logger = get_logger(__name__)


class KernelAnalystAgent:

    def run(self, state: HIPDebugState) -> HIPDebugState:
        src = state.get("kernel_source", "") or ""
        result = parse_kernel_source(src)
        state["kernel_parse"] = result.to_dict()
        logger.info(
            f"[kernel_analyst] {len(result.kernels)} kernels, "
            f"{len(result.suspicious_patterns)} suspicious patterns, "
            f"{len(result.launches)} launches"
        )
        return state
