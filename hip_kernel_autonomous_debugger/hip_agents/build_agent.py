"""
Build Agent
Compile the target HIP kernel via hipcc. If compile fails, we short-circuit
straight to the fix planner (skipping runtime).
"""
from __future__ import annotations

import os
from pathlib import Path

from hip_orchestrator.state import HIPDebugState
from hip_tools.docker_runner import host_to_container
from hip_tools.hipcc_wrapper import compile_hip
from utils.logger import get_logger

logger = get_logger(__name__)


class BuildAgent:

    def run(self, state: HIPDebugState) -> HIPDebugState:
        kernel_container = state["kernel_file_container"]
        out_dir_container = state["output_dir_container"]
        name = Path(kernel_container).stem
        binary_container = os.path.join(out_dir_container, f"{name}_baseline")

        br = compile_hip(kernel_container, binary_container, timeout=90)
        state["build_result"] = br.to_dict()
        state["binary_container_path"] = br.binary
        logger.info(
            f"[build] {name} status={br.status} "
            f"warnings={len(br.warnings)} errors={len(br.errors)}"
        )
        return state
