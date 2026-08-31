"""
Docker Runner
Executes shell commands inside the feng_picasso container that hosts the HIP toolchain.

We orchestrate everything from the container as well, so this module is currently a thin
wrapper around subprocess. If the pipeline is ever driven from the host (outside the
container), swap `_run_local` for `_run_docker_exec` — no other change needed.
"""
from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from utils.logger import get_logger

logger = get_logger(__name__)

CONTAINER_NAME = os.environ.get("HIP_CONTAINER", "feng_picasso")
HOST_WORKSPACE_PREFIX = "/public/home/dingf"
CONTAINER_WORKSPACE_PREFIX = "/workspace"

# Are we currently inside the container?
INSIDE_CONTAINER = os.path.isdir(CONTAINER_WORKSPACE_PREFIX) and not os.path.isdir(HOST_WORKSPACE_PREFIX)


@dataclass
class ExecResult:
    status: str          # "ok" | "error" | "timeout"
    stdout: str
    stderr: str
    exit_code: int
    duration_s: float
    command: str

    def to_dict(self) -> Dict:
        return {
            "status": self.status,
            "stdout": self.stdout[:8000],
            "stderr": self.stderr[:8000],
            "exit_code": self.exit_code,
            "duration_s": round(self.duration_s, 3),
            "command": self.command,
        }


def host_to_container(path: str) -> str:
    """Map a host path under /public/home/dingf to /workspace."""
    if path.startswith(HOST_WORKSPACE_PREFIX):
        return CONTAINER_WORKSPACE_PREFIX + path[len(HOST_WORKSPACE_PREFIX):]
    return path


def container_to_host(path: str) -> str:
    if path.startswith(CONTAINER_WORKSPACE_PREFIX):
        return HOST_WORKSPACE_PREFIX + path[len(CONTAINER_WORKSPACE_PREFIX):]
    return path


def run_in_container(
    cmd: str,
    timeout: int = 60,
    cwd: Optional[str] = None,
    env_extra: Optional[Dict[str, str]] = None,
) -> ExecResult:
    """Execute `cmd` (a shell string) in the DCU container.

    When running inside the container already, use local subprocess.
    """
    full_cmd = f"cd {cwd} && {cmd}" if cwd else cmd
    logger.info(f"[docker_runner] exec: {full_cmd[:180]}")

    start = time.time()
    if INSIDE_CONTAINER:
        return _run_local(full_cmd, timeout, env_extra, start)
    return _run_docker_exec(full_cmd, timeout, env_extra, start)


def _run_local(full_cmd: str, timeout: int, env_extra: Optional[Dict[str, str]], start: float) -> ExecResult:
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    try:
        proc = subprocess.run(
            ["bash", "-lc", full_cmd],
            capture_output=True,
            timeout=timeout,
            env=env,
        )
        return ExecResult(
            status="ok" if proc.returncode == 0 else "error",
            stdout=proc.stdout.decode("utf-8", errors="replace"),
            stderr=proc.stderr.decode("utf-8", errors="replace"),
            exit_code=proc.returncode,
            duration_s=time.time() - start,
            command=full_cmd,
        )
    except subprocess.TimeoutExpired as e:
        return ExecResult(
            status="timeout",
            stdout=(e.stdout or b"").decode("utf-8", errors="replace"),
            stderr=(e.stderr or b"").decode("utf-8", errors="replace") + f"\n[TIMEOUT after {timeout}s]",
            exit_code=-9,
            duration_s=time.time() - start,
            command=full_cmd,
        )


def _run_docker_exec(full_cmd: str, timeout: int, env_extra: Optional[Dict[str, str]], start: float) -> ExecResult:
    env_args: List[str] = []
    if env_extra:
        for k, v in env_extra.items():
            env_args.extend(["-e", f"{k}={v}"])
    argv = ["docker", "exec"] + env_args + [CONTAINER_NAME, "bash", "-lc", full_cmd]
    try:
        proc = subprocess.run(argv, capture_output=True, timeout=timeout)
        return ExecResult(
            status="ok" if proc.returncode == 0 else "error",
            stdout=proc.stdout.decode("utf-8", errors="replace"),
            stderr=proc.stderr.decode("utf-8", errors="replace"),
            exit_code=proc.returncode,
            duration_s=time.time() - start,
            command=full_cmd,
        )
    except subprocess.TimeoutExpired as e:
        return ExecResult(
            status="timeout",
            stdout=(e.stdout or b"").decode("utf-8", errors="replace"),
            stderr=(e.stderr or b"").decode("utf-8", errors="replace") + f"\n[TIMEOUT after {timeout}s]",
            exit_code=-9,
            duration_s=time.time() - start,
            command=full_cmd,
        )
