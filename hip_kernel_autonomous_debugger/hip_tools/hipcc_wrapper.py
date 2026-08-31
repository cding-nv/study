"""
hipcc Wrapper
Compile HIP source files and parse warnings/errors emitted by the compiler.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from hip_tools.docker_runner import run_in_container, host_to_container
from utils.logger import get_logger

logger = get_logger(__name__)

WARNING_RE = re.compile(r"^(?P<file>[^\s:][^:]*):(?P<line>\d+):(?P<col>\d+):\s+warning:\s+(?P<msg>.+)$", re.MULTILINE)
ERROR_RE   = re.compile(r"^(?P<file>[^\s:][^:]*):(?P<line>\d+):(?P<col>\d+):\s+error:\s+(?P<msg>.+)$",   re.MULTILINE)


@dataclass
class CompilerIssue:
    kind: str          # "warning" | "error"
    file: str
    line: int
    col: int
    message: str


@dataclass
class BuildResult:
    status: str        # "ok" | "error"
    binary: str
    warnings: List[CompilerIssue] = field(default_factory=list)
    errors: List[CompilerIssue] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    duration_s: float = 0.0
    command: str = ""

    def to_dict(self) -> Dict:
        return {
            "status": self.status,
            "binary": self.binary,
            "warnings": [w.__dict__ for w in self.warnings],
            "errors":   [e.__dict__ for e in self.errors],
            "warning_count": len(self.warnings),
            "error_count":   len(self.errors),
            "stdout": self.stdout[:4000],
            "stderr": self.stderr[:4000],
            "duration_s": round(self.duration_s, 3),
            "command": self.command,
        }


def parse_compiler_output(text: str) -> Dict[str, List[CompilerIssue]]:
    warnings = [
        CompilerIssue(
            kind="warning",
            file=m.group("file"),
            line=int(m.group("line")),
            col=int(m.group("col")),
            message=m.group("msg").strip(),
        )
        for m in WARNING_RE.finditer(text)
    ]
    errors = [
        CompilerIssue(
            kind="error",
            file=m.group("file"),
            line=int(m.group("line")),
            col=int(m.group("col")),
            message=m.group("msg").strip(),
        )
        for m in ERROR_RE.finditer(text)
    ]
    return {"warnings": warnings, "errors": errors}


def compile_hip(
    hip_file_container_path: str,
    out_binary_container_path: str,
    extra_flags: Optional[List[str]] = None,
    timeout: int = 90,
) -> BuildResult:
    """Compile a HIP source file inside the container. Paths are container-side."""
    flags = extra_flags or []
    # -std=c++17 for structured bindings/if constexpr; -O2 for realistic timing
    cmd = f"hipcc -O2 -std=c++17 {' '.join(flags)} {hip_file_container_path} -o {out_binary_container_path}"
    r = run_in_container(cmd, timeout=timeout)
    parsed = parse_compiler_output(r.stdout + "\n" + r.stderr)
    status = "ok" if r.exit_code == 0 else "error"
    logger.info(
        f"[hipcc] {hip_file_container_path} → status={status} "
        f"warnings={len(parsed['warnings'])} errors={len(parsed['errors'])}"
    )
    return BuildResult(
        status=status,
        binary=out_binary_container_path if status == "ok" else "",
        warnings=parsed["warnings"],
        errors=parsed["errors"],
        stdout=r.stdout,
        stderr=r.stderr,
        duration_s=r.duration_s,
        command=r.command,
    )
