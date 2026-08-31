"""
Numeric Validator
Parses GPU program stdout and compares against a CPU reference or expected value.

Our HIP bug binaries follow a strict output protocol so this parser stays simple:

    PROBLEM: <name>
    N: <count>
    GPU_RESULT: <space-separated numbers, up to 32 samples>
    CPU_REFERENCE: <space-separated numbers, up to 32 samples>
    MAX_ABS_ERR: <float>
    ELAPSED_MS: <float>
    STATUS: <PASS|FAIL_NUMERIC|FAIL_RUNTIME|FAIL_LAUNCH>
    HIP_ERR: <string or NONE>

Any missing tag is tolerated — we return whatever we can extract.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from utils.logger import get_logger

logger = get_logger(__name__)


TAG_RE = re.compile(r"^([A-Z_]+):\s*(.*)$", re.MULTILINE)


@dataclass
class NumericResult:
    problem: str = ""
    n: int = 0
    gpu_sample: List[float] = field(default_factory=list)
    cpu_sample: List[float] = field(default_factory=list)
    max_abs_err: Optional[float] = None
    elapsed_ms: Optional[float] = None
    status: str = "UNKNOWN"          # PASS | FAIL_NUMERIC | FAIL_RUNTIME | FAIL_LAUNCH
    hip_err: str = "NONE"
    correct: bool = False
    first_mismatch_idx: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "problem": self.problem,
            "n": self.n,
            "gpu_sample": self.gpu_sample,
            "cpu_sample": self.cpu_sample,
            "max_abs_err": self.max_abs_err,
            "elapsed_ms": self.elapsed_ms,
            "status": self.status,
            "hip_err": self.hip_err,
            "correct": self.correct,
            "first_mismatch_idx": self.first_mismatch_idx,
        }


def _parse_floats(text: str) -> List[float]:
    out: List[float] = []
    for tok in text.replace(",", " ").split():
        try:
            out.append(float(tok))
        except ValueError:
            continue
    return out


def _parse_int(text: str, default: int = 0) -> int:
    try:
        return int(text.strip())
    except (ValueError, AttributeError):
        return default


def _parse_float(text: str) -> Optional[float]:
    try:
        return float(text.strip())
    except (ValueError, AttributeError):
        return None


def parse_program_output(stdout: str, tolerance: float = 1e-5) -> NumericResult:
    """Extract structured NumericResult from the program's stdout."""
    tags: Dict[str, str] = {}
    for m in TAG_RE.finditer(stdout):
        tags[m.group(1).upper()] = m.group(2).strip()

    r = NumericResult(
        problem=tags.get("PROBLEM", ""),
        n=_parse_int(tags.get("N", "0")),
        gpu_sample=_parse_floats(tags.get("GPU_RESULT", "")),
        cpu_sample=_parse_floats(tags.get("CPU_REFERENCE", "")),
        max_abs_err=_parse_float(tags.get("MAX_ABS_ERR", "")),
        elapsed_ms=_parse_float(tags.get("ELAPSED_MS", "")),
        status=tags.get("STATUS", "UNKNOWN").upper(),
        hip_err=tags.get("HIP_ERR", "NONE"),
    )
    # Correctness verdict from binary's own STATUS if provided; otherwise fall back to samples
    if r.status == "PASS":
        r.correct = True
    elif r.status.startswith("FAIL"):
        r.correct = False
    else:
        # No explicit STATUS — infer from samples
        if r.gpu_sample and r.cpu_sample:
            paired = list(zip(r.gpu_sample, r.cpu_sample))
            for i, (g, c) in enumerate(paired):
                if math.isnan(g) or math.isnan(c) or abs(g - c) > tolerance:
                    r.first_mismatch_idx = i
                    r.correct = False
                    break
            else:
                r.correct = True
                r.status = "PASS"
    if not r.correct and r.first_mismatch_idx is None and r.gpu_sample and r.cpu_sample:
        for i, (g, c) in enumerate(zip(r.gpu_sample, r.cpu_sample)):
            if abs(g - c) > tolerance:
                r.first_mismatch_idx = i
                break
    logger.info(
        f"[numeric_validator] problem={r.problem} status={r.status} correct={r.correct} "
        f"max_err={r.max_abs_err} elapsed_ms={r.elapsed_ms}"
    )
    return r


def compare_speedup(baseline_ms: float, patched_ms: float) -> Dict[str, Any]:
    if not baseline_ms or not patched_ms or patched_ms <= 0:
        return {"speedup": None, "note": "insufficient timing data"}
    return {
        "speedup": round(baseline_ms / patched_ms, 3),
        "baseline_ms": round(baseline_ms, 3),
        "patched_ms": round(patched_ms, 3),
    }
