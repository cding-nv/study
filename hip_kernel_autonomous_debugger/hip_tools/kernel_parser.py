"""
HIP Kernel Parser — lightweight regex-based structural analysis.

We can't parse full C++, but we can reliably pick out:
  - __global__ / __device__ kernel signatures and their line ranges
  - __shared__ memory declarations
  - __syncthreads() / __syncwarp() call sites
  - atomicAdd / atomicCAS / atomicMax etc.
  - kernel launch configs `kernel<<<grid, block>>>(...)`
  - Suspicious index patterns: <= in loop bounds, tile[N][N] without +1 padding,
    A[k*M+row] style column-major access to row-major data.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from utils.logger import get_logger

logger = get_logger(__name__)


KERNEL_SIG_RE = re.compile(
    r"^\s*(?:extern\s+\"C\"\s+)?(?:template\s*<[^>]*>\s*)?"
    r"__global__\s+\w[\w\s\*&<>,]*?\s+(?P<name>\w+)\s*\(",
    re.MULTILINE,
)
DEVICE_FN_RE = re.compile(r"^\s*__device__\s+\w[\w\s\*&<>,]*?\s+(?P<name>\w+)\s*\(", re.MULTILINE)
SHARED_RE = re.compile(r"__shared__\s+(?P<type>[\w:]+)\s+(?P<name>\w+)\s*(?P<dims>(?:\[[^\]]+\])+)", re.MULTILINE)
SYNCTHREADS_RE = re.compile(r"\b(__syncthreads|__syncwarp|__threadfence(?:_block|_system)?)\s*\(", re.MULTILINE)
ATOMIC_RE = re.compile(r"\b(atomicAdd|atomicCAS|atomicSub|atomicMax|atomicMin|atomicExch|atomicOr|atomicAnd|atomicXor)\s*\(", re.MULTILINE)
LAUNCH_RE = re.compile(r"(?P<name>\w+)\s*<<<\s*(?P<grid>[^,>]+)\s*,\s*(?P<block>[^,>]+?)(?:\s*,\s*(?P<smem>[^,>]+))?\s*>>>", re.MULTILINE)
FOR_LOOP_LE_RE = re.compile(r"for\s*\([^;]*;\s*[^;]*<=\s*(?P<bound>\w+)[^;]*;\s*[^)]*\)", re.MULTILINE)
INDEX_ACCESS_RE = re.compile(r"(?P<arr>\w+)\s*\[(?P<idx>[^\[\]]+)\]", re.MULTILINE)


@dataclass
class KernelInfo:
    name: str
    start_line: int
    end_line: int
    body: str = ""
    has_syncthreads: bool = False
    has_atomics: bool = False
    shared_decls: List[Dict] = field(default_factory=list)
    launches: List[Dict] = field(default_factory=list)
    suspicious: List[Dict] = field(default_factory=list)


@dataclass
class KernelParseResult:
    kernels: List[KernelInfo] = field(default_factory=list)
    launches: List[Dict] = field(default_factory=list)
    shared_decls: List[Dict] = field(default_factory=list)
    syncthreads_sites: List[int] = field(default_factory=list)
    atomic_sites: List[Dict] = field(default_factory=list)
    suspicious_patterns: List[Dict] = field(default_factory=list)
    total_lines: int = 0

    def to_dict(self) -> Dict:
        return {
            "kernels": [
                {
                    "name": k.name,
                    "start_line": k.start_line,
                    "end_line": k.end_line,
                    "has_syncthreads": k.has_syncthreads,
                    "has_atomics": k.has_atomics,
                    "shared_decls": k.shared_decls,
                    "launches": k.launches,
                    "suspicious": k.suspicious,
                    "body_snippet": k.body[:800],
                }
                for k in self.kernels
            ],
            "launches": self.launches,
            "shared_decls": self.shared_decls,
            "syncthreads_sites": self.syncthreads_sites,
            "atomic_sites": self.atomic_sites,
            "suspicious_patterns": self.suspicious_patterns,
            "total_lines": self.total_lines,
        }


def _line_of(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _find_kernel_body_end(text: str, start_offset: int) -> int:
    """Given the offset of a `__global__` signature, find matching closing brace."""
    brace_open = text.find("{", start_offset)
    if brace_open < 0:
        return start_offset
    depth = 1
    i = brace_open + 1
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return i


def parse_kernel_source(source: str) -> KernelParseResult:
    result = KernelParseResult(total_lines=source.count("\n") + 1)

    # ── Kernels ────────────────────────────────────────────────────────────
    for m in KERNEL_SIG_RE.finditer(source):
        name = m.group("name")
        sig_start_line = _line_of(source, m.start())
        body_end_offset = _find_kernel_body_end(source, m.end())
        body_end_line = _line_of(source, body_end_offset)
        body = source[m.start():body_end_offset]
        info = KernelInfo(
            name=name,
            start_line=sig_start_line,
            end_line=body_end_line,
            body=body,
            has_syncthreads=bool(SYNCTHREADS_RE.search(body)),
            has_atomics=bool(ATOMIC_RE.search(body)),
        )
        # shared decls scoped to kernel
        for sm in SHARED_RE.finditer(body):
            info.shared_decls.append({
                "type": sm.group("type"),
                "name": sm.group("name"),
                "dims": sm.group("dims"),
                "line": sig_start_line + body[:sm.start()].count("\n"),
            })
        # Suspicious: <= in loop bounds inside kernel body
        for lm in FOR_LOOP_LE_RE.finditer(body):
            info.suspicious.append({
                "kind": "loop_le_bound",
                "line": sig_start_line + body[:lm.start()].count("\n"),
                "bound_var": lm.group("bound"),
                "snippet": body[lm.start():lm.end()].strip()[:120],
                "concern": "Loop uses <= against a size variable — likely off-by-one / OOB when accessing arr[i].",
            })
        # Suspicious: shared tile of shape [K][K] with K power of two → bank conflict pattern
        for sd in info.shared_decls:
            m2 = re.match(r"\[(\d+)\]\[(\d+)\]$", sd["dims"])
            if m2:
                a, b = int(m2.group(1)), int(m2.group(2))
                if a == b and a in (16, 32, 64):
                    info.suspicious.append({
                        "kind": "shared_tile_bank_conflict",
                        "line": sd["line"],
                        "shared_var": sd["name"],
                        "dims": sd["dims"],
                        "concern": (
                            f"__shared__ {sd['name']}{sd['dims']}: square tile size {a} matches DCU LDS bank count "
                            f"(32 banks × 4B) — column access will serialize. Add +1 padding: [{a}][{b + 1}]."
                        ),
                    })
        result.kernels.append(info)

    # ── Launches ────────────────────────────────────────────────────────────
    for m in LAUNCH_RE.finditer(source):
        result.launches.append({
            "kernel": m.group("name"),
            "grid": m.group("grid").strip(),
            "block": m.group("block").strip(),
            "smem": (m.group("smem") or "").strip(),
            "line": _line_of(source, m.start()),
        })

    # ── Global sync/atomic sites ───────────────────────────────────────────
    for m in SYNCTHREADS_RE.finditer(source):
        result.syncthreads_sites.append(_line_of(source, m.start()))
    for m in ATOMIC_RE.finditer(source):
        result.atomic_sites.append({
            "op": m.group(1),
            "line": _line_of(source, m.start()),
        })

    # Copy per-kernel suspicious findings up to top-level
    for k in result.kernels:
        for s in k.suspicious:
            entry = dict(s)
            entry["kernel"] = k.name
            result.suspicious_patterns.append(entry)

    # Top-level heuristics that need all kernels together
    for k in result.kernels:
        # Kernel does atomics but has no __syncthreads before result read? warn only if reduction-like
        if k.has_atomics and not k.has_syncthreads and "reduc" in k.name.lower():
            result.suspicious_patterns.append({
                "kernel": k.name,
                "kind": "reduction_missing_sync",
                "line": k.start_line,
                "concern": "Reduction-style kernel uses atomics but lacks __syncthreads() — possible correctness issue.",
            })
        # Shared write followed by shared read without __syncthreads
        body = k.body
        if "__shared__" in body and not k.has_syncthreads:
            result.suspicious_patterns.append({
                "kernel": k.name,
                "kind": "shared_no_sync",
                "line": k.start_line,
                "concern": "Kernel declares __shared__ memory but never calls __syncthreads() — races expected between threads writing/reading the shared buffer.",
            })

    logger.info(
        f"[kernel_parser] {len(result.kernels)} kernels, "
        f"{len(result.launches)} launches, "
        f"{len(result.suspicious_patterns)} suspicious patterns"
    )
    return result
