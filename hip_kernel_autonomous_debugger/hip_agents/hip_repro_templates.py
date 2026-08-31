"""
Heuristic HIP kernel repair templates — used when the Anthropic LLM is
unavailable or its response fails validation. Each templater takes the
original source and returns (root_cause_dict, patch_plan_dict, patched_src).

We rely on regex/string surgery here — good enough for the 4 canonical bug
classes the demo covers; more elaborate cases should go through the LLM path.
"""
from __future__ import annotations

import re
from typing import Dict, Tuple


def _rc(hypothesis, mechanism, bug_type, conf=0.75, affected=()):
    return {
        "hypothesis": hypothesis,
        "mechanism": mechanism,
        "confidence": conf,
        "bug_type": bug_type,
        "affected_lines": list(affected),
        "_source": "heuristic",
    }


def _pp(summary, diff, expected, validation=()):
    return {
        "summary": summary,
        "diff": diff,
        "risks": ["Heuristic template — verify with recompile/rerun"],
        "expected_effect": expected,
        "validation_steps": list(validation) or ["recompile", "rerun", "check STATUS: PASS"],
    }


# ─── Templates ──────────────────────────────────────────────────────────────

def fix_out_of_bounds(src: str) -> Tuple[Dict, Dict, str]:
    # Replace `<=` with `<` in `for (...; ... <= N; ...)` inside kernel body
    patched, n = re.subn(r"(for\s*\([^;]*;\s*[^;]*)<=(\s*\w+[^;]*;\s*[^)]*\))", r"\1<\2", src)
    return (
        _rc("Grid-stride loop condition uses `<= N` and accesses arr[N], one past end.",
            "Buffers are allocated as N * sizeof(T); index N is out of bounds. "
            "DCU flags this as VMFault / hipErrorIllegalAddress.",
            "out_of_bounds", conf=0.9),
        _pp("Change `i <= N` to `i < N` in the grid-stride loop.",
            "- for (int i = tid; i <= N; i += stride)\n+ for (int i = tid; i < N; i += stride)",
            "STATUS: PASS with max_abs_err ~= 0.",
            ["hipcc -O2 -std=c++17 patched.hip -o patched", "./patched", "grep 'STATUS: PASS'"]),
        patched,
    )


def fix_reduction_race(src: str) -> Tuple[Dict, Dict, str]:
    """Insert __syncthreads() before the reduction loop and inside each iter."""
    # 1) Right after `sdata[tid] = ...` line, insert __syncthreads() if not there
    patched = re.sub(
        r"(sdata\[tid\]\s*=\s*[^;]+;)\s*\n",
        r"\1\n    __syncthreads();\n",
        src,
        count=1,
    )
    # 2) Inside the `for (int stride = ...; ...)` loop, insert __syncthreads() as first line
    patched = re.sub(
        r"(for\s*\(\s*int\s+stride\s*=\s*blockDim\.x\s*/\s*2[^)]*\)\s*\{)",
        r"\1\n        __syncthreads();",
        patched,
        count=1,
    )
    return (
        _rc("Tree reduction has no __syncthreads() between halving rounds.",
            "Threads in different waves (block=256 spans 4 waves on DCU) read stale "
            "sdata values before writes from the previous stride are visible, yielding "
            "systematically low and non-deterministic sums.",
            "reduction_race", conf=0.92),
        _pp("Add __syncthreads() after the initial shared-mem load and at the top of each reduction round.",
            "  sdata[tid] = ...;\n+ __syncthreads();\n  for (int stride = ...; ...) {\n+   __syncthreads();\n    if (tid < stride) sdata[tid] += sdata[tid+stride];\n  }",
            "STATUS: PASS with sum == N for every run; RUN_SPREAD delta = 0.",
            ["rerun 5 times and confirm identical sums"]),
        patched,
    )


def fix_bank_conflict(src: str) -> Tuple[Dict, Dict, str]:
    # Replace `tile[N][N]` where N in {16,32,64} with `tile[N][N+1]`
    def repl(m):
        a, b = int(m.group(2)), int(m.group(3))
        if a == b and a in (16, 32, 64):
            return f"{m.group(1)}[{a}][{a + 1}]"
        return m.group(0)
    patched = re.sub(r"(__shared__[^\[]+)\[(\d+)\]\[(\d+)\]", repl, src)
    return (
        _rc("32-wide shared-memory tile causes systematic LDS bank conflicts.",
            "DCU LDS has 32 banks × 4 B. A `tile[32][32]` layout puts every column "
            "in the same bank; column read `tile[x][y]` serializes across the wave.",
            "bank_conflict", conf=0.88),
        _pp("Add +1 padding to the second dimension of the shared tile.",
            "- __shared__ float tile[32][32];\n+ __shared__ float tile[32][33];",
            "Correctness unchanged; ELAPSED_MS drops by 1.5–3×.",
            ["compare ELAPSED_MS baseline vs patched"]),
        patched,
    )


def fix_uncoalesced_gemm(src: str) -> Tuple[Dict, Dict, str]:
    # Swap A[k*M+row] → A[row*K+k]
    patched = re.sub(r"A\s*\[\s*k\s*\*\s*M\s*\+\s*row\s*\]", "A[row * K + k]", src)
    return (
        _rc("SGEMM indexes A as column-major although the host stores it row-major.",
            "`A[k*M+row]` reads the transpose of A → every element of C is wrong. "
            "Also breaks coalescing: neighbors along threadIdx.x differ by row, not k, "
            "so adjacent threads touch stride-M offsets in A.",
            "uncoalesced_access", conf=0.9),
        _pp("Swap the A index to match row-major layout.",
            "- float a = A[k * M + row];\n+ float a = A[row * K + k];",
            "STATUS: PASS with max_abs_err ~= 0 and improved coalescing.",
            ["diff CPU_REFERENCE vs GPU_RESULT is zero on first 4 elements"]),
        patched,
    )


TEMPLATES = {
    "out_of_bounds":       fix_out_of_bounds,
    "reduction_race":      fix_reduction_race,
    "missing_sync":        fix_reduction_race,
    "bank_conflict":       fix_bank_conflict,
    "uncoalesced_access":  fix_uncoalesced_gemm,
}


def build_fix(bug_class: str, original: str) -> Tuple[Dict, Dict, str]:
    fn = TEMPLATES.get(bug_class)
    if fn:
        return fn(original)
    # Unknown → return original with a note
    return (
        _rc(f"No template for bug_class={bug_class}.",
            "LLM produced no valid plan and no heuristic template matches this class.",
            "unknown", conf=0.1),
        _pp("Manual investigation required.",
            "(no diff)",
            "N/A",
            ["Manual review"]),
        original,
    )
