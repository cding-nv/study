# HIP Kernel Autonomous Debugger

A multi-agent LangGraph pipeline that takes a buggy `.hip` file + a bug
report, and drives the full **compile → run → diagnose → patch → re-verify**
loop on a hygon DCU (BW151). Uses **Anthropic Claude** for reasoning,
augmented with the `hipc_kernel_opt_skills` SKILL.md as a domain-knowledge
system prompt.

Sits alongside the original Python bug pipeline (`main.py`); this one is
invoked via `main_hip.py`.

## Requirements

- Container `feng_picasso` (or set `HIP_CONTAINER` env) with:
  - `hipcc` toolchain (DTK 26.04 / gfx936 tested)
  - `python 3.10` + `pip install anthropic langgraph`
- Anthropic env vars: `ANTHROPIC_API_KEY` / `ANTHROPIC_AUTH_TOKEN` / `ANTHROPIC_BASE_URL` / `ANTHROPIC_MODEL`

## Quick start (from inside the container)

```bash
cd /workspace/workspace/Autonomous-debugging-system
python main_hip.py --bug   hip_bugs/bug_reports/HIP-002-reduction-race.json \
                   --kernel hip_bugs/kernels/reduction_race.hip
```

Or from the host, wrapping the whole call:

```bash
docker exec \
  -e ANTHROPIC_API_KEY -e ANTHROPIC_AUTH_TOKEN -e ANTHROPIC_BASE_URL -e ANTHROPIC_MODEL \
  feng_picasso bash -lc "cd /workspace/workspace/Autonomous-debugging-system && \
    /data/miniconda3/envs/env-3.10/bin/python3 main_hip.py \
    --bug   hip_bugs/bug_reports/HIP-002-reduction-race.json \
    --kernel hip_bugs/kernels/reduction_race.hip"
```

Reports land at `outputs/hip_reports/<BUG-ID>/report_<BUG-ID>.json`; the
patched kernel and its compiled binary live in the same directory.

## Pipeline

```
preprocess → triage → build ─┬─ ok  ─→ runtime → kernel_analyst → probe → fix → verify → reviewer → finalize
                             └─ err ────────────────────────→ fix ────────→ verify → reviewer → finalize
```

| Agent | Responsibility |
|---|---|
| **preprocess**     | Check Anthropic reachability + kernel source presence |
| **triage**         | Classify bug (`out_of_bounds`, `reduction_race`, `bank_conflict`, `uncoalesced_access`, ...) — LLM first, keyword fallback |
| **build**          | `hipcc -O2 -std=c++17` compile inside container; collect warnings/errors |
| **runtime**        | Run the baseline binary; parse `STATUS/GPU_RESULT/MAX_ABS_ERR/ELAPSED_MS/HIP_ERR` protocol |
| **kernel_analyst** | Regex-based static analysis: `__syncthreads` presence, `__shared__` tiles, `<=` loop bounds, atomics |
| **probe**          | Rerun 5× to detect non-determinism (races produce run-to-run variance) |
| **fix**            | LLM plan (SKILL.md injected in system prompt) → produces `patched_kernel_source`; heuristic templates as fallback |
| **verify**         | Write patched source → recompile → rerun → compare correctness + speedup |
| **reviewer**       | Structural checks: patched file preserves output protocol; contradictions between claim and verify outcome |
| **finalize**       | Aggregate confidence score |

## Bug catalog

Four canonical HIP kernel bugs live in `hip_bugs/`:

| ID | Kernel | Bug | Symptom | Fix |
|---|---|---|---|---|
| HIP-001 | `vector_add_oob.hip` | Grid-stride loop uses `<= N` | KERNEL VMFault / hipErrorIllegalAddress | `<= N` → `< N` |
| HIP-002 | `reduction_race.hip` | No `__syncthreads()` between reduction rounds | FAIL_NUMERIC + non-deterministic sums | Insert barriers |
| HIP-003 | `transpose_bank_conflict.hip` | `__shared__ float tile[32][32]` | Correct but slow (LDS bank conflict) | `[32][33]` padding |
| HIP-004 | `gemm_uncoalesced.hip` | `A[k*M+row]` on row-major A | FAIL_NUMERIC — every element wrong | `A[row*K+k]` |

Each kernel's `main()` follows a strict output protocol:

```
PROBLEM: <name>
N: <count>
GPU_RESULT: <space-separated numbers>
CPU_REFERENCE: <space-separated numbers>
MAX_ABS_ERR: <float>
ELAPSED_MS: <float>
HIP_ERR: <string or NONE>
STATUS: PASS | FAIL_NUMERIC | FAIL_RUNTIME | FAIL_LAUNCH
```

`hip_tools/numeric_validator.py` parses this into a `NumericResult`.

## Adding a new bug

1. Drop a `.hip` file into `hip_bugs/kernels/` that follows the output protocol.
2. Write a `bug_reports/HIP-XXX-*.json` with `id / title / severity / description / expected_behavior / actual_behavior / reproduction_hints`.
3. Point `main_hip.py` at the pair. If your bug isn't in the four canonical classes, the LLM path will handle it — the heuristic fallback only knows those four.

## Files

```
main_hip.py                          # CLI entry
hip_orchestrator/
  ├── state.py                       # HIPDebugState TypedDict
  └── graph.py                       # LangGraph flow
hip_agents/
  ├── triage_agent.py                # bug-class classifier (LLM + heuristic)
  ├── build_agent.py                 # hipcc compile
  ├── runtime_agent.py               # execute + parse protocol
  ├── kernel_analyst.py              # regex-based static analysis
  ├── probe_agent.py                 # rerun N times → detect non-determinism
  ├── fix_agent.py                   # LLM patch generator (SKILL.md system prompt)
  ├── reviewer_agent.py              # structural + contradiction checks
  ├── verify_agent.py                # patched compile+rerun+compare
  └── hip_repro_templates.py         # heuristic fallback fixes for 4 classes
hip_tools/
  ├── anthropic_client.py            # Anthropic wrapper + SKILL.md loader
  ├── docker_runner.py               # subprocess / docker exec unified interface
  ├── hipcc_wrapper.py               # compile + warning/error parsing
  ├── kernel_parser.py               # regex-based structural analysis
  └── numeric_validator.py           # parse STATUS protocol + compute speedup
hip_bugs/
  ├── kernels/*.hip
  └── bug_reports/*.json
```

## Design decisions

- **We do NOT generate synthetic repro scripts** (unlike the Python
  version). Each buggy kernel *is* its own repro — self-contained with
  a CPU reference and a `STATUS:` verdict. The pipeline's job is to
  observe the failure, diagnose it, patch the source, and prove the fix.
- **Static analysis stays regex-based.** C++ has no cheap AST for HIP;
  our regex catches the four canonical patterns and leaves the rest to
  the LLM, which gets the full source anyway.
- **SKILL.md is loaded once** at import time and appended to fix / reviewer
  system prompts. This grounds the LLM in hygon-specific facts (wave=64,
  LDS 32 banks × 4 B, `__launch_bounds__`, +1 padding, etc.) so it doesn't
  give generic ROCm advice.
- **Verification closes the loop.** Confidence score jumps to 1.0 only
  when the patched kernel compiles, runs, and returns the correct output.
