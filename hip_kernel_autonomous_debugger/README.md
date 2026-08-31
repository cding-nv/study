# HIP Kernel Autonomous Debugger

A multi-agent LangGraph pipeline that takes a buggy `.hip` file plus a bug
report and drives the full **compile → run → diagnose → patch → re-verify**
loop on a hygon DCU (BW151). Reasoning is done by any
**Anthropic-Messages-API-compatible LLM** (Claude, DeepSeek, and others),
augmented with the `hipc_kernel_opt_skills` **SKILL.md** as a
domain-knowledge system prompt.

Given a broken kernel, the system autonomously produces:
- structured evidence (compile warnings, runtime status, HIP errors, static-analysis findings, non-determinism probing),
- a root-cause hypothesis with a mechanism explanation,
- a **patched kernel source ready to compile**,
- and a verification result proving the patch (recompiles, runs, passes numerics, sometimes with a measured speedup).

## Verified end-to-end (4 canonical bug classes)

| ID | Kernel | Bug | Baseline symptom | AI diagnosis (conf.) | Patch | Verify | Speedup |
|---|---|---|---|---|---|---|---|
| HIP-001 | `vector_add_oob.hip` | Grid-stride loop `i <= N` | KERNEL VMFault / `hipErrorIllegalAddress` | `out_of_bounds` (0.99) | `<= N` → `< N` | ✅ PASS | — |
| HIP-002 | `reduction_race.hip` | Missing `__syncthreads()` in tree reduction | `FAIL_NUMERIC`, non-deterministic sums | `missing_sync` (0.98) | Insert two `__syncthreads()` | ✅ PASS | **7.13×** |
| HIP-003 | `transpose_bank_conflict.hip` | `__shared__ float tile[32][32]` | Correct but slow (LDS bank conflict) | `bank_conflict` (0.95) | `[32][33]` padding | ✅ PASS | 1.02× |
| HIP-004 | `gemm_uncoalesced.hip` | `A[k*M+row]` on row-major A | `FAIL_NUMERIC` — every element wrong | `uncoalesced_access` (0.98) | `A[row*K+k]` | ✅ PASS | 1.39× |

Full reports live under [`outputs/hip_reports/`](outputs/hip_reports/).

## Requirements

- **HIP toolchain**: `hipcc` (tested with DTK 26.04 / gfx936) and access to
  a hygon DCU (BW151 verified; other DCU generations should work but
  wave/LDS constants in the skill file are BW151-specific).
- **Python 3.10+** with two packages:
  ```
  pip install -r requirements.txt   # → anthropic, langgraph
  ```
- **Anthropic-compatible LLM endpoint**. The client reads:
  ```
  ANTHROPIC_API_KEY    or ANTHROPIC_AUTH_TOKEN
  ANTHROPIC_BASE_URL     (e.g. https://api.anthropic.com  or your DeepSeek-compat gateway)
  ANTHROPIC_MODEL        (any model your endpoint exposes)
  ```
  Works with Claude models directly, or with any provider that exposes an
  Anthropic-Messages-compatible endpoint (e.g. DeepSeek via a compatibility
  gateway). The pipeline was validated with both.

### Optional: run from outside the container

If your Python 3.10 and `hipcc` live inside a Docker container (the setup
used during development), invoke through `docker exec`. The runner
auto-detects whether it's inside or outside the container.

```bash
docker exec \
  -e ANTHROPIC_API_KEY -e ANTHROPIC_AUTH_TOKEN \
  -e ANTHROPIC_BASE_URL -e ANTHROPIC_MODEL \
  <container_name> bash -lc "cd /path/to/project && \
    python main_hip.py --bug hip_bugs/bug_reports/HIP-002-reduction-race.json \
                       --kernel hip_bugs/kernels/reduction_race.hip"
```

Set `HIP_CONTAINER=<name>` if you want the runner to call `docker exec`
itself. Set `HIP_SKILL_PATH=/abs/path/to/SKILL.md` if the domain skill
isn't at the default location.

## Quick start

```bash
git clone <repo>
cd hip_kernel_autonomous_debugger
pip install -r requirements.txt
export ANTHROPIC_API_KEY=<your-key>
export ANTHROPIC_BASE_URL=<endpoint>      # e.g. https://api.anthropic.com
export ANTHROPIC_MODEL=<model-id>

# Reproduce one of the four canonical bugs end-to-end
python main_hip.py \
    --bug   hip_bugs/bug_reports/HIP-002-reduction-race.json \
    --kernel hip_bugs/kernels/reduction_race.hip
```

Report goes to `outputs/hip_reports/<BUG-ID>/report_<BUG-ID>.json`; the
patched kernel source and its compiled binary live in the same directory.

## Pipeline

```
preprocess → triage → build ─┬─ ok  ─→ runtime → kernel_analyst → probe → fix → verify → reviewer → finalize
                             └─ err ────────────────────────────────────→ fix ────→ verify → reviewer → finalize
```

Every node is wrapped so an agent failure logs an error but never kills
the pipeline — the next agent still gets a chance to add signal.

| Agent | Job |
|---|---|
| **preprocess** | Verify Anthropic reachability + kernel source presence |
| **triage** | Classify bug (`out_of_bounds` / `reduction_race` / `bank_conflict` / `uncoalesced_access` / `launch_config_error` / `precision_error` / `atomic_contention` / `missing_sync`) — LLM first, keyword-heuristic fallback |
| **build** | `hipcc -O2 -std=c++17` inside container; capture warnings + errors |
| **runtime** | Run the baseline binary; parse `STATUS/GPU_RESULT/MAX_ABS_ERR/ELAPSED_MS/HIP_ERR` protocol; detect VMFault |
| **kernel_analyst** | Regex-based static analysis: `__syncthreads` presence, `__shared__` tile shapes, `<=` loop bounds, atomics, launch config |
| **probe** | Rerun 5× to detect run-to-run variance — the fingerprint of a race |
| **fix** | LLM plan **with `SKILL.md` injected into the system prompt** → produces the full `patched_kernel_source`. Heuristic templates handle the four canonical classes offline. |
| **verify** | Write patched source → recompile → rerun → check `STATUS: PASS` + measure speedup vs baseline |
| **reviewer** | Structural checks + LLM critique: patched file preserves output protocol; contradictions between claim and verify outcome are flagged |
| **finalize** | Aggregate confidence score (peaks at 1.0 when verify closes the loop) |

## Bug catalog & output protocol

Four bug samples in `hip_bugs/`:

```
hip_bugs/
├── README.md
├── bug_reports/
│   ├── HIP-001-vector-add-oob.json
│   ├── HIP-002-reduction-race.json
│   ├── HIP-003-transpose-bank-conflict.json
│   └── HIP-004-gemm-uncoalesced.json
└── kernels/
    ├── vector_add_oob.hip
    ├── reduction_race.hip
    ├── transpose_bank_conflict.hip
    └── gemm_uncoalesced.hip
```

Every kernel's `main()` follows a strict, machine-parseable protocol so
the pipeline can classify the outcome deterministically:

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

### Adding a new bug

1. Drop a `.hip` file into `hip_bugs/kernels/` that follows the output protocol.
2. Write a `bug_reports/HIP-XXX-*.json` with `id / title / severity / description / expected_behavior / actual_behavior / reproduction_hints`.
3. Point `main_hip.py` at the pair. If your bug isn't in the four
   canonical classes, the LLM path handles it — the heuristic fallback
   only covers the four templates in `hip_agents/hip_repro_templates.py`.

## Project layout

```
main_hip.py                         # CLI entry
hip_orchestrator/
├── state.py                        # HIPDebugState TypedDict
└── graph.py                        # LangGraph flow
hip_agents/
├── triage_agent.py                 # bug-class classifier (LLM + heuristic)
├── build_agent.py                  # hipcc compile
├── runtime_agent.py                # execute + parse protocol
├── kernel_analyst.py               # regex-based static analysis
├── probe_agent.py                  # 5× rerun → non-determinism detection
├── fix_agent.py                    # LLM patch generator (SKILL.md system prompt)
├── reviewer_agent.py               # structural + contradiction checks
├── verify_agent.py                 # patched compile + rerun + compare
└── hip_repro_templates.py          # heuristic fallback fixes for 4 bug classes
hip_tools/
├── anthropic_client.py             # Anthropic wrapper + SKILL.md loader
├── docker_runner.py                # in-container / local subprocess dispatch
├── hipcc_wrapper.py                # compile + warning/error parsing
├── kernel_parser.py                # regex-based structural analysis
└── numeric_validator.py            # STATUS protocol parser + speedup calc
hip_bugs/
├── kernels/*.hip                   # four buggy demo kernels
└── bug_reports/*.json              # matching bug reports
utils/logger.py                     # stdlib-only logging helper
outputs/hip_reports/                # per-run reports (gitignored)
outputs/hip_build/                  # baseline binaries (gitignored)
```

## Design choices

- **Kernels *are* the repros.** Unlike a generic Python debugger, we
  don't synthesize a repro script — every buggy kernel is self-contained
  with a CPU reference and a `STATUS:` verdict, so the pipeline can
  focus on observing → diagnosing → patching.
- **Regex-only static analysis.** C++ has no cheap AST for HIP; regex
  patterns cover the four canonical bug shapes and hand everything else
  to the LLM, which gets the full kernel source anyway.
- **`SKILL.md` injected as system prompt.** Loaded once at import time
  and appended to fix/reviewer system prompts. This grounds the LLM in
  hygon-specific facts (wave = 64, LDS 32 banks × 4 B, `__launch_bounds__`,
  `+1` tile padding, …) so it doesn't fall back to generic ROCm advice.
- **Verification closes the loop.** Confidence only reaches 1.0 when
  the patched kernel recompiles, reruns, and returns the correct output.
  A patch that "looks right" but fails re-verification lowers confidence
  and triggers a `REVISIT` recommendation from the reviewer.
- **Bring your own LLM.** Any endpoint exposing Anthropic's Messages API
  works. Validated with Claude and DeepSeek (via anthropic-compat gateway).

## Extension points

- **New bug class** — add a template to `hip_agents/hip_repro_templates.py`
  and a keyword group to `hip_agents/triage_agent.py::_heuristic`.
- **Different DCU / GPU** — replace the SKILL.md with a target-specific
  reference and set `HIP_SKILL_PATH` to point at it.
- **Different orchestration** — the graph in `hip_orchestrator/graph.py`
  is a stock LangGraph `StateGraph`; you can insert profiling or fuzzing
  nodes without touching agents.
