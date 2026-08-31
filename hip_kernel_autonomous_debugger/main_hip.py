#!/usr/bin/env python3
"""
HIP Kernel Autonomous Debugger — CLI entry point.

Example:
    python main_hip.py --bug hip_bugs/bug_reports/HIP-002-reduction-race.json \
                       --kernel hip_bugs/kernels/reduction_race.hip
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# make imports work when run from the project root
sys.path.insert(0, str(Path(__file__).parent))

from hip_orchestrator.graph import run_pipeline
from hip_tools.docker_runner import host_to_container, INSIDE_CONTAINER, CONTAINER_WORKSPACE_PREFIX
from utils.logger import get_logger

logger = get_logger("main_hip")

REPO_ROOT_HOST = Path(__file__).parent.resolve()


def resolve_paths(kernel_path_arg: str, output_dir_arg: str):
    """Turn user-provided paths into (host, container) pairs."""
    kernel_host = str(Path(kernel_path_arg).resolve())
    out_host    = str(Path(output_dir_arg).resolve())
    Path(out_host).mkdir(parents=True, exist_ok=True)

    if INSIDE_CONTAINER:
        # host == container inside the container
        return kernel_host, kernel_host, out_host, out_host

    kernel_container = host_to_container(kernel_host)
    out_container    = host_to_container(out_host)
    return kernel_host, kernel_container, out_host, out_container


def build_report(state):
    """Assemble a canonical JSON report from the final state."""
    br    = state.get("bug_report", {}) or {}
    tr    = state.get("triage", {}) or {}
    build = state.get("build_result", {}) or {}
    run   = state.get("runtime_result", {}) or {}
    num   = state.get("numeric_result", {}) or {}
    parse = state.get("kernel_parse", {}) or {}
    probe = state.get("probe_evidence", {}) or {}
    rc    = state.get("root_cause", {}) or {}
    pp    = state.get("patch_plan", {}) or {}
    rev   = state.get("review", {}) or {}
    ver   = state.get("verify_result", {}) or {}
    return {
        "bug_summary": {
            "id": br.get("id", "?"),
            "title": br.get("title", "?"),
            "severity": tr.get("severity", br.get("severity", "unknown")),
            "bug_class": tr.get("bug_class", "unknown"),
            "component": br.get("component", ""),
            "environment": br.get("environment", {}),
        },
        "evidence": {
            "build": {
                "status": build.get("status"),
                "warning_count": build.get("warning_count", 0),
                "error_count": build.get("error_count", 0),
                "warnings_head": build.get("warnings", [])[:5],
                "errors_head":   build.get("errors", [])[:5],
            },
            "runtime": {
                "status": run.get("status"),
                "exit_code": run.get("exit_code"),
                "duration_s": run.get("duration_s"),
                "numeric_status": num.get("status"),
                "correct": num.get("correct"),
                "max_abs_err": num.get("max_abs_err"),
                "elapsed_ms": num.get("elapsed_ms"),
                "gpu_sample": num.get("gpu_sample"),
                "cpu_sample": num.get("cpu_sample"),
                "hip_err": num.get("hip_err"),
                "stderr_tail": (run.get("stderr") or "")[-400:],
            },
            "static_analysis": {
                "kernels": [{"name": k["name"], "start_line": k["start_line"],
                             "has_syncthreads": k["has_syncthreads"],
                             "has_atomics": k["has_atomics"]}
                            for k in parse.get("kernels", [])],
                "suspicious_patterns": parse.get("suspicious_patterns", []),
                "launches": parse.get("launches", []),
            },
            "probe": probe,
        },
        "root_cause": rc,
        "patch_plan": pp,
        "verification": ver,
        "review": rev,
        "metadata": {
            "pipeline_confidence": state.get("confidence"),
            "llm_used": state.get("llm_available", False),
            "agent_trace": state.get("agent_trace", []),
            "errors": state.get("errors", []),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    }


def print_human_summary(report):
    sep = "=" * 74
    bs  = report["bug_summary"]
    rc  = report["root_cause"]
    pp  = report["patch_plan"]
    ver = report["verification"]
    rev = report["review"]
    ev  = report["evidence"]
    meta= report["metadata"]

    print(f"\n{sep}")
    print(f"  HIP KERNEL AUTONOMOUS DEBUGGER  —  {bs['id']}  {bs['title']}")
    print(sep)
    print(f"  Severity: {bs['severity']}   Bug class: {bs['bug_class']}   Component: {bs['component']}")

    print("\n▸ EVIDENCE")
    print(f"    Build:   status={ev['build']['status']}  warnings={ev['build']['warning_count']}  errors={ev['build']['error_count']}")
    r = ev["runtime"]
    print(f"    Runtime: exit={r['exit_code']}  numeric_status={r['numeric_status']}  correct={r['correct']}  max_err={r['max_abs_err']}")
    if r.get("hip_err") and r["hip_err"] != "NONE":
        print(f"    HIP err: {r['hip_err']}")
    sp = report["evidence"]["static_analysis"]["suspicious_patterns"]
    if sp:
        print(f"    Static:  {len(sp)} suspicious pattern(s)")
        for p in sp[:3]:
            print(f"      · [{p.get('kind')}] line {p.get('line')} — {p.get('concern','')[:90]}")

    print(f"\n▸ ROOT CAUSE  (confidence={rc.get('confidence')})")
    print(f"    Type:       {rc.get('bug_type')}")
    print(f"    Hypothesis: {rc.get('hypothesis','')}")
    if rc.get("mechanism"):
        print(f"    Mechanism:  {rc['mechanism'][:400]}")

    print("\n▸ PATCH PLAN")
    print(f"    {pp.get('summary','')}")
    if pp.get("diff"):
        print("    Diff:")
        for line in (pp["diff"] or "").splitlines()[:10]:
            print(f"      {line}")

    print("\n▸ VERIFICATION")
    if not ver.get("ran"):
        print(f"    Did not run — {ver.get('reason','compile failed')}")
    else:
        print(f"    Compiled: {ver.get('compiled')}    STATUS: {ver.get('status')}    correct: {ver.get('numerically_correct')}")
        if ver.get("speedup") is not None:
            print(f"    Speedup:  baseline={ver.get('baseline_elapsed_ms')} ms → patched={ver.get('patched_elapsed_ms')} ms   ({ver['speedup']}×)")
        if ver.get("patched_source_file"):
            print(f"    Patched file: {ver['patched_source_file']}")

    print(f"\n▸ REVIEW")
    print(f"    Recommendation: {rev.get('recommendation','')}")
    if rev.get("issues"):
        for i in rev["issues"][:5]:
            print(f"      · {i}")

    print(f"\n▸ PIPELINE")
    print(f"    Confidence: {meta['pipeline_confidence']}    LLM: {'yes' if meta['llm_used'] else 'no (heuristic)'}")
    for entry in meta.get("agent_trace", []):
        st = entry.get("status", "?")
        dur = entry.get("duration_s", 0)
        marker = "✔" if st == "success" else "✘"
        print(f"      {marker} {entry.get('agent'):<16} {st:<8} {dur:>7.2f}s")
    print(sep + "\n")


def main():
    ap = argparse.ArgumentParser(description="HIP kernel autonomous debugger")
    ap.add_argument("--bug",    "-b", required=True, help="bug_report .json path")
    ap.add_argument("--kernel", "-k", required=True, help=".hip source path")
    ap.add_argument("--output", "-o", default="outputs/hip_reports", help="report + build output dir (default: outputs/hip_reports)")
    ap.add_argument("--no-summary", action="store_true")
    args = ap.parse_args()

    bug_report = json.loads(Path(args.bug).read_text(encoding="utf-8"))
    kernel_source = Path(args.kernel).read_text(encoding="utf-8")

    # Build output subdirectory keyed by bug id
    out_base = Path(args.output) / bug_report.get("id", "UNKNOWN")
    kernel_host, kernel_container, out_host, out_container = resolve_paths(args.kernel, str(out_base))

    logger.info(f"Kernel  (host):      {kernel_host}")
    logger.info(f"Kernel  (container): {kernel_container}")
    logger.info(f"Output  (host):      {out_host}")

    t0 = time.time()
    final_state = run_pipeline(
        bug_report=bug_report,
        kernel_source=kernel_source,
        kernel_file_host=kernel_host,
        kernel_file_container=kernel_container,
        output_dir_host=out_host,
        output_dir_container=out_container,
    )
    elapsed = time.time() - t0
    logger.info(f"Pipeline finished in {elapsed:.1f}s")

    report = build_report(final_state)
    report_path = Path(out_host) / f"report_{bug_report.get('id','X')}.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\n📄 Report written to: {report_path}")

    if not args.no_summary:
        print_human_summary(report)

    # exit code: 0 if verification succeeded, 1 otherwise
    verified = bool(final_state.get("verify_result", {}).get("numerically_correct"))
    sys.exit(0 if verified else 1)


if __name__ == "__main__":
    main()
