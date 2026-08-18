"""AI Radar entry point. Run with: python -m AI_Radar.run  (or python run.py)"""
from __future__ import annotations
import argparse
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

# support both `python run.py` (script) and `python -m AI_Radar.run` (module)
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from AI_Radar import config as _cfg
    from AI_Radar.collectors import ALL_COLLECTORS
    from AI_Radar.analyzers import keyword_trend, llm_synthesizer
    from AI_Radar.reporter import markdown as reporter
    from AI_Radar.utils import cache
else:
    from . import config as _cfg
    from .collectors import ALL_COLLECTORS
    from .analyzers import keyword_trend, llm_synthesizer
    from .reporter import markdown as reporter
    from .utils import cache


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname).1s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # tenacity retry logs are noisy
    logging.getLogger("httpx").setLevel(logging.WARNING)


def _run_collectors(cfg: dict, only: list[str] | None) -> dict[str, dict]:
    log = logging.getLogger("radar")
    picked = ALL_COLLECTORS
    if only:
        picked = [c for c in ALL_COLLECTORS if c.name in only]
        log.info("Running only: %s", [c.name for c in picked])

    results: dict[str, dict] = {}
    concurrency = min(cfg.get("concurrent_limit", 8), len(picked))
    with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="collector") as pool:
        futures = {pool.submit(cls(cfg).run): cls.name for cls in picked}
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                results[name] = fut.result()
                status = "OK" if results[name]["ok"] else "FAIL"
                log.info("  %-16s  %s  (%d items)", name, status, results[name]["count"])
            except Exception as e:  # noqa: BLE001
                log.exception("collector %s crashed: %s", name, e)
                results[name] = {"ok": False, "count": 0, "items": [], "error": str(e)}
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AI Radar daily digest builder")
    parser.add_argument("--only", nargs="*", default=None,
                        help="only run these collectors (names), e.g. --only huggingface arxiv")
    parser.add_argument("--no-llm", action="store_true", help="skip LLM synthesis, use rules only")
    parser.add_argument("--from-cache", action="store_true",
                        help="do not fetch — reuse today's cached collector output to rebuild the report")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--day", default=None, help="YYYY-MM-DD (for --from-cache regeneration)")
    args = parser.parse_args(argv)

    _setup_logging(args.verbose)
    log = logging.getLogger("radar")

    cfg = _cfg.load_config()
    if args.no_llm:
        cfg.setdefault("llm", {})["enabled"] = False

    day = date.today()
    if args.day:
        from datetime import datetime as _dt
        day = _dt.strptime(args.day, "%Y-%m-%d").date()

    t0 = time.time()

    # 1. collect
    if args.from_cache:
        log.info("Loading cached payloads for %s", day.isoformat())
        payloads: dict[str, dict] = {}
        for cls in ALL_COLLECTORS:
            p = cache.load(cls.name, day)
            if p is not None:
                payloads[cls.name] = p
        if not payloads:
            log.error("No cache for %s under %s", day.isoformat(), _cfg.CACHE_DIR)
            return 2
    else:
        log.info("=== Collecting from %d sources ===", len(ALL_COLLECTORS) if not args.only else len(args.only))
        payloads = _run_collectors(cfg, args.only)

    # 2. analyze
    all_items: list[dict] = []
    for p in payloads.values():
        if p.get("ok"):
            all_items.extend(p["items"])
    log.info("Total items across sources: %d", len(all_items))

    kw = keyword_trend.analyze(all_items, cfg.get("keywords", {}))
    log.info("Top keywords: %s", [k for k, _, _ in kw["top_keywords"][:5]])

    # 3. synthesize
    log.info("Running LLM synthesis (enabled=%s)", cfg.get("llm", {}).get("enabled"))
    synthesis = llm_synthesizer.synthesize(payloads, kw, cfg)
    log.info("Synthesis generated_by: %s", synthesis.get("_generated_by"))

    # 4. report
    content = reporter.build_report(day, payloads, kw, synthesis)
    fp = reporter.write_report(day, content)
    log.info("Report written: %s  (%.1fs total)", fp, time.time() - t0)

    # 5. prune old cache
    cache.prune(cfg.get("cache_days", 30))

    # print the path to stdout so the user can click it.
    # Windows default console is GBK — fall back to ASCII if the emoji trips it.
    try:
        print(f"\n✅ Report: {fp}")
    except UnicodeEncodeError:
        print(f"\n[OK] Report: {fp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
