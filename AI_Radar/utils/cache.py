"""Cache raw collector output to cache/YYYY-MM-DD/<source>.json."""
from __future__ import annotations
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from .. import config as _cfg


def today_dir(today: date | None = None) -> Path:
    d = today or date.today()
    p = _cfg.CACHE_DIR / d.isoformat()
    p.mkdir(parents=True, exist_ok=True)
    return p


def save(source: str, data, today: date | None = None) -> Path:
    fp = today_dir(today) / f"{source}.json"
    with fp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    return fp


def load(source: str, day: date | None = None):
    d = day or date.today()
    fp = _cfg.CACHE_DIR / d.isoformat() / f"{source}.json"
    if not fp.exists():
        return None
    with fp.open("r", encoding="utf-8") as f:
        return json.load(f)


def prune(keep_days: int) -> None:
    if keep_days <= 0:
        return
    cutoff = date.today() - timedelta(days=keep_days)
    for child in _cfg.CACHE_DIR.iterdir():
        if not child.is_dir():
            continue
        try:
            d = datetime.strptime(child.name, "%Y-%m-%d").date()
        except ValueError:
            continue
        if d < cutoff:
            for f in child.iterdir():
                f.unlink()
            child.rmdir()
