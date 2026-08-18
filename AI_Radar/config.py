"""Loads config.yaml with env-var overrides."""
from __future__ import annotations
import os
from pathlib import Path
from typing import Any
import yaml

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.yaml"


def _get_env(*names: str) -> str | None:
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return None


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # env overrides
    cfg["_env"] = {
        "anthropic_token": _get_env("ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY"),
        "anthropic_base_url": _get_env("ANTHROPIC_BASE_URL"),
        "anthropic_model": _get_env("ANTHROPIC_MODEL"),
        "github_token": _get_env("GITHUB_TOKEN") or cfg.get("github", {}).get("token"),
    }
    print(f"Anthropic Base URL: {cfg['_env']['anthropic_base_url']}")
    print(f"Anthropic Model: {cfg['_env']['anthropic_model']}")
    print(f"GitHub Token: {cfg['_env']['github_token']}")
    print(f"Anthropic Token: {'set' if cfg['_env']['anthropic_token'] else 'not set'}")
    if cfg["_env"]["anthropic_model"]:
        cfg.setdefault("llm", {})["model"] = cfg["_env"]["anthropic_model"]
    return cfg


CACHE_DIR = ROOT / "cache"
REPORTS_DIR = ROOT / "reports"
CACHE_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)
