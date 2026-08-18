"""GitHub inference-stack watcher.

For each configured repo, pull:
  - releases (last N days)
  - merged PRs (last N days)
  - opened issues (last N days)

Uses REST API. If GITHUB_TOKEN present -> auth'd (5000/h), else 60/h.
"""
from __future__ import annotations
import logging
from datetime import datetime, timedelta, timezone
from .base import BaseCollector
from ..utils.http import get

log = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"


class GitHubReposCollector(BaseCollector):
    name = "github"

    def extra_headers(self):
        h = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
        tok = self.cfg["_env"].get("github_token")
        if tok:
            h["Authorization"] = f"Bearer {tok}"
        return h

    def collect(self) -> list[dict]:
        gh_cfg = self.cfg.get("github", {})
        repos = gh_cfg.get("repos", [])
        lookback = self.cfg.get("lookback_hours", 36)
        # For releases we allow a wider window (2 weeks) since releases are rarer
        release_since = datetime.now(timezone.utc) - timedelta(days=14)
        activity_since = datetime.now(timezone.utc) - timedelta(hours=lookback)

        items: list[dict] = []
        for repo in repos:
            try:
                items.extend(self._releases(repo, release_since))
            except Exception as e:  # noqa: BLE001
                log.warning("[github %s] releases failed: %s", repo, e)
            try:
                items.extend(self._pulls(repo, activity_since))
            except Exception as e:  # noqa: BLE001
                log.warning("[github %s] pulls failed: %s", repo, e)
            try:
                items.extend(self._issues(repo, activity_since))
            except Exception as e:  # noqa: BLE001
                log.warning("[github %s] issues failed: %s", repo, e)
        return items

    def _releases(self, repo: str, since: datetime) -> list[dict]:
        r = get(self.client, f"{GITHUB_API}/repos/{repo}/releases", params={"per_page": 10})
        out = []
        for rel in r.json():
            published = rel.get("published_at") or rel.get("created_at")
            if not published:
                continue
            dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
            if dt < since:
                continue
            body = (rel.get("body") or "")[:600]
            out.append({
                "source": "github_release",
                "title": f"{repo}  {rel.get('name') or rel.get('tag_name')}",
                "url": rel.get("html_url"),
                "published": published,
                "summary": body,
                "meta": {"repo": repo, "tag": rel.get("tag_name"), "prerelease": rel.get("prerelease")},
            })
        return out

    def _pulls(self, repo: str, since: datetime) -> list[dict]:
        # Search API: merged PRs since <date>
        q = f"repo:{repo} is:pr is:merged merged:>={since.date().isoformat()}"
        r = get(self.client, f"{GITHUB_API}/search/issues",
                params={"q": q, "sort": "updated", "order": "desc", "per_page": 20})
        out = []
        for pr in r.json().get("items", []):
            out.append({
                "source": "github_pr",
                "title": f"{repo}#{pr['number']}  {pr['title']}",
                "url": pr["html_url"],
                "published": pr.get("closed_at") or pr.get("updated_at"),
                "summary": (pr.get("body") or "")[:400],
                "meta": {
                    "repo": repo, "number": pr["number"],
                    "user": pr.get("user", {}).get("login"),
                    "labels": [l.get("name") for l in pr.get("labels", [])],
                    "comments": pr.get("comments", 0),
                },
            })
        return out

    def _issues(self, repo: str, since: datetime) -> list[dict]:
        # Issues opened in window, sort by reactions to surface important ones
        q = f"repo:{repo} is:issue created:>={since.date().isoformat()}"
        r = get(self.client, f"{GITHUB_API}/search/issues",
                params={"q": q, "sort": "reactions", "order": "desc", "per_page": 10})
        out = []
        for iss in r.json().get("items", []):
            out.append({
                "source": "github_issue",
                "title": f"{repo}#{iss['number']}  {iss['title']}",
                "url": iss["html_url"],
                "published": iss.get("created_at"),
                "summary": (iss.get("body") or "")[:400],
                "meta": {
                    "repo": repo, "number": iss["number"],
                    "user": iss.get("user", {}).get("login"),
                    "reactions": iss.get("reactions", {}).get("total_count", 0),
                    "comments": iss.get("comments", 0),
                    "state": iss.get("state"),
                },
            })
        return out
