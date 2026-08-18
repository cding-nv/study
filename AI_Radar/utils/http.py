"""Shared HTTP client with proxy + retry."""
from __future__ import annotations
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36 AI-Radar/1.0"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def make_client(proxy: str | None, timeout: int = 30, extra_headers: dict | None = None) -> httpx.Client:
    headers = {**DEFAULT_HEADERS, **(extra_headers or {})}
    kwargs: dict = {"headers": headers, "timeout": timeout, "follow_redirects": True}
    if proxy:
        kwargs["proxy"] = proxy
    return httpx.Client(**kwargs)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    reraise=True,
)
def get(client: httpx.Client, url: str, **kw) -> httpx.Response:
    r = client.get(url, **kw)
    r.raise_for_status()
    return r
