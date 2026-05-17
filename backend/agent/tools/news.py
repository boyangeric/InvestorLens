"""
News search tool — recent financial articles via Tavily.

Tavily is a search API designed for LLM agents: it returns clean text snippets
plus URLs and (where available) publication dates. Free tier is 1000 searches/
month, comfortable for a portfolio demo. Topic="news" biases ranking toward
recent reporting rather than evergreen pages.
"""

import logging
import os

from pydantic import BaseModel
from tavily import TavilyClient

logger = logging.getLogger(__name__)

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

SNIPPET_MAX_CHARS = 500


class NewsResult(BaseModel):
    """One article returned from a news search."""

    title: str
    url: str
    snippet: str
    published_at: str | None = None
    source: str  # domain, e.g. "afr.com"


def _domain_of(url: str) -> str:
    """Best-effort domain extraction without pulling in urllib for one line."""
    if "//" not in url:
        return url
    try:
        return url.split("//", 1)[1].split("/", 1)[0]
    except IndexError:
        return url


def search_news(query: str, days: int = 30, max_results: int = 5) -> list[NewsResult]:
    """
    Search recent financial news for `query` over the last `days` days.

    Returns up to `max_results` articles. Returns [] (not raises) when Tavily
    is unconfigured or the call fails — the agent can still answer from the
    document corpus, and the trace will record `news_count: 0` so the user
    sees the empty result wasn't suppressed.
    """
    if not TAVILY_API_KEY:
        logger.warning("TAVILY_API_KEY not set — search_news returning []")
        return []

    try:
        client = TavilyClient(api_key=TAVILY_API_KEY)
        response = client.search(
            query=query,
            search_depth="basic",
            topic="news",
            days=days,
            max_results=max_results,
        )
    except Exception:
        logger.exception("Tavily search failed for %r", query)
        return []

    results = response.get("results", []) if isinstance(response, dict) else []
    out: list[NewsResult] = []
    for r in results:
        url = r.get("url", "")
        snippet = (r.get("content") or "")[:SNIPPET_MAX_CHARS]
        out.append(
            NewsResult(
                title=r.get("title", ""),
                url=url,
                snippet=snippet,
                published_at=r.get("published_date"),
                source=_domain_of(url),
            )
        )
    return out
