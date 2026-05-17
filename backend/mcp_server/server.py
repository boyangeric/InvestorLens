"""
MCP server for InvestorLens — exposes the agent's capabilities as MCP tools.

This lets any MCP client (Claude Desktop, Cursor, custom CLIs) use InvestorLens
as a tool source. Each tool wraps a piece of the agent so callers can invoke
specific capabilities directly without going through the full graph.

Tools exposed:
  - search_docs:        semantic search over the document corpus
  - extract_financials: pull structured numeric metrics from a document
                        (the same extraction the agent's HITL gate sits on top of)
  - compare_docs:       cross-document comparison
  - get_market_quote:   live price snapshot via yfinance (no API key)
  - search_news:        recent financial news via Tavily (needs TAVILY_API_KEY)

Run via stdio (the standard transport for Claude Desktop integration):

    python -m backend.mcp_server.server

Add to Claude Desktop's `claude_desktop_config.json`:

    {
      "mcpServers": {
        "investorlens": {
          "command": "python",
          "args": ["-m", "backend.mcp_server.server"],
          "cwd": "/Users/ericli/Desktop/InvestorLens"
        }
      }
    }
"""

import logging

from mcp.server.fastmcp import FastMCP

from backend.agent.nodes.retriever import _compare, _direct_extract, _semantic_search
from backend.agent.schemas import MetricExtractorResponse
from backend.agent.tools.market import get_market_quote as _get_market_quote
from backend.agent.tools.news import search_news as _search_news
from backend.agent.utils import call_llm_structured, load_prompt

logger = logging.getLogger(__name__)

mcp = FastMCP("investorlens")


@mcp.tool()
def search_docs(query: str, top_k: int = 8) -> dict:
    """
    Semantic search over ingested financial documents.

    Use this when you need to find passages relevant to a conceptual question,
    e.g. "what did the company say about its outlook?" or "key strategic priorities".

    Args:
        query: Natural language question or topic to search for.
        top_k: Number of chunks to return (default 8).

    Returns:
        A dict with the top matching chunks (text, source, page, score).
    """
    logger.info("MCP search_docs: %s", query[:80])
    chunks = _semantic_search(query, top_k=top_k)
    return {
        "query": query,
        "results": [
            {
                "content": c["content"],
                "source": c["source"],
                "page": c["page"],
                "score": round(c["score"], 4),
            }
            for c in chunks
        ],
        "count": len(chunks),
    }


@mcp.tool()
def extract_financials(query: str, top_k: int = 20) -> dict:
    """
    Extract structured numeric financial metrics from ingested documents.

    Use this when the caller wants specific figures (revenue, profit, margins,
    EPS, ratios, etc.) extracted as data, not summarised as prose. Returns a
    list of named metrics with exact figures, reporting periods, and source
    pages — the same shape the agent's HITL gate verifies.

    Note: in the full agent (via WebSocket), this extraction is HITL-gated.
    When invoked directly via MCP, it runs without the human review gate —
    the assumption is that an MCP client (e.g. Claude Desktop) is already a
    reviewed, intentional invocation.

    Args:
        query: What to extract, e.g. "all financial metrics for the half year"
               or "balance sheet items".
        top_k: Number of chunks to read for extraction (default 20 — extraction
               benefits from broader coverage).

    Returns:
        A dict with `metrics` (list of {name, value, period, source, page}),
        `document`, `period`, and `key_highlights`.
    """
    logger.info("MCP extract_financials: %s", query[:80])
    chunks = _direct_extract(query, top_k=top_k)

    if not chunks:
        return {"query": query, "metrics": [], "note": "No relevant chunks found."}

    source = chunks[0].get("source", "unknown")
    content = "\n\n".join(f"[Page {c['page']}]\n{c['content']}" for c in chunks)

    prompt = load_prompt("extractor_v1")
    result = call_llm_structured(
        prompt,
        {"source": source, "content": content},
        MetricExtractorResponse,
    )
    response: MetricExtractorResponse = result["response"]

    return {
        "query": query,
        "document": response.document,
        "period": response.period,
        "metrics": [m.model_dump() for m in response.metrics],
        "key_highlights": response.key_highlights,
        "tokens_in": result["tokens_in"],
        "tokens_out": result["tokens_out"],
        "cost_usd": result["cost_usd"],
    }


@mcp.tool()
def compare_docs(query: str, top_k: int = 6) -> dict:
    """
    Cross-document comparison — retrieves chunks from multiple sources for
    side-by-side analysis.

    Use this when the caller wants to compare two or more documents (e.g.,
    "compare risks between Fund A and Fund B"). Caps chunks per source to
    ensure diversity, so the caller actually gets material from each document.

    Args:
        query: The comparison prompt.
        top_k: Total chunks to return across documents (default 6).

    Returns:
        A dict grouping retrieved chunks by source document.
    """
    logger.info("MCP compare_docs: %s", query[:80])
    chunks = _compare(query, top_k=top_k)

    grouped: dict[str, list[dict]] = {}
    for c in chunks:
        grouped.setdefault(c["source"], []).append({
            "content": c["content"],
            "page": c["page"],
            "score": round(c["score"], 4),
        })

    return {
        "query": query,
        "by_source": grouped,
        "source_count": len(grouped),
        "total_chunks": len(chunks),
    }


@mcp.tool()
def get_market_quote(ticker: str) -> dict:
    """
    Fetch a current live quote (price, day change, market cap, 52-week range)
    for a single security via yfinance.

    Use the .AX suffix for ASX-listed Australian companies — `QAN.AX` for
    Qantas, `CBA.AX` for Commonwealth Bank, `BHP.AX` for BHP, `WBC.AX` for
    Westpac. US tickers use no suffix (`AAPL`, `MSFT`).

    Args:
        ticker: Ticker symbol with exchange suffix.

    Returns:
        A dict with the quote (ticker, name, price, currency, day change %,
        market cap, 52w low/high, as-of timestamp) or `{"error": "..."}` if the
        ticker could not be resolved.
    """
    logger.info("MCP get_market_quote: %s", ticker)
    quote = _get_market_quote(ticker)
    if quote is None:
        return {"error": f"No live data for ticker {ticker!r}"}
    return quote.model_dump()


@mcp.tool()
def search_news(query: str, days: int = 30, max_results: int = 5) -> dict:
    """
    Search recent financial news via Tavily. Returns title, URL, snippet, and
    publication date for the top matching articles.

    Use this when the caller wants context on what has happened recently — ASX
    announcements, broker upgrades, profit warnings, regulatory actions — that
    a static document corpus cannot answer.

    Requires `TAVILY_API_KEY` in the environment. Returns an empty list (with a
    `note`) rather than raising when the key is missing or the upstream call
    fails, so callers can degrade gracefully.

    Args:
        query: Natural-language news query, e.g. "Qantas profit downgrade".
        days: Lookback window in days (default 30).
        max_results: Number of articles to return (default 5).

    Returns:
        A dict with `articles` (list of {title, url, snippet, published_at,
        source}) and `count`.
    """
    logger.info("MCP search_news: %s (days=%d)", query[:80], days)
    articles = _search_news(query, days=days, max_results=max_results)
    out = {
        "query": query,
        "articles": [a.model_dump() for a in articles],
        "count": len(articles),
    }
    if not articles:
        out["note"] = (
            "No articles returned — either TAVILY_API_KEY is unset or the "
            "search produced no recent matches."
        )
    return out


def main() -> None:
    """Entry point for stdio transport (used by Claude Desktop)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger.info("Starting InvestorLens MCP server (stdio)")
    mcp.run()


if __name__ == "__main__":
    main()
