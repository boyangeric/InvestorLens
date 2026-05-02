"""
MCP server for InvestorLens — exposes the agent's capabilities as MCP tools.

This lets any MCP client (Claude Desktop, Cursor, custom CLIs) use InvestorLens
as a tool source. Each tool wraps a piece of the agent so callers can invoke
specific capabilities directly without going through the full graph.

Tools exposed:
  - search_docs:           semantic search over the document corpus
  - extract_financials:    pull structured financial metrics from a document
  - compare_docs:          cross-document comparison
  - analyze_disclosures:   extract disclosed material risks (was "flag_risks")

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
from typing import cast

from mcp.server.fastmcp import FastMCP

from backend.agent.nodes.disclosure_analyzer import disclosure_analyzer
from backend.agent.nodes.retriever import _compare, _direct_extract, _semantic_search
from backend.agent.state import AgentState
from backend.agent.utils import call_llm, load_prompt

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
    Extract structured financial metrics from ingested documents.

    Use this when the caller wants specific figures (revenue, profit, margins,
    EPS, etc.) extracted as data, not summarised as prose. Returns a JSON
    object with named metrics and their source pages.

    Args:
        query: What to extract, e.g. "all financial metrics for the half year"
               or "balance sheet items".
        top_k: Number of chunks to read for extraction (default 20 — extraction
               benefits from broader coverage).

    Returns:
        A dict with `metrics` (list of {name, value, unit, source, page}) and
        the raw extractor reasoning.
    """
    logger.info("MCP extract_financials: %s", query[:80])
    chunks = _direct_extract(query, top_k=top_k)

    if not chunks:
        return {"query": query, "metrics": [], "note": "No relevant chunks found."}

    context = "\n\n---\n\n".join(
        f"[{c['source']}, Page {c['page']}]\n{c['content']}" for c in chunks
    )

    prompt = load_prompt("extractor_v1")
    result = call_llm(prompt, {"query": query, "context": context})
    response = result["response"]

    return {
        "query": query,
        "metrics": response.get("metrics", []),
        "reasoning": response.get("reasoning", ""),
        "tokens_in": result["tokens_in"],
        "tokens_out": result["tokens_out"],
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
def analyze_disclosures(query: str = "key disclosed risks", top_k: int = 8) -> dict:
    """
    Extract material risks the company has disclosed in its filings.

    Returns a structured list of disclosed risks with severity (high/medium/low),
    a plain-language summary, and the exact passage from the document. Use this
    when the caller asks about disclosed risks, warnings, regulatory issues,
    going concern, or material uncertainties.

    Note: in the full agent (via WebSocket), this tool is HITL-gated. When
    invoked directly via MCP, it runs without the human review gate — the
    assumption is that an MCP client (e.g., Claude Desktop) is already a
    reviewed, intentional invocation.

    Args:
        query: Topic or scope for the disclosure analysis.
        top_k: Number of chunks to feed into the analyser (default 8).

    Returns:
        Structured disclosures with severity and source citations.
    """
    logger.info("MCP analyze_disclosures: %s", query[:80])
    chunks = _semantic_search(query, top_k=top_k)

    # Build a minimal AgentState-like dict — the node only reads two keys.
    fake_state = cast(AgentState, {
        "query": query,
        "relevant_docs": chunks,
        "node_trace": [],
    })
    result = disclosure_analyzer(fake_state)

    return {
        "query": query,
        "report": result.get("generation", ""),
        "confidence": result.get("confidence", 0.0),
        "trace": result.get("node_trace", []),
    }


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
