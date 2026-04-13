"""
Adaptive Router Node — classifies queries into retrieval strategies.

Uses GPT-4o-mini to pick the optimal retrieval approach per query:
  - semantic_search: conceptual questions ("what are the key risks?")
  - keyword_search: specific lookups ("what is the MER?")
  - direct_extract: structured extraction ("extract all financial metrics")
  - compare: multi-document comparison ("compare these two funds")

This is what makes the RAG system "adaptive" — different queries need
fundamentally different retrieval approaches.
"""

import logging

from backend.agent.state import AgentState
from backend.agent.utils import call_llm, load_prompt

logger = logging.getLogger(__name__)

PROMPT = load_prompt("adaptive_router_v1")

VALID_STRATEGIES = {"semantic_search", "keyword_search", "direct_extract", "compare"}


def router(state: AgentState) -> dict:
    """
    Classify the query into a retrieval strategy.

    Reads: query
    Writes: retrieval_strategy, strategy_reasoning, node_trace
    """
    query = state["query"]
    logger.info("Router: classifying query — %s", query[:80])

    result = call_llm(PROMPT, {"query": query})
    response = result["response"]

    strategy = response.get("strategy", "semantic_search")
    reasoning = response.get("reasoning", "")

    # Fallback to semantic_search if LLM returns an invalid strategy
    if strategy not in VALID_STRATEGIES:
        logger.warning("Router returned invalid strategy '%s', falling back to semantic_search", strategy)
        strategy = "semantic_search"

    trace_entry = {
        "node": "router",
        "status": strategy,
        "model": result["model"],
        "duration_ms": result["duration_ms"],
        "tokens_in": result["tokens_in"],
        "tokens_out": result["tokens_out"],
    }

    node_trace = state.get("node_trace", []) + [trace_entry]

    logger.info("Router: selected '%s' — %s", strategy, reasoning)

    return {
        "retrieval_strategy": strategy,
        "strategy_reasoning": reasoning,
        "current_node": "router",
        "node_trace": node_trace,
    }
