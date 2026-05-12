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

from backend.agent.schemas import RouterResponse
from backend.agent.state import AgentState
from backend.agent.utils import call_llm_structured, load_prompt

logger = logging.getLogger(__name__)

PROMPT = load_prompt("adaptive_router_v1")


def adaptive_router(state: AgentState) -> dict:
    """
    Classify the query into a retrieval strategy.

    Reads: query
    Writes: retrieval_strategy, strategy_reasoning, node_trace
    """
    query = state["query"]
    logger.info("Router: classifying query — %s", query[:80])

    # Structured Outputs constrains `strategy` to the Literal enum in
    # RouterResponse, so the previous fallback-to-semantic_search guard is
    # no longer reachable — an invalid value cannot leave the API.
    result = call_llm_structured(PROMPT, {"query": query}, RouterResponse)
    response: RouterResponse = result["response"]

    strategy = response.strategy
    reasoning = response.reasoning

    trace_entry = {
        "node": "router",
        "status": strategy,
        "model": result["model"],
        "duration_ms": result["duration_ms"],
        "tokens_in": result["tokens_in"],
        "tokens_out": result["tokens_out"],
        "cost_usd": result["cost_usd"],
    }

    node_trace = state.get("node_trace", []) + [trace_entry]

    logger.info("Router: selected '%s' — %s", strategy, reasoning)

    return {
        "retrieval_strategy": strategy,
        "strategy_reasoning": reasoning,
        "current_node": "router",
        "node_trace": node_trace,
    }
