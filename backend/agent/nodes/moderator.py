"""
Moderator Node — first node in the graph. Checks query safety and relevance.

Uses GPT-4o-mini to classify queries as "pass" (continue to rewriter) or
"block" (return rejection message and end). This prevents off-topic queries
and prompt injection from consuming retrieval/generation resources.
"""

import logging

from backend.agent.schemas import ModeratorResponse
from backend.agent.state import AgentState
from backend.agent.utils import call_llm_structured, load_prompt

logger = logging.getLogger(__name__)

PROMPT = load_prompt("moderator_v1")


def moderator(state: AgentState) -> dict:
    """
    Moderate the user's query for safety and relevance.

    Returns:
        - If passed: updates current_node and node_trace
        - If blocked: sets generation to rejection message (triggers END via conditional edge)
    """
    query = state["query"]
    logger.info("Moderator: checking query — %s", query[:80])

    result = call_llm_structured(PROMPT, {"query": query}, ModeratorResponse)
    response: ModeratorResponse = result["response"]

    decision = response.decision
    reason = response.reason

    # Build trace entry for observability
    trace_entry = {
        "node": "moderator",
        "status": decision,
        "model": result["model"],
        "duration_ms": result["duration_ms"],
        "tokens_in": result["tokens_in"],
        "tokens_out": result["tokens_out"],
        "cost_usd": result["cost_usd"],
    }

    node_trace = state.get("node_trace", []) + [trace_entry]

    if decision == "block":
        logger.warning("Moderator blocked query: %s", reason)
        return {
            "current_node": "moderator",
            "generation": f"I can't help with that. {reason}",
            "node_trace": node_trace,
        }

    logger.info("Moderator passed query")
    return {
        "current_node": "moderator",
        "node_trace": node_trace,
    }
