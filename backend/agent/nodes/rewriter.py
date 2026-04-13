"""
Query Rewriter Node — resolves ambiguous or follow-up queries using chat history.

Uses GPT-4o-mini to rewrite queries like "what about their revenue?" into
"What is Qantas's revenue in HY26?" so downstream nodes don't need chat context.
Only rewrites when necessary — clear, self-contained queries pass through unchanged.
"""

import json
import logging

from backend.agent.state import AgentState
from backend.agent.utils import call_llm, load_prompt

logger = logging.getLogger(__name__)

PROMPT = load_prompt("rewriter_v1")

# Only pass the last 5 messages to keep prompt concise and cheap
MAX_HISTORY = 5


def rewriter(state: AgentState) -> dict:
    """
    Rewrite the query if it's ambiguous or references prior conversation.

    Reads: query, chat_history
    Writes: query (possibly rewritten), original_query (preserved), node_trace
    """
    query = state["query"]
    chat_history = state.get("chat_history", [])

    # Format recent chat history as readable text
    recent_history = chat_history[-MAX_HISTORY:]
    history_text = "\n".join(
        f"{msg['role']}: {msg['content']}" for msg in recent_history
    ) if recent_history else "(no prior messages)"

    logger.info("Rewriter: processing query — %s", query[:80])

    result = call_llm(PROMPT, {
        "query": query,
        "chat_history": history_text,
    })
    response = result["response"]

    rewritten = response.get("rewritten_query", query)
    was_rewritten = response.get("was_rewritten", False)

    trace_entry = {
        "node": "rewriter",
        "status": "rewritten" if was_rewritten else "unchanged",
        "model": result["model"],
        "duration_ms": result["duration_ms"],
        "tokens_in": result["tokens_in"],
        "tokens_out": result["tokens_out"],
    }

    node_trace = state.get("node_trace", []) + [trace_entry]

    if was_rewritten:
        logger.info("Rewriter rewrote: '%s' → '%s'", query[:50], rewritten[:50])
    else:
        logger.info("Rewriter: query unchanged")

    return {
        "query": rewritten,
        "original_query": state.get("original_query", query),
        "current_node": "rewriter",
        "node_trace": node_trace,
    }
