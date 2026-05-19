"""
Query Rewriter Node — resolves ambiguous or follow-up queries using chat history.

Uses GPT-4o-mini to do one of three things:
  1. Pass through clear, self-contained queries unchanged.
  2. Rewrite vague queries when chat history supplies the missing subject
     (e.g. "what about their revenue?" → "What is Qantas's revenue?").
  3. Ask the user a clarification question when the query is genuinely
     vague AND no usable history is available. The rewriter ends the turn
     by writing the clarification question to `generation` — the same
     short-circuit pattern the moderator uses for blocked queries.

Wrong-entity queries ("Tesla's revenue" when only Qantas is uploaded)
are intentionally treated as clear and passed through. The downstream
CRAG loop detects the corpus miss and the general_generator answers from
general knowledge with a disclaimer — that's a better UX than asking the
user "which document?" when they've already named one we don't have.
"""

import logging

from backend.agent.schemas import RewriterResponse
from backend.agent.state import AgentState
from backend.agent.utils import call_llm_structured, load_prompt

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

    result = call_llm_structured(
        PROMPT,
        {"query": query, "chat_history": history_text},
        RewriterResponse,
    )
    response: RewriterResponse = result["response"]

    rewritten = response.rewritten_query
    was_rewritten = response.was_rewritten
    needs_clarification = response.needs_clarification
    clarification_question = response.clarification_question

    base_trace = {
        "node": "rewriter",
        "model": result["model"],
        "duration_ms": result["duration_ms"],
        "tokens_in": result["tokens_in"],
        "tokens_out": result["tokens_out"],
        "cost_usd": result["cost_usd"],
    }

    # Outcome 3 — vague + no resolvable history → ask the user and end the turn.
    # The conditional edge after the rewriter inspects `generation`; if set,
    # it routes to END (same pattern as the moderator's block path).
    if needs_clarification and clarification_question:
        logger.info("Rewriter: clarification requested — %s", clarification_question[:80])
        trace_entry = {**base_trace, "status": "clarification_requested"}
        node_trace = state.get("node_trace", []) + [trace_entry]
        return {
            "generation": clarification_question,
            "grounded": False,
            "original_query": state.get("original_query", query),
            "current_node": "rewriter",
            "node_trace": node_trace,
        }

    # Outcomes 1 & 2 — clear query or successful rewrite.
    trace_entry = {**base_trace, "status": "rewritten" if was_rewritten else "unchanged"}
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
