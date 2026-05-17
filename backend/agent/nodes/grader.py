"""
Relevance Grader Node — quality gate between retrieval and generation.

Uses GPT-4o-mini to score each retrieved chunk as relevant or not.
If too few chunks pass, triggers a rewrite-and-retry loop:
  grader → rewriter → retriever → grader (max 2 retries)

This self-correcting cycle is a key LangGraph pattern — the agent doesn't
just fail silently on poor retrieval, it actively tries to fix it.
"""

import logging

from backend.agent.schemas import GraderResponse
from backend.agent.state import AgentState
from backend.agent.utils import call_llm_structured, load_prompt

logger = logging.getLogger(__name__)

PROMPT = load_prompt("grader_v1")

# Minimum number of relevant chunks needed to proceed to generation
MIN_RELEVANT = 2

# Maximum retry attempts before giving up and generating with what we have
MAX_RETRIES = 2


def grader(state: AgentState) -> dict:
    """
    Grade each retrieved chunk for relevance to the query.

    Reads: query, retrieved_docs, retry_count
    Writes: relevant_docs, retry_count, node_trace
    """
    query = state["query"]
    retrieved_docs = state.get("retrieved_docs", [])
    retry_count = state.get("retry_count", 0)

    logger.info("Grader: scoring %d chunks (attempt %d)", len(retrieved_docs), retry_count + 1)

    relevant_docs = []
    uncertain_count = 0
    total_tokens_in = 0
    total_tokens_out = 0
    total_duration_ms = 0
    total_cost_usd = 0.0

    for doc in retrieved_docs:
        result = call_llm_structured(
            PROMPT,
            {
                "query": query,
                "chunk_text": doc["content"],
                "source": doc.get("source", "unknown"),
                "page": doc.get("page", 0),
            },
            GraderResponse,
        )

        total_tokens_in += result["tokens_in"]
        total_tokens_out += result["tokens_out"]
        total_duration_ms += result["duration_ms"]
        total_cost_usd += result["cost_usd"]

        response: GraderResponse = result["response"]
        if response.relevant == "yes":
            relevant_docs.append(doc)
        elif response.relevant == "uncertain":
            # Don't ground on an uncertain chunk — citation-hallucination risk.
            # But surface the count in the trace so the signal doesn't vanish.
            uncertain_count += 1

    # Decide whether we have enough relevant chunks
    enough = len(relevant_docs) >= MIN_RELEVANT
    needs_retry = not enough and retry_count < MAX_RETRIES

    if needs_retry:
        status = f"insufficient ({len(relevant_docs)}/{MIN_RELEVANT}), retry {retry_count + 1}"
    else:
        status = f"{len(relevant_docs)}/{len(retrieved_docs)} relevant"
    if uncertain_count:
        status += f" ({uncertain_count} uncertain)"

    logger.info("Grader: %s", status)

    trace_entry = {
        "node": "grader",
        "status": status,
        "model": result["model"],
        "duration_ms": total_duration_ms,
        "tokens_in": total_tokens_in,
        "tokens_out": total_tokens_out,
        "cost_usd": total_cost_usd,
        "chunks_graded": len(retrieved_docs),
        "chunks_relevant": len(relevant_docs),
        "chunks_uncertain": uncertain_count,
    }

    node_trace = state.get("node_trace", []) + [trace_entry]

    return {
        "relevant_docs": relevant_docs,
        "retry_count": retry_count + 1 if needs_retry else retry_count,
        "current_node": "grader",
        "node_trace": node_trace,
    }
