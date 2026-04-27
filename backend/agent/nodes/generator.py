"""
Generator Node — produces the final answer with citations and self-assessed confidence.

Uses GPT-4o (the bigger model) because this is where quality matters most.
The LLM is instructed to:
  1. Answer only from the provided context (no outside knowledge)
  2. Cite every claim with [Source: filename, Page X]
  3. Assess its own confidence in the answer (0.0-1.0)

The confidence score becomes the signal the risk flagger uses to decide
whether human review is needed.
"""

import logging

from backend.agent.state import AgentState
from backend.agent.utils import call_llm, load_prompt

logger = logging.getLogger(__name__)

PROMPT = load_prompt("generator_v1")


def _format_context(docs: list[dict]) -> str:
    """
    Format retrieved chunks into a numbered context string for the LLM.

    Each chunk is labelled with source and page so the LLM can cite it.
    """
    if not docs:
        return "No relevant context available."

    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.get("source", "unknown")
        page = doc.get("page", 0)
        content = doc.get("content", "")
        parts.append(f"[{i}] Source: {source}, Page: {page}\n{content}")

    return "\n\n---\n\n".join(parts)


def generator(state: AgentState) -> dict:
    """
    Generate the final answer from relevant retrieved chunks.

    Reads: query, relevant_docs
    Writes: generation, confidence, node_trace
    """
    query = state["query"]
    relevant_docs = state.get("relevant_docs", [])

    logger.info("Generator: synthesising answer from %d chunks", len(relevant_docs))

    context = _format_context(relevant_docs)

    result = call_llm(PROMPT, {
        "query": query,
        "context": context,
    })

    response = result["response"]
    answer = response.get("answer", "I could not generate an answer.")
    confidence = float(response.get("confidence", 0.0))
    reasoning = response.get("reasoning", "")
    sources = response.get("sources", [])

    logger.info("Generator: confidence=%.2f", confidence)

    trace_entry = {
        "node": "generator",
        "status": f"confidence={confidence:.2f}",
        "model": result["model"],
        "duration_ms": result["duration_ms"],
        "tokens_in": result["tokens_in"],
        "tokens_out": result["tokens_out"],
        "confidence": confidence,
        "reasoning": reasoning,
        "sources": sources,
    }

    node_trace = state.get("node_trace", []) + [trace_entry]

    return {
        "generation": answer,
        "confidence": confidence,
        "current_node": "generator",
        "node_trace": node_trace,
    }
