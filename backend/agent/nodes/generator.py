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

from backend.agent.schemas import GeneratorResponse
from backend.agent.state import AgentState
from backend.agent.utils import call_llm_structured, format_chunks_safely, load_prompt

logger = logging.getLogger(__name__)

PROMPT = load_prompt("generator_v1")


def generator(state: AgentState) -> dict:
    """
    Generate the final answer from relevant retrieved chunks.

    Reads: query, relevant_docs
    Writes: generation, confidence, node_trace
    """
    query = state["query"]
    relevant_docs = state.get("relevant_docs", [])

    logger.info("Generator: synthesising answer from %d chunks", len(relevant_docs))

    # Chunks are wrapped in <chunk> envelopes so the LLM can distinguish
    # untrusted document data from system instructions — see utils.py.
    context = format_chunks_safely(relevant_docs)

    result = call_llm_structured(
        PROMPT,
        {"query": query, "context": context},
        GeneratorResponse,
    )

    response: GeneratorResponse = result["response"]
    answer = response.answer
    # Clamp into [0, 1] — Structured Outputs enforces the float type but
    # doesn't enforce numeric bounds, so the model could still return 1.2.
    confidence = max(0.0, min(1.0, response.confidence))
    reasoning = response.reasoning
    sources = response.sources

    logger.info("Generator: grounded, confidence=%.2f", confidence)

    trace_entry = {
        "node": "generator",
        "status": f"confidence={confidence:.2f}",
        "model": result["model"],
        "duration_ms": result["duration_ms"],
        "tokens_in": result["tokens_in"],
        "tokens_out": result["tokens_out"],
        "cost_usd": result["cost_usd"],
        "grounded": True,
        "confidence": confidence,
        "reasoning": reasoning,
        "sources": sources,
    }

    node_trace = state.get("node_trace", []) + [trace_entry]

    return {
        "generation": answer,
        "confidence": confidence,
        "grounded": True,
        "current_node": "generator",
        "node_trace": node_trace,
    }
