"""
General Generator Node — CRAG fallback when retrieval can't answer.

Reached only when the grader produces fewer than MIN_RELEVANT chunks AFTER
the rewrite-retry loop is exhausted. Answers from the LLM's parametric
knowledge with no document citations and `grounded=False`, so the UI can
clearly mark the answer as not corpus-grounded.

Pattern reference: Corrective RAG (CRAG) — Yan et al., 2024.
"""

import logging

from backend.agent.schemas import GeneralGeneratorResponse
from backend.agent.state import AgentState
from backend.agent.utils import call_llm_structured, load_prompt

logger = logging.getLogger(__name__)

PROMPT = load_prompt("general_generator_v1")


def general_generator(state: AgentState) -> dict:
    """
    Answer from general knowledge when the corpus has no relevant passages.

    Reads: query
    Writes: generation, confidence, grounded=False, node_trace
    """
    query = state["query"]
    logger.info("General generator: corpus miss, answering from general knowledge")

    result = call_llm_structured(PROMPT, {"query": query}, GeneralGeneratorResponse)
    response: GeneralGeneratorResponse = result["response"]

    answer = response.answer
    reasoning = response.reasoning

    trace_entry = {
        "node": "general_generator",
        "status": "general knowledge",
        "model": result["model"],
        "duration_ms": result["duration_ms"],
        "tokens_in": result["tokens_in"],
        "tokens_out": result["tokens_out"],
        "cost_usd": result["cost_usd"],
        "grounded": False,
        "reasoning": reasoning,
        "sources": [],
    }

    node_trace = state.get("node_trace", []) + [trace_entry]

    return {
        "generation": answer,
        "confidence": 0.0,
        "grounded": False,
        "current_node": "general_generator",
        "node_trace": node_trace,
    }
