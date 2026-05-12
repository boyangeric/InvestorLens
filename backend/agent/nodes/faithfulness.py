"""
Faithfulness Node — post-generation audit for citation hallucination.

The generator can cite a chunk that's about the right topic but doesn't
actually contain the specific fact being claimed. This node compares the
generated answer against the retrieved chunks and labels it as either
`faithful` (all claims supported) or `unsupported` (parametric leak).

When unsupported, the graph routes to `general_generator` instead of
returning the misleading grounded answer to the user.

Pattern reference: RAGAS faithfulness metric / Self-RAG IS_SUP token.
"""

import logging

from backend.agent.schemas import FaithfulnessResponse
from backend.agent.state import AgentState
from backend.agent.utils import call_llm_structured, format_chunks_safely, load_prompt

logger = logging.getLogger(__name__)

PROMPT = load_prompt("faithfulness_v1")


def faithfulness(state: AgentState) -> dict:
    """
    Audit the generator's answer against the chunks it was given.

    Reads: query, generation, relevant_docs
    Writes: faithful (bool), node_trace
    """
    query = state["query"]
    answer = state.get("generation", "")
    relevant_docs = state.get("relevant_docs", [])

    logger.info("Faithfulness: auditing %d-char answer against %d chunks",
                len(answer), len(relevant_docs))

    # Same <chunk>-delimited format the generator saw, so the audit is
    # aligned and the same injection defences apply.
    context = format_chunks_safely(relevant_docs)

    result = call_llm_structured(
        PROMPT,
        {"query": query, "answer": answer, "context": context},
        FaithfulnessResponse,
    )
    response: FaithfulnessResponse = result["response"]

    verdict = response.verdict
    unsupported_claims = response.unsupported_claims
    reasoning = response.reasoning
    is_faithful = verdict == "faithful"

    logger.info("Faithfulness: %s (%d unsupported claims)",
                verdict, len(unsupported_claims))

    trace_entry = {
        "node": "faithfulness",
        "status": verdict,
        "model": result["model"],
        "duration_ms": result["duration_ms"],
        "tokens_in": result["tokens_in"],
        "tokens_out": result["tokens_out"],
        "cost_usd": result["cost_usd"],
        "faithful": is_faithful,
        "unsupported_claims": unsupported_claims,
        "reasoning": reasoning,
    }

    node_trace = state.get("node_trace", []) + [trace_entry]

    return {
        "faithful": is_faithful,
        "current_node": "faithfulness",
        "node_trace": node_trace,
    }
