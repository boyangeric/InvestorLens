"""
AgentState — the shared contract between all LangGraph nodes.

Every node reads from and writes to this TypedDict. Fields are grouped by
purpose: query handling, retrieval, generation, and observability.
"""

from typing import TypedDict


class AgentState(TypedDict):
    # --- Query handling ---
    query: str                      # Current (possibly rewritten) query
    original_query: str             # User's original query, preserved for reference
    chat_history: list[dict]        # Last N messages for context: [{role, content}]

    # --- Retrieval ---
    retrieval_strategy: str         # "semantic_search" | "keyword_search" | "direct_extract" | "compare"
    strategy_reasoning: str         # Router's explanation for choosing this strategy
    retrieved_docs: list[dict]      # Raw chunks from retriever: [{content, metadata, score}]
    relevant_docs: list[dict]       # Chunks that passed the grader filter

    # --- Generation ---
    generation: str                 # Final generated answer
    confidence: float               # Generator's self-assessed confidence (0.0–1.0)

    # --- Control flow ---
    retry_count: int                # Tracks grader→rewriter loop iterations (max 2)

    # --- Observability ---
    current_node: str               # Name of the currently executing node
    node_trace: list[dict]          # Execution trace: [{node, status, model, duration_ms, tokens_in, tokens_out}]
    token_usage: dict               # Aggregated per-node token counts
