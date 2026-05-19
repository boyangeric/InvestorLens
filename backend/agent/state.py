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
    retrieval_strategy: str         # "semantic_search" | "keyword_search" | "compare" | "extract_metrics" | "hybrid_live"
    strategy_reasoning: str         # Router's explanation for choosing this strategy
    retrieved_docs: list[dict]      # Raw chunks from retriever: [{content, metadata, score}]
    relevant_docs: list[dict]       # Chunks that passed the grader filter

    # --- Generation ---
    generation: str                 # Final generated answer
    confidence: float               # Generator's self-assessed confidence (0.0–1.0); only meaningful when grounded
    grounded: bool                  # True if answer is grounded in retrieved docs; False if from general knowledge
    faithful: bool                  # True if grounded claims are actually supported by cited chunks (post-gen audit)

    # --- Extraction (HITL gate) ---
    extracted_metrics: list[dict]      # Numeric metrics pulled by metric_extractor; surfaced to the human reviewer
                                       # via the approval modal before they propagate downstream
    extracted_highlights: list[str]    # Reviewer-facing notes from metric_extractor (e.g. unusual movements)
    verification_status: str           # "" before HITL | "verified" | "edited" | "skipped"
                                       # Set by the API when the reviewer responds to the modal; finalize_extraction
                                       # re-renders `generation` to reflect any edits, and the frontend renders a
                                       # provenance badge based on this value.

    # --- Live tool calling (hybrid_live strategy) ---
    live_quotes: list[dict]            # MarketQuote dicts fetched from yfinance via the live_tools node
    live_news:   list[dict]            # NewsResult dicts fetched from Tavily via the live_tools node

    # --- Control flow ---
    retry_count: int                # Tracks grader→rewriter loop iterations (max 2)

    # --- Observability ---
    current_node: str               # Name of the currently executing node
    node_trace: list[dict]          # Execution trace: [{node, status, model, duration_ms, tokens_in, tokens_out}]
    token_usage: dict               # Aggregated per-node token counts
