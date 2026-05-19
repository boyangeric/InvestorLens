"""
LangGraph StateGraph definition for InvestorLens.

This wires all nodes together with conditional edges:

  Moderator ─(blocked)──────────────────────────────────────→ END
      │
      │(continue)
      ↓
  Rewriter ─(clarify)──────────────────────────────────────→ END
      │
      │(continue)
      ↓
  Router → Retriever ─(extract_metrics)─→ Metric Extractor → [INTERRUPT] → END
                                  │
                                  │(otherwise)
                                  ↓
                              Grader ─(retry)─────────────────→ Rewriter
                                  │
                                  │(grounded)              (corpus_miss)
                                  ↓                            ↓
                              Generator → Faithfulness ─(ok)─→ END
                                              │
                                              │(unsupported)
                                              ↓
                                          General Generator → END

Two CRAG-style fallback gates protect grounded answers:
  1. Grader → after retries, if relevant_docs < MIN_RELEVANT, route directly
     to `general_generator`.
  2. Faithfulness → after generation, audit each claim against the cited
     chunks. If unsupported (citation hallucination), fall back to
     `general_generator` so the user sees a clearly-labelled general-
     knowledge answer instead of a misleading grounded one.

The Metric Extractor branch is HITL-gated via `interrupt_after`. The node
runs the extraction first, then the graph pauses with the actual numbers in
state, so the human reviewer can verify the figures and their page citations
before they propagate downstream. Approval resumes the graph; rejection
overwrites the generation with a rejection message (handled in the API
layer) and does not resume.

The pause is AFTER the node, not before, because the reviewer is verifying
the EXTRACTED FIGURES — they need to be computed first for the modal to be
useful. Pausing before extraction would only let the reviewer approve an
abstract operation, which adds friction without adding safety.

The `hybrid_live` branch (retriever → live_tools → generator → END) is
deliberately NOT HITL-gated. Live tool outputs (yfinance quotes, Tavily news)
arrive as structured Pydantic models tagged with their source and timestamp,
so the provenance is verifiable by construction rather than by human review.
Live data is also read-only and ephemeral — there is no irreversible decision
downstream of the quote like there is downstream of an extracted figure that
flows into a comparison or recommendation. Different risk profile → different
gating policy.
"""

import logging

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from backend.agent.nodes.adaptive_router import adaptive_router
from backend.agent.nodes.faithfulness import faithfulness
from backend.agent.nodes.finalize_extraction import finalize_extraction
from backend.agent.nodes.general_generator import general_generator
from backend.agent.nodes.generator import generator
from backend.agent.nodes.grader import grader
from backend.agent.nodes.live_tools import live_tools
from backend.agent.nodes.metric_extractor import metric_extractor
from backend.agent.nodes.moderator import moderator
from backend.agent.nodes.retriever import retriever
from backend.agent.nodes.rewriter import rewriter
from backend.agent.state import AgentState

# CRAG threshold — relevant docs needed to use the grounded generator.
# Mirrors the grader's MIN_RELEVANT; below this we fall back to general knowledge.
MIN_RELEVANT_FOR_GROUNDED = 2

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Conditional edge functions
# ---------------------------------------------------------------------------

def should_continue_after_moderator(state: AgentState) -> str:
    """If moderator blocks, go to END. Otherwise continue to rewriter."""
    if state.get("generation"):  # Moderator set a rejection message
        return "blocked"
    return "continue"


def should_continue_after_rewriter(state: AgentState) -> str:
    """
    If the rewriter asked the user for clarification, end the turn so the
    clarification question reaches the UI. Otherwise continue to the router.

    The rewriter signals "clarification requested" by writing the question
    to `generation` (same short-circuit pattern as the moderator). Since the
    moderator's block path bypasses the rewriter entirely, any generation
    visible at this edge must be the rewriter's clarification request.
    """
    if state.get("generation"):
        return "clarify"
    return "continue"


def route_after_retrieval(state: AgentState) -> str:
    """
    After retrieval, branch based on the strategy chosen earlier by the router.

    - extract_metrics → metric_extractor (HITL-gated downstream)
    - hybrid_live     → live_tools (parallel market + news fan-out)
    - everything else → grader (relevance scoring + self-correcting loop)
    """
    strategy = state.get("retrieval_strategy", "semantic_search")
    if strategy == "extract_metrics":
        return "metric_extractor"
    if strategy == "hybrid_live":
        return "live_tools"
    return "grader"


def route_after_generator(state: AgentState) -> str:
    """
    Post-generation routing: do we run the faithfulness audit, or skip to END?

    The audit checks claims against retrieved chunks. For `hybrid_live` answers,
    half the claims come from LIVE tools (yfinance, Tavily) — by definition not
    in any chunk — so the audit would always flag them as unsupported and
    push us into the corpus-miss fallback, throwing away the live data.

    Live tool outputs are already structured (Pydantic models from named APIs),
    so the provenance for the live half is verifiable by construction rather
    than by LLM audit. We trust the structured output and skip the audit.
    """
    if state.get("retrieval_strategy") == "hybrid_live":
        return "skip_audit"
    return "audit"


def route_after_faithfulness(state: AgentState) -> str:
    """
    Post-audit decision: keep the grounded answer or fall back.

    If the audit found unsupported claims (citation hallucination), route to
    `general_generator` so the user gets a clearly-labelled general-knowledge
    answer instead of a misleading grounded one.
    """
    if state.get("faithful", True):
        return "ok"
    return "unsupported"


def should_retry_after_grader(state: AgentState) -> str:
    """
    Three-way CRAG decision after grading:
      - retry        → loop back to rewriter (insufficient relevant docs, retries remain)
      - generate     → grounded generator (enough relevant docs)
      - corpus_miss  → general-knowledge fallback (still insufficient after retries)
    """
    relevant = state.get("relevant_docs", [])
    retry_count = state.get("retry_count", 0)

    if len(relevant) < MIN_RELEVANT_FOR_GROUNDED:
        if retry_count < 2:
            return "retry"
        return "corpus_miss"
    return "generate"


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

def build_graph() -> CompiledStateGraph:
    """Construct and compile the InvestorLens agent graph."""

    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("moderator", moderator)
    graph.add_node("rewriter", rewriter)
    graph.add_node("adaptive_router", adaptive_router)
    graph.add_node("retriever", retriever)
    graph.add_node("grader", grader)
    graph.add_node("generator", generator)
    graph.add_node("faithfulness", faithfulness)
    graph.add_node("general_generator", general_generator)
    graph.add_node("metric_extractor", metric_extractor)
    graph.add_node("finalize_extraction", finalize_extraction)
    graph.add_node("live_tools", live_tools)

    # Set entry point
    graph.set_entry_point("moderator")

    # Moderator → (blocked → END) | (continue → rewriter)
    graph.add_conditional_edges(
        "moderator",
        should_continue_after_moderator,
        {"blocked": END, "continue": "rewriter"},
    )

    # Rewriter → (clarify → END) | (continue → Router)
    # The rewriter ends the turn early when it needs to ask the user a
    # clarification question (vague query + no resolvable history).
    graph.add_conditional_edges(
        "rewriter",
        should_continue_after_rewriter,
        {"clarify": END, "continue": "adaptive_router"},
    )

    # Router → Retriever (always — every strategy needs documents)
    graph.add_edge("adaptive_router", "retriever")

    # Retriever → (metric_extractor) | (live_tools) | (grader)
    graph.add_conditional_edges(
        "retriever",
        route_after_retrieval,
        {
            "metric_extractor": "metric_extractor",
            "live_tools": "live_tools",
            "grader": "grader",
        },
    )

    # Live tools (parallel market + news) → generator, skipping the grader.
    # The node has already promoted retrieved_docs to relevant_docs so the
    # generator sees both the document context and the live data.
    graph.add_edge("live_tools", "generator")

    # Grader → (retry → rewriter) | (generate → generator) | (corpus_miss → general_generator)
    graph.add_conditional_edges(
        "grader",
        should_retry_after_grader,
        {
            "retry": "rewriter",
            "generate": "generator",
            "corpus_miss": "general_generator",
        },
    )

    # Generator → (audit → faithfulness) | (skip_audit → END for hybrid_live)
    # Faithfulness audit → (ok → END) | (unsupported → general_generator)
    graph.add_conditional_edges(
        "generator",
        route_after_generator,
        {"audit": "faithfulness", "skip_audit": END},
    )
    graph.add_conditional_edges(
        "faithfulness",
        route_after_faithfulness,
        {"ok": END, "unsupported": "general_generator"},
    )

    graph.add_edge("general_generator", END)

    # Metric Extractor → finalize_extraction → END.
    # The pause happens BEFORE finalize_extraction so the reviewer can
    # verify, edit, or skip the extracted figures. After they respond, the
    # API writes their verdict (verification_status + possibly edited
    # extracted_metrics) into state and resumes. finalize_extraction then
    # re-renders `generation` from the post-edit state.
    graph.add_edge("metric_extractor", "finalize_extraction")
    graph.add_edge("finalize_extraction", END)

    # Compile with checkpointer + HITL interrupt BEFORE finalize_extraction.
    # - MemorySaver persists state between invocations, enabling pause/resume.
    # - interrupt_before=["finalize_extraction"] pauses after metric_extractor
    #   has written its extracted_metrics + generation to state. snapshot.next
    #   contains ("finalize_extraction",) — a real node, so the API can
    #   unambiguously detect the pause.
    return graph.compile(
        checkpointer=MemorySaver(),
        interrupt_before=["finalize_extraction"],
    )


# Compiled graph instance
app = build_graph()
