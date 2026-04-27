"""
LangGraph StateGraph definition for InvestorLens.

This wires all nodes together with conditional edges:

  Moderator ─(blocked)──────────────────────────────────────→ END
      │
      │(continue)
      ↓
  Rewriter → Router → Retriever ─(analyze_disclosures)─→ [INTERRUPT] Disclosure Analyzer → END
                                  │
                                  │(otherwise)
                                  ↓
                              Grader ─(retry)──→ Rewriter
                                  │
                                  │(generate)
                                  ↓
                              Generator → END

The Grader → Rewriter cycle runs up to 2 retries before falling back to
generation. The Disclosure Analyzer branch pauses before execution so a
human can review the query and the retrieved source material (HITL via
`interrupt_before`).

The retriever runs for ALL strategies — including `analyze_disclosures` —
so that disclosure analysis has actual document chunks to work with, and
so the human reviewer can see what would be analysed before approving.

"Disclosed risks" here means material business risks the company has disclosed
in its own filings — NOT risky user queries or AI safety risk. The moderator
handles query-level safety.
"""

import logging

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from backend.agent.nodes.adaptive_router import adaptive_router
from backend.agent.nodes.disclosure_analyzer import disclosure_analyzer
from backend.agent.nodes.generator import generator
from backend.agent.nodes.grader import grader
from backend.agent.nodes.moderator import moderator
from backend.agent.nodes.retriever import retriever
from backend.agent.nodes.rewriter import rewriter
from backend.agent.state import AgentState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Conditional edge functions
# ---------------------------------------------------------------------------

def should_continue_after_moderator(state: AgentState) -> str:
    """If moderator blocks, go to END. Otherwise continue to rewriter."""
    if state.get("generation"):  # Moderator set a rejection message
        return "blocked"
    return "continue"


def route_after_retrieval(state: AgentState) -> str:
    """
    After retrieval, branch based on the strategy chosen earlier by the router.

    Disclosure queries skip the grader and go straight to disclosure_analyzer
    (which is then HITL-gated). All other strategies go through the grader for
    relevance scoring and the self-correcting loop.
    """
    strategy = state.get("retrieval_strategy", "semantic_search")
    if strategy == "analyze_disclosures":
        return "disclosure_analyzer"
    return "grader"


def should_retry_after_grader(state: AgentState) -> str:
    """If not enough relevant docs and retries remain, loop back to rewriter."""
    relevant = state.get("relevant_docs", [])
    retry_count = state.get("retry_count", 0)

    if len(relevant) < 2 and retry_count < 2:
        return "retry"
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
    graph.add_node("disclosure_analyzer", disclosure_analyzer)

    # Set entry point
    graph.set_entry_point("moderator")

    # Moderator → (blocked → END) | (continue → rewriter)
    graph.add_conditional_edges(
        "moderator",
        should_continue_after_moderator,
        {"blocked": END, "continue": "rewriter"},
    )

    # Rewriter → Router
    graph.add_edge("rewriter", "adaptive_router")

    # Router → Retriever (always — every strategy needs documents)
    graph.add_edge("adaptive_router", "retriever")

    # Retriever → (disclosure_analyzer) | (grader)
    graph.add_conditional_edges(
        "retriever",
        route_after_retrieval,
        {"disclosure_analyzer": "disclosure_analyzer", "grader": "grader"},
    )

    # Grader → (retry → rewriter) | (generate → generator)
    graph.add_conditional_edges(
        "grader",
        should_retry_after_grader,
        {"retry": "rewriter", "generate": "generator"},
    )

    # Generator → END
    graph.add_edge("generator", END)

    # Disclosure Analyzer → END
    graph.add_edge("disclosure_analyzer", END)

    # Compile with checkpointer + HITL interrupt before disclosure_analyzer.
    # - MemorySaver persists state between invocations, enabling pause/resume.
    # - interrupt_before=["disclosure_analyzer"] pauses BEFORE the node runs so
    #   a human reviewer can approve or reject the query via the frontend.
    return graph.compile(
        checkpointer=MemorySaver(),
        interrupt_before=["disclosure_analyzer"],
    )


# Compiled graph instance
app = build_graph()
