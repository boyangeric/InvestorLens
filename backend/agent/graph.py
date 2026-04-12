"""
LangGraph StateGraph definition for InvestorLens.

This wires all nodes together with conditional edges:
  Moderator → Rewriter → Router → Retriever → Grader → Generator
                                                 ↑          ↓
                                                 └── (retry if low relevance)

Risk Flagger branches off when the router detects risk-related queries.
The Grader → Rewriter cycle runs up to 2 retries before falling back.
"""

import logging

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from backend.agent.state import AgentState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stub nodes — each will be replaced with real implementations in later steps
# ---------------------------------------------------------------------------

def moderator(state: AgentState) -> dict:
    """Check query safety and relevance. Stub: always passes."""
    logger.info("Node: moderator (stub)")
    return {"current_node": "moderator"}


def rewriter(state: AgentState) -> dict:
    """Rewrite ambiguous queries using chat history. Stub: passes query through."""
    logger.info("Node: rewriter (stub)")
    return {"current_node": "rewriter"}


def router(state: AgentState) -> dict:
    """Classify query into retrieval strategy. Stub: defaults to semantic_search."""
    logger.info("Node: router (stub)")
    return {
        "current_node": "router",
        "retrieval_strategy": "semantic_search",
        "strategy_reasoning": "default stub routing",
    }


def retriever(state: AgentState) -> dict:
    """Execute retrieval strategy. Stub: returns empty docs."""
    logger.info("Node: retriever (stub)")
    return {"current_node": "retriever", "retrieved_docs": []}


def grader(state: AgentState) -> dict:
    """Score chunk relevance. Stub: passes all docs through."""
    logger.info("Node: grader (stub)")
    return {
        "current_node": "grader",
        "relevant_docs": state.get("retrieved_docs", []),
    }


def generator(state: AgentState) -> dict:
    """Generate grounded answer with confidence. Stub: placeholder response."""
    logger.info("Node: generator (stub)")
    return {
        "current_node": "generator",
        "generation": "This is a stub response.",
        "confidence": 1.0,
    }


def risk_flagger(state: AgentState) -> dict:
    """Flag risk disclosures for human review. Stub: no-op."""
    logger.info("Node: risk_flagger (stub)")
    return {"current_node": "risk_flagger"}


# ---------------------------------------------------------------------------
# Conditional edge functions
# ---------------------------------------------------------------------------

def should_continue_after_moderator(state: AgentState) -> str:
    """If moderator blocks, go to END. Otherwise continue to rewriter."""
    if state.get("generation"):  # Moderator set a rejection message
        return "blocked"
    return "continue"


def route_after_router(state: AgentState) -> str:
    """Branch based on retrieval strategy. Risk queries go to risk_flagger."""
    strategy = state.get("retrieval_strategy", "semantic_search")
    if strategy == "flag_risks":
        return "risk_flagger"
    return "retriever"


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
    graph.add_node("router", router)
    graph.add_node("retriever", retriever)
    graph.add_node("grader", grader)
    graph.add_node("generator", generator)
    graph.add_node("risk_flagger", risk_flagger)

    # Set entry point
    graph.set_entry_point("moderator")

    # Moderator → (blocked → END) | (continue → rewriter)
    graph.add_conditional_edges(
        "moderator",
        should_continue_after_moderator,
        {"blocked": END, "continue": "rewriter"},
    )

    # Rewriter → Router
    graph.add_edge("rewriter", "router")

    # Router → (risk_flagger) | (retriever)
    graph.add_conditional_edges(
        "router",
        route_after_router,
        {"risk_flagger": "risk_flagger", "retriever": "retriever"},
    )

    # Retriever → Grader
    graph.add_edge("retriever", "grader")

    # Grader → (retry → rewriter) | (generate → generator)
    graph.add_conditional_edges(
        "grader",
        should_retry_after_grader,
        {"retry": "rewriter", "generate": "generator"},
    )

    # Generator → END
    graph.add_edge("generator", END)

    # Risk Flagger → END
    graph.add_edge("risk_flagger", END)

    return graph.compile()


# Compiled graph instance
app = build_graph()
