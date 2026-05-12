"""
Disclosure Analyzer Node — extracts disclosed risks from financial documents and pauses for human review.

Named "disclosure_analyzer" (not "risk_flagger") because "risk" is ambiguous —
here we specifically mean MATERIAL BUSINESS RISKS THAT THE COMPANY HAS DISCLOSED
in its filings (annual reports, PDS documents, half-year reports). We are not
flagging risky user queries or AI safety risk — we are analysing the company's
own disclosures.

This is the human-in-the-loop (HITL) gate. The node itself runs normally and
produces a structured disclosure report, but the graph is configured with
`interrupt_before=["disclosure_analyzer"]` so execution pauses BEFORE this node
runs.

The FastAPI layer then surfaces the paused state to the frontend, a human
reviews it, and resumes the graph — optionally modifying state first.

When to route here (decided in the router's conditional edge):
  - Queries asking about disclosed risks, warnings, material uncertainties
  - Queries about regulatory issues, litigation, going concern
"""

import logging

from backend.agent.schemas import DisclosureAnalyzerResponse
from backend.agent.state import AgentState
from backend.agent.utils import call_llm_structured, load_prompt

logger = logging.getLogger(__name__)

PROMPT = load_prompt("disclosure_analyzer_v1")


def _format_content(docs: list[dict]) -> str:
    """Join retrieved chunks into a single content blob for disclosure analysis."""
    if not docs:
        return "No document content available."

    parts = []
    for doc in docs:
        source = doc.get("source", "unknown")
        page = doc.get("page", 0)
        content = doc.get("content", "")
        parts.append(f"[Page {page}]\n{content}")

    return "\n\n".join(parts)


def _format_disclosure_report(risks: list, assessment: str) -> str:
    """Turn the structured disclosure list into a readable answer for the user."""
    if not risks:
        return f"No material risks disclosed.\n\n{assessment}"

    lines = ["## Disclosed Risks\n"]
    for risk in risks:
        lines.append(f"### [{risk.severity.upper()}] {risk.title}")
        lines.append(f"{risk.summary}")
        lines.append(f"*[Source: {risk.source}, Page {risk.page}]*\n")

    lines.append(f"\n**Overall Assessment:** {assessment}")
    return "\n".join(lines)


def disclosure_analyzer(state: AgentState) -> dict:
    """
    Analyse retrieved chunks for disclosed material risks.

    Reads: query, relevant_docs (or retrieved_docs as fallback)
    Writes: generation, confidence, node_trace
    """
    query = state["query"]
    docs = state.get("relevant_docs") or state.get("retrieved_docs", [])

    logger.info("Disclosure analyzer: analysing %d chunks", len(docs))

    content = _format_content(docs)
    source = docs[0].get("source", "unknown") if docs else "unknown"

    result = call_llm_structured(
        PROMPT,
        {"source": source, "disclosure_topic": query, "content": content},
        DisclosureAnalyzerResponse,
    )

    response: DisclosureAnalyzerResponse = result["response"]
    risks = response.risks
    assessment = response.overall_risk_assessment

    # Confidence is derived from how many disclosures were found + severity distribution
    high_severity_count = sum(1 for r in risks if r.severity == "high")
    confidence = 0.9 if risks else 0.5  # Lower confidence if none found

    answer = _format_disclosure_report(risks, assessment)

    logger.info(
        "Disclosure analyzer: found %d disclosed risks (%d high severity)",
        len(risks),
        high_severity_count,
    )

    trace_entry = {
        "node": "disclosure_analyzer",
        "status": f"{len(risks)} disclosed risks ({high_severity_count} high)",
        "model": result["model"],
        "duration_ms": result["duration_ms"],
        "tokens_in": result["tokens_in"],
        "tokens_out": result["tokens_out"],
        "cost_usd": result["cost_usd"],
        "disclosed_risks_found": len(risks),
        "high_severity": high_severity_count,
        "requires_review": True,  # Disclosure output ALWAYS requires human review
    }

    node_trace = state.get("node_trace", []) + [trace_entry]

    return {
        "generation": answer,
        "confidence": confidence,
        "current_node": "disclosure_analyzer",
        "node_trace": node_trace,
    }
