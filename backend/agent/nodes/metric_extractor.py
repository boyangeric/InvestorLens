"""
Metric Extractor Node — high-stakes data extraction with human-in-the-loop verification.

Extracts specific numeric financial metrics (revenue, net profit, margins, EPS,
ratios, etc.) from retrieved chunks, with exact figures and page citations.
Because a single wrong number can move investment decisions, the graph is
configured with `interrupt_after=["metric_extractor"]` so the agent pauses
AFTER extraction — the human reviewer sees the actual extracted figures and
their source pages, and verifies them before they are used downstream.

This is the human-in-the-loop (HITL) gate. The node itself runs normally and
writes both:
  - `extracted_metrics`: typed list of {name, value, period, source, page} for
                         the approval modal and any downstream consumers
  - `generation`:        a formatted markdown table that becomes the final
                         answer if approved

If approved, the formatted table flows out as the final answer. If rejected,
the API layer overwrites `generation` with the reviewer's rejection text and
does not resume the graph.
"""

import logging

from backend.agent.schemas import ExtractedMetric, MetricExtractorResponse
from backend.agent.state import AgentState
from backend.agent.utils import call_llm_structured, load_prompt

logger = logging.getLogger(__name__)

PROMPT = load_prompt("extractor_v1")


def _format_content(docs: list[dict]) -> str:
    """Join retrieved chunks into a single content blob for extraction."""
    if not docs:
        return "No document content available."

    parts = []
    for doc in docs:
        page = doc.get("page", 0)
        content = doc.get("content", "")
        parts.append(f"[Page {page}]\n{content}")

    return "\n\n".join(parts)


def format_metrics_table(
    document: str,
    period: str,
    metrics: list[ExtractedMetric],
    highlights: list[str],
) -> str:
    """Render the extracted metrics as a reviewable markdown table."""
    if not metrics:
        return f"No financial metrics extracted from {document}."

    lines = [
        f"## Extracted Financial Metrics — {document} ({period})\n",
        "| Metric | Value | Period | Page |",
        "|---|---|---|---|",
    ]
    for m in metrics:
        lines.append(f"| {m.name} | {m.value} | {m.period} | {m.page} |")

    if highlights:
        lines.append("\n**Key highlights:**")
        for h in highlights:
            lines.append(f"- {h}")

    return "\n".join(lines)


def metric_extractor(state: AgentState) -> dict:
    """
    Extract financial metrics from retrieved chunks.

    Reads: query, relevant_docs (or retrieved_docs as fallback)
    Writes: extracted_metrics, generation, confidence, node_trace
    """
    query = state["query"]
    docs = state.get("relevant_docs") or state.get("retrieved_docs", [])

    logger.info("Metric extractor: extracting from %d chunks", len(docs))

    content = _format_content(docs)
    source = docs[0].get("source", "unknown") if docs else "unknown"

    result = call_llm_structured(
        PROMPT,
        {"source": source, "content": content},
        MetricExtractorResponse,
    )

    response: MetricExtractorResponse = result["response"]
    metrics = response.metrics
    document = response.document or source
    period = response.period
    highlights = response.key_highlights

    answer = format_metrics_table(document, period, metrics, highlights)

    # Confidence rises with the number of extracted metrics — a sparse
    # extraction is more likely to have missed something than a rich one.
    confidence = 0.9 if metrics else 0.4

    logger.info("Metric extractor: extracted %d metrics from %s", len(metrics), document)

    trace_entry = {
        "node": "metric_extractor",
        "status": f"{len(metrics)} metrics extracted",
        "model": result["model"],
        "duration_ms": result["duration_ms"],
        "tokens_in": result["tokens_in"],
        "tokens_out": result["tokens_out"],
        "cost_usd": result["cost_usd"],
        "metrics_extracted": len(metrics),
        "requires_review": True,  # Extracted figures ALWAYS require human verification
    }

    node_trace = state.get("node_trace", []) + [trace_entry]

    # Surface the structured list to state so the approval modal can render
    # the actual numbers being verified, not just a free-text summary.
    extracted_dicts = [m.model_dump() for m in metrics]

    return {
        "generation": answer,
        "confidence": confidence,
        "extracted_metrics": extracted_dicts,
        "extracted_highlights": highlights,
        # Reset verification_status — a new extraction round invalidates any prior verdict.
        "verification_status": "",
        "current_node": "metric_extractor",
        "node_trace": node_trace,
    }
