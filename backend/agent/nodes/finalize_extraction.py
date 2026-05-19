"""
Finalize extraction — runs after the HITL pause and re-renders the answer.

The graph pauses at `interrupt_before=["finalize_extraction"]`, so by the
time this node executes, the reviewer has responded via the API:
  - "verified": metrics untouched, this is just a pass-through
  - "edited":   reviewer corrected one or more values; state.extracted_metrics
                has been overwritten via update_state. Re-render the markdown
                table to reflect the corrected figures.
  - "skipped":  reviewer chose to proceed without verifying; the answer flows
                through but the frontend tags it with an "unverified" badge.

We always re-render from `extracted_metrics` so the output is consistent with
state — edits are reflected, and the original (pre-edit) markdown written by
metric_extractor is replaced. The verification_status itself is *not* stamped
into the markdown body; the frontend renders the badge as separate UI so the
provenance metadata stays distinct from the data.
"""

import logging

from backend.agent.nodes.metric_extractor import format_metrics_table
from backend.agent.schemas import ExtractedMetric
from backend.agent.state import AgentState

logger = logging.getLogger(__name__)


def finalize_extraction(state: AgentState) -> dict:
    """
    Re-render `generation` from (possibly edited) extracted_metrics and append
    a trace entry recording the reviewer's verdict.
    """
    metrics_dicts = state.get("extracted_metrics", []) or []
    status = state.get("verification_status", "") or "verified"
    highlights = state.get("extracted_highlights", []) or []

    update: dict = {
        "verification_status": status,
        "current_node": "finalize_extraction",
    }

    if metrics_dicts:
        # Round-trip through Pydantic so we get validation on edited values
        # and consistent rendering with metric_extractor.
        metrics = [ExtractedMetric(**m) for m in metrics_dicts]
        document = metrics[0].source or "unknown"
        period = metrics[0].period or ""
        update["generation"] = format_metrics_table(
            document, period, metrics, highlights
        )

    # Re-derive citations from the (possibly edited) metrics so the Sources
    # panel reflects post-edit state, not the original extraction.
    source_strings = sorted(
        {f"{m.get('source', 'unknown')}, Page {m.get('page', 0)}" for m in metrics_dicts}
    )

    trace_entry = {
        "node": "finalize_extraction",
        "status": f"reviewer: {status}",
        "duration_ms": 0,
        "verification_status": status,
        "metric_count": len(metrics_dicts),
        "sources": source_strings,
    }
    update["node_trace"] = state.get("node_trace", []) + [trace_entry]

    logger.info(
        "Finalize extraction: status=%s metrics=%d", status, len(metrics_dicts)
    )
    return update
