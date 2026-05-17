"""
WebSocket chat endpoint — streams agent execution to the frontend in real time.

Protocol (messages are JSON):
  Client → Server:
    {"type": "query",         "query": "...", "chat_history": [...]}
    {"type": "approve"}                                    # resume; figures unchanged
    {"type": "approve_edits", "metrics": [{...}, ...]}     # resume with edited figures
    {"type": "skip"}                                       # resume; tag as unverified
    {"type": "reject",        "reason": "..."}             # abort paused graph

  Server → Client:
    {"type": "node_start", "node": "moderator"}
    {"type": "node_end",   "node": "moderator", "trace": {...}}
    {"type": "review_required", "query": "...", "reasoning": "...",
                                "extracted_metrics": [...], "preview": "..."}
    {"type": "final_answer",    "answer": "...", "confidence": 0.9,
                                "verification_status": "verified|edited|skipped|",
                                "trace": [...]}
    {"type": "error",   "message": "..."}

The session_id maps 1:1 to the LangGraph `thread_id`, so state is durable
across WebSocket reconnects (via the MemorySaver checkpointer).
"""

import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from langchain_core.runnables import RunnableConfig

from backend.agent.graph import app as agent_graph
from backend.agent.utils import aggregate_usage

logger = logging.getLogger(__name__)

router = APIRouter()


def _config(session_id: str) -> RunnableConfig:
    """LangGraph config for a given session — thread_id pins the checkpoint."""
    return {"configurable": {"thread_id": session_id}}


async def _stream_graph(websocket: WebSocket, initial_input, session_id: str) -> None:
    """
    Run the graph with streaming and forward events to the WebSocket.

    `initial_input` is either a dict (new query) or None (resume after pause).
    """
    config = _config(session_id)

    try:
        # astream yields per-node *deltas* (not accumulated state) — perfect
        # for live trace, but useless for the final answer payload because the
        # last node's delta only contains the fields *that node* writes.
        # E.g. faithfulness returns {faithful, node_trace} with no `generation`.
        # We use the deltas only to push trace events to the UI, then read the
        # full accumulated state from the checkpointer below.
        async for event in agent_graph.astream(initial_input, config=config):
            if not isinstance(event, dict):
                continue
            for node_name, partial_state in event.items():
                if not isinstance(partial_state, dict):
                    continue  # skip interrupt sentinels and similar non-state events

                trace = partial_state.get("node_trace", [])
                latest_trace = trace[-1] if trace else {}

                await websocket.send_json({
                    "type": "node_end",
                    "node": node_name,
                    "trace": latest_trace,
                })
    except Exception as e:
        logger.exception("Graph streaming failed")
        await websocket.send_json({"type": "error", "message": str(e)})
        return

    # Pull the full accumulated state from the checkpointer — this has every
    # field every node has written, not just the last delta.
    snapshot = agent_graph.get_state(config)
    final_state = snapshot.values or {}

    # If we're paused at an interrupt, surface the approval modal instead.
    # For the metric_extractor gate this is interrupt_AFTER, so the state
    # already contains the extracted figures — we surface them so the modal
    # can render the actual numbers being verified, not just an abstract
    # "approve this step" prompt.
    if snapshot.next:
        await websocket.send_json({
            "type": "review_required",
            "paused_after": final_state.get("current_node", ""),
            "query": final_state.get("query", ""),
            "reasoning": final_state.get("strategy_reasoning", ""),
            "extracted_metrics": final_state.get("extracted_metrics", []),
            "preview": final_state.get("generation", ""),
        })
        return

    # Otherwise we've reached END — send the final answer
    trace = final_state.get("node_trace", [])
    await websocket.send_json({
        "type": "final_answer",
        "answer": final_state.get("generation", ""),
        "confidence": final_state.get("confidence", 0.0),
        "grounded": final_state.get("grounded", True),
        "verification_status": final_state.get("verification_status", ""),
        "trace": trace,
        "token_usage": aggregate_usage(trace),
    })


@router.websocket("/ws/{session_id}")
async def chat_ws(websocket: WebSocket, session_id: str) -> None:
    """
    One long-lived WebSocket per session. Handles initial queries and
    HITL approve/reject messages over the same connection.
    """
    await websocket.accept()
    logger.info("WebSocket connected: session=%s", session_id)

    try:
        while True:
            msg = await websocket.receive_json()
            msg_type = msg.get("type")

            if msg_type == "query":
                # Reset every ephemeral per-query field. The thread_id is
                # reused across turns (so HITL resume + chat-history context
                # work), so the checkpointer still holds last turn's
                # `generation`, `live_quotes`, etc. If we don't reset them,
                # `should_continue_after_moderator` sees a non-empty
                # `generation` from the prior turn, decides the moderator
                # "blocked" this query, and routes straight to END —
                # replaying last turn's answer for the new query.
                initial_state = {
                    "query": msg["query"],
                    "original_query": msg["query"],
                    "chat_history": msg.get("chat_history", []),
                    "retry_count": 0,
                    "node_trace": [],
                    # Retrieval / generation
                    "retrieval_strategy": "",
                    "strategy_reasoning": "",
                    "retrieved_docs": [],
                    "relevant_docs": [],
                    "generation": "",
                    "confidence": 0.0,
                    "grounded": True,
                    "faithful": True,
                    # HITL
                    "extracted_metrics": [],
                    "extracted_highlights": [],
                    "verification_status": "",
                    # Live tools
                    "live_quotes": [],
                    "live_news": [],
                    # Observability
                    "current_node": "",
                }
                await _stream_graph(websocket, initial_state, session_id)

            elif msg_type == "approve":
                # Resume with metrics unchanged — finalize_extraction re-renders
                # from existing state and stamps verification_status="verified".
                agent_graph.update_state(
                    _config(session_id),
                    {"verification_status": "verified"},
                )
                await _stream_graph(websocket, None, session_id)

            elif msg_type == "approve_edits":
                # Reviewer corrected one or more values. Overwrite extracted_metrics
                # with the edited list and resume; finalize_extraction re-renders
                # the markdown table from the new values.
                edited = msg.get("metrics", [])
                agent_graph.update_state(
                    _config(session_id),
                    {
                        "extracted_metrics": edited,
                        "verification_status": "edited",
                    },
                )
                await _stream_graph(websocket, None, session_id)

            elif msg_type == "skip":
                # Reviewer chose not to verify — proceed with original figures but
                # mark them as unverified so the frontend renders a warning badge.
                agent_graph.update_state(
                    _config(session_id),
                    {"verification_status": "skipped"},
                )
                await _stream_graph(websocket, None, session_id)

            elif msg_type == "reject":
                reason = msg.get("reason", "Query rejected by reviewer.")
                agent_graph.update_state(
                    _config(session_id),
                    {
                        "generation": f"Request rejected: {reason}",
                        "confidence": 0.0,
                        "verification_status": "rejected",
                    },
                )
                await websocket.send_json({
                    "type": "final_answer",
                    "answer": f"Request rejected: {reason}",
                    "confidence": 0.0,
                    "verification_status": "rejected",
                    "trace": [],
                })

            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Unknown message type: {msg_type}",
                })

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: session=%s", session_id)
