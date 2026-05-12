"""
WebSocket chat endpoint — streams agent execution to the frontend in real time.

Protocol (messages are JSON):
  Client → Server:
    {"type": "query",   "query": "...", "chat_history": [...]}
    {"type": "approve"}                            # resume paused graph
    {"type": "reject",  "reason": "..."}           # abort paused graph

  Server → Client:
    {"type": "node_start", "node": "moderator"}
    {"type": "node_end",   "node": "moderator", "trace": {...}}
    {"type": "review_required", "query": "...", "reasoning": "..."}
    {"type": "final_answer",    "answer": "...", "confidence": 0.9, "trace": [...]}
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
    if snapshot.next:
        await websocket.send_json({
            "type": "review_required",
            "next_node": snapshot.next[0],
            "query": final_state.get("query", ""),
            "reasoning": final_state.get("strategy_reasoning", ""),
        })
        return

    # Otherwise we've reached END — send the final answer
    trace = final_state.get("node_trace", [])
    await websocket.send_json({
        "type": "final_answer",
        "answer": final_state.get("generation", ""),
        "confidence": final_state.get("confidence", 0.0),
        "grounded": final_state.get("grounded", True),
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
                initial_state = {
                    "query": msg["query"],
                    "original_query": msg["query"],
                    "chat_history": msg.get("chat_history", []),
                    "retry_count": 0,
                    "node_trace": [],
                }
                await _stream_graph(websocket, initial_state, session_id)

            elif msg_type == "approve":
                # Resume from the checkpoint — passing None means continue where we paused
                await _stream_graph(websocket, None, session_id)

            elif msg_type == "reject":
                reason = msg.get("reason", "Query rejected by reviewer.")
                agent_graph.update_state(
                    _config(session_id),
                    {"generation": f"Request rejected: {reason}", "confidence": 0.0},
                )
                await websocket.send_json({
                    "type": "final_answer",
                    "answer": f"Request rejected: {reason}",
                    "confidence": 0.0,
                    "trace": [],
                })

            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Unknown message type: {msg_type}",
                })

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: session=%s", session_id)
