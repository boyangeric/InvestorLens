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
    last_state: dict = {}

    try:
        # astream yields state after each node completes — perfect for live trace.
        # When the graph hits an interrupt, the event may include a sentinel
        # whose value isn't a dict (e.g., __interrupt__ → tuple). Guard for it.
        async for event in agent_graph.astream(initial_input, config=config):
            if not isinstance(event, dict):
                continue
            for node_name, partial_state in event.items():
                if not isinstance(partial_state, dict):
                    continue  # skip interrupt sentinels and similar non-state events

                last_state = partial_state

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

    # After streaming, check if we're paused at an interrupt
    snapshot = agent_graph.get_state(config)
    if snapshot.next:
        # Paused — tell the frontend we need human review
        await websocket.send_json({
            "type": "review_required",
            "next_node": snapshot.next[0],
            "query": snapshot.values.get("query", ""),
            "reasoning": snapshot.values.get("strategy_reasoning", ""),
        })
        return

    # Otherwise we've reached END — send the final answer
    await websocket.send_json({
        "type": "final_answer",
        "answer": last_state.get("generation", ""),
        "confidence": last_state.get("confidence", 0.0),
        "trace": last_state.get("node_trace", []),
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
