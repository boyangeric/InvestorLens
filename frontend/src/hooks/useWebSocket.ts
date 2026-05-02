/**
 * useWebSocket — single source of truth for the agent's live state.
 *
 * Owns the WebSocket connection to /ws/{session_id}, parses every server
 * message into typed state, and exposes a small action surface (sendQuery,
 * approve, reject) to the UI.
 *
 * The connection is intentionally long-lived: one socket per session, kept
 * open across multiple queries and HITL pauses. session_id maps 1:1 to the
 * LangGraph thread_id, so reconnects resume from the same checkpoint.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

// --- Server → Client message types (mirror backend/api/chat.py) ---

export type TraceEntry = {
  node: string;
  status?: string;
  duration_ms?: number;
  [key: string]: unknown;
};

export type ServerMessage =
  | { type: "node_end"; node: string; trace: TraceEntry }
  | {
      type: "review_required";
      next_node: string;
      query: string;
      reasoning: string;
    }
  | {
      type: "final_answer";
      answer: string;
      confidence: number;
      trace: TraceEntry[];
    }
  | { type: "error"; message: string };

// --- UI-facing chat shape ---

export type ChatMessage = {
  role: "user" | "assistant";
  content: string;
  confidence?: number;
  trace?: TraceEntry[];
};

export type ReviewState = {
  query: string;
  reasoning: string;
  next_node: string;
} | null;

export type ConnectionStatus = "connecting" | "open" | "closed";

// --- Hook ---

export function useWebSocket(sessionId: string) {
  const [status, setStatus] = useState<ConnectionStatus>("connecting");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [liveTrace, setLiveTrace] = useState<TraceEntry[]>([]);
  const [review, setReview] = useState<ReviewState>(null);
  const [isThinking, setIsThinking] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);

  // The dev server proxies /ws → ws://localhost:8000 (see vite.config.ts),
  // so a relative URL works in both dev and prod behind the same origin.
  const url = useMemo(() => {
    const proto = window.location.protocol === "https:" ? "wss" : "ws";
    return `${proto}://${window.location.host}/ws/${sessionId}`;
  }, [sessionId]);

  useEffect(() => {
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      setStatus("open");
      // Clear any prior error — e.g. React StrictMode's dev-mode
      // double-mount tears down the first connection and fires onerror
      // before the real connection succeeds.
      setError(null);
    };
    ws.onclose = () => setStatus("closed");
    ws.onerror = () => setError("WebSocket connection error");

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data) as ServerMessage;

      switch (msg.type) {
        case "node_end":
          // Append the trace entry as the node finishes — drives live trace UI
          setLiveTrace((prev) => [...prev, msg.trace]);
          break;

        case "review_required":
          // Graph paused at interrupt_before — surface the approval modal
          setIsThinking(false);
          setReview({
            query: msg.query,
            reasoning: msg.reasoning,
            next_node: msg.next_node,
          });
          break;

        case "final_answer":
          // Graph reached END — commit the assistant message
          setIsThinking(false);
          setReview(null);
          setMessages((prev) => [
            ...prev,
            {
              role: "assistant",
              content: msg.answer,
              confidence: msg.confidence,
              trace: msg.trace,
            },
          ]);
          break;

        case "error":
          setIsThinking(false);
          setError(msg.message);
          break;
      }
    };

    return () => {
      ws.close();
    };
  }, [url]);

  const send = useCallback((payload: object) => {
    const ws = wsRef.current;
    if (ws?.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(payload));
    }
  }, []);

  const sendQuery = useCallback(
    (query: string) => {
      // Build chat_history from prior messages so the rewriter has context.
      // Backend expects [{role, content}, ...] — we only send role+content.
      const chat_history = messages.map((m) => ({
        role: m.role,
        content: m.content,
      }));

      setMessages((prev) => [...prev, { role: "user", content: query }]);
      setLiveTrace([]);
      setError(null);
      setIsThinking(true);

      send({ type: "query", query, chat_history });
    },
    [messages, send],
  );

  const approve = useCallback(() => {
    setIsThinking(true);
    setReview(null);
    send({ type: "approve" });
  }, [send]);

  const reject = useCallback(
    (reason: string) => {
      setReview(null);
      send({ type: "reject", reason });
    },
    [send],
  );

  return {
    status,
    messages,
    liveTrace,
    review,
    isThinking,
    error,
    sendQuery,
    approve,
    reject,
  };
}
