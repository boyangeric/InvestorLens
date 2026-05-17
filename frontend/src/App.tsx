/**
 * App — workspace shell for InvestorLens.
 *
 * Layout: Sidebar nav (left) + active page (right). Two pages share the
 * same WebSocket connection and session state so the user can flip between
 * Documents and Chat without losing the conversation.
 *
 * Why the WebSocket lives at App level (not per-page): the agent's
 * conversation is a session-long thing. If the user is mid-chat and pops
 * over to upload a new doc, we don't want to drop the socket and lose
 * checkpointed state. One socket, one session_id, two views over it.
 */

import { useMemo, useState } from "react";

import { ApprovalModal } from "./components/ApprovalModal";
import { Sidebar } from "./components/Sidebar";
import { useWebSocket } from "./hooks/useWebSocket";
import { ChatPage } from "./pages/ChatPage";
import { DocumentsPage } from "./pages/DocumentsPage";

export type Page = "chat" | "documents";

function newSessionId(): string {
  return crypto.randomUUID();
}

export default function App() {
  const sessionId = useMemo(newSessionId, []);
  const [page, setPage] = useState<Page>("chat");

  const {
    status,
    messages,
    liveTrace,
    review,
    isThinking,
    error,
    sendQuery,
    approve,
    approveEdits,
    skip,
    reject,
  } = useWebSocket(sessionId);

  return (
    <div className="flex h-screen bg-slate-50/40">
      <Sidebar
        page={page}
        onChange={setPage}
        status={status}
        sessionId={sessionId}
      />

      <main className="flex flex-1 flex-col overflow-hidden">
        {/* Only show error banner if we're actually disconnected — transient
            errors during connect (e.g. StrictMode double-mount) are noise. */}
        {error && status !== "open" && (
          <div className="border-b border-red-200 bg-red-50 px-6 py-2 text-xs text-red-700">
            {error}
          </div>
        )}

        <div className="flex-1 overflow-hidden">
          {page === "chat" ? (
            <ChatPage
              messages={messages}
              liveTrace={liveTrace}
              isThinking={isThinking}
              review={review}
              status={status}
              onSend={sendQuery}
            />
          ) : (
            <DocumentsPage />
          )}
        </div>
      </main>

      {review && (
        <ApprovalModal
          review={review}
          onApprove={approve}
          onApproveEdits={approveEdits}
          onSkip={skip}
          onReject={reject}
        />
      )}
    </div>
  );
}
