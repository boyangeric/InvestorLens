/**
 * ChatPage — conversation + live agent trace + sources.
 *
 * Layout: chat (flex-1) | trace (380px) | sources (260px) on wide screens;
 * on narrow screens, sources collapses (handled via responsive classes).
 *
 * All data comes from useWebSocket via props — this page is purely a
 * compositional layer that arranges the existing primitives.
 */

import { AgentTrace } from "../components/AgentTrace";
import { ChatPanel } from "../components/ChatPanel";
import { Sources } from "../components/Sources";
import type { ChatMessage, ReviewState, TraceEntry } from "../hooks/useWebSocket";

type Props = {
  messages: ChatMessage[];
  liveTrace: TraceEntry[];
  isThinking: boolean;
  review: ReviewState;
  status: "connecting" | "open" | "closed";
  onSend: (query: string) => void;
};

export function ChatPage({
  messages,
  liveTrace,
  isThinking,
  review,
  status,
  onSend,
}: Props) {
  return (
    <div className="flex h-full overflow-hidden">
      {/* Conversation column */}
      <section className="flex flex-1 flex-col border-r border-slate-200 bg-white">
        <ChatPanel
          messages={messages}
          isThinking={isThinking}
          onSend={onSend}
          disabled={status !== "open" || review !== null}
        />
      </section>

      {/* Live trace column */}
      <section className="w-[380px] shrink-0 overflow-y-auto border-r border-slate-200 bg-slate-50">
        <AgentTrace liveTrace={liveTrace} isThinking={isThinking} />
      </section>

      {/* Sources column */}
      <section className="hidden w-[260px] shrink-0 overflow-y-auto bg-white lg:block">
        <Sources messages={messages} liveTrace={liveTrace} />
      </section>
    </div>
  );
}
