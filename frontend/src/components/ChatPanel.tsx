/**
 * ChatPanel — the conversation view.
 *
 * Pure presentational: takes messages + an `onSend` callback and renders
 * them. All agent state lives upstream in useWebSocket.
 */

import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import type { ChatMessage, VerificationStatus } from "../hooks/useWebSocket";

type Props = {
  messages: ChatMessage[];
  isThinking: boolean;
  onSend: (query: string) => void;
  disabled: boolean;
};

const SUGGESTIONS = [
  "What was the underlying profit before tax?",
  "Extract revenue, underlying profit, and EPS for Qantas HY26",
  "Compare domestic and international segment performance",
];

export function ChatPanel({ messages, isThinking, onSend, disabled }: Props) {
  const [input, setInput] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages, isThinking]);

  const submit = (text?: string) => {
    const value = (text ?? input).trim();
    if (!value || disabled) return;
    onSend(value);
    setInput("");
  };

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between border-b border-slate-200/70 bg-white/70 px-6 py-4 backdrop-blur">
        <div>
          <h2 className="text-[13px] font-semibold tracking-tight text-slate-900">
            Conversation
          </h2>
          <p className="mt-0.5 text-[11px] text-slate-500">
            Multi-turn, grounded in your indexed documents
          </p>
        </div>
      </div>

      <div ref={scrollRef} className="flex-1 space-y-5 overflow-y-auto px-6 py-6">
        {messages.length === 0 && !isThinking ? (
          <EmptyState onPick={(q) => submit(q)} />
        ) : (
          messages.map((msg, i) => <MessageBubble key={i} message={msg} />)
        )}

        {isThinking && <ThinkingIndicator />}
      </div>

      <div className="border-t border-slate-200/70 bg-white p-4">
        <div className="flex gap-2 rounded-xl border border-slate-200 bg-white p-1.5 shadow-sm transition-colors focus-within:border-slate-400 focus-within:shadow-md">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                submit();
              }
            }}
            placeholder={
              disabled
                ? "Waiting…"
                : "Ask about your documents — ⏎ to send, ⇧⏎ for newline"
            }
            disabled={disabled}
            rows={2}
            className="flex-1 resize-none border-0 bg-transparent px-2.5 py-1.5 text-[13px] leading-relaxed placeholder:text-slate-400 focus:outline-none focus:ring-0 disabled:bg-transparent disabled:text-slate-400"
          />
          <button
            onClick={() => submit()}
            disabled={disabled || !input.trim()}
            className="self-end rounded-lg bg-slate-900 px-3.5 py-1.5 text-[12px] font-medium text-white shadow-sm transition-all hover:bg-slate-800 active:scale-[0.98] disabled:cursor-not-allowed disabled:bg-slate-200 disabled:text-slate-400 disabled:shadow-none"
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
}

function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";
  return (
    <div
      className={`flex animate-slide-in ${isUser ? "justify-end" : "justify-start"}`}
    >
      <div
        className={`max-w-[85%] rounded-2xl px-4 py-2.5 text-[13px] ${
          isUser
            ? "bg-slate-900 text-white shadow-sm"
            : "border border-slate-200/70 bg-white text-slate-900 shadow-sm"
        }`}
      >
        {!isUser && <VerificationBadge status={message.verification_status} />}
        {isUser ? (
          <div className="whitespace-pre-wrap break-words leading-relaxed">
            {message.content}
          </div>
        ) : (
          <div className="break-words leading-relaxed">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                h1: ({ children }) => (
                  <h1 className="mb-2 mt-1 text-[14px] font-semibold text-slate-900">
                    {children}
                  </h1>
                ),
                h2: ({ children }) => (
                  <h2 className="mb-2 mt-3 text-[13px] font-semibold text-slate-900 first:mt-0">
                    {children}
                  </h2>
                ),
                h3: ({ children }) => (
                  <h3 className="mb-1.5 mt-2.5 text-[12px] font-semibold text-slate-900">
                    {children}
                  </h3>
                ),
                p: ({ children }) => (
                  <p className="my-1.5 first:mt-0 last:mb-0">{children}</p>
                ),
                ul: ({ children }) => (
                  <ul className="my-1.5 list-disc space-y-0.5 pl-5">
                    {children}
                  </ul>
                ),
                ol: ({ children }) => (
                  <ol className="my-1.5 list-decimal space-y-0.5 pl-5">
                    {children}
                  </ol>
                ),
                strong: ({ children }) => (
                  <strong className="font-semibold text-slate-900">
                    {children}
                  </strong>
                ),
                em: ({ children }) => (
                  <em className="italic text-slate-700">{children}</em>
                ),
                code: ({ children }) => (
                  <code className="rounded bg-slate-100 px-1 py-0.5 font-mono text-[11px] text-slate-800">
                    {children}
                  </code>
                ),
                a: ({ href, children }) => (
                  <a
                    href={href}
                    target="_blank"
                    rel="noreferrer"
                    className="text-indigo-600 underline underline-offset-2 hover:text-indigo-700"
                  >
                    {children}
                  </a>
                ),
                table: ({ children }) => (
                  <div className="my-2.5 overflow-x-auto rounded-lg border border-slate-200">
                    <table className="w-full text-[12px]">{children}</table>
                  </div>
                ),
                thead: ({ children }) => (
                  <thead className="bg-slate-50 text-left text-[11px] uppercase tracking-wide text-slate-500">
                    {children}
                  </thead>
                ),
                tbody: ({ children }) => (
                  <tbody className="divide-y divide-slate-100">{children}</tbody>
                ),
                tr: ({ children }) => <tr>{children}</tr>,
                th: ({ children }) => (
                  <th className="px-2.5 py-1.5 font-semibold">{children}</th>
                ),
                td: ({ children }) => (
                  <td className="px-2.5 py-1.5 align-top tabular-nums">
                    {children}
                  </td>
                ),
                hr: () => <hr className="my-3 border-slate-200" />,
              }}
            >
              {message.content}
            </ReactMarkdown>
          </div>
        )}
        {!isUser && message.grounded === false ? (
          <div className="mt-2.5 flex items-center gap-2 border-t border-slate-100 pt-2 text-[10px] text-slate-500">
            <span className="inline-flex h-1.5 w-1.5 rounded-full bg-slate-400" />
            <span className="font-medium uppercase tracking-[0.06em] text-slate-500">
              Answered from general knowledge
            </span>
          </div>
        ) : (
          !isUser &&
          message.confidence !== undefined && (
            <div className="mt-2.5 flex items-center gap-2 border-t border-slate-100 pt-2 text-[10px] text-slate-500">
              <span className="font-medium uppercase tracking-[0.06em]">
                Confidence
              </span>
              <ConfidenceBar value={message.confidence} />
              <span className="font-mono font-semibold tabular-nums text-slate-700">
                {Math.round(message.confidence * 100)}%
              </span>
            </div>
          )
        )}
      </div>
    </div>
  );
}

function VerificationBadge({
  status,
}: {
  status: VerificationStatus | undefined;
}) {
  if (!status || status === "rejected") return null;

  const config: Record<
    Exclude<VerificationStatus, "" | "rejected">,
    { label: string; tone: string; icon: string }
  > = {
    verified: {
      label: "Human-verified",
      tone: "bg-emerald-50 text-emerald-700 border-emerald-200",
      icon: "✓",
    },
    edited: {
      label: "Human-corrected",
      tone: "bg-indigo-50 text-indigo-700 border-indigo-200",
      icon: "✎",
    },
    skipped: {
      label: "Unverified extraction",
      tone: "bg-amber-50 text-amber-800 border-amber-200",
      icon: "⚠",
    },
  };
  const meta = config[status as Exclude<VerificationStatus, "" | "rejected">];
  if (!meta) return null;

  return (
    <div
      className={`mb-2 inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide ${meta.tone}`}
    >
      <span aria-hidden>{meta.icon}</span>
      <span>{meta.label}</span>
    </div>
  );
}

function ConfidenceBar({ value }: { value: number }) {
  const pct = Math.max(0, Math.min(1, value)) * 100;
  const gradient =
    value >= 0.75
      ? "from-emerald-400 to-emerald-500"
      : value >= 0.5
        ? "from-amber-400 to-amber-500"
        : "from-red-400 to-red-500";
  return (
    <div className="h-1.5 w-24 overflow-hidden rounded-full bg-slate-100">
      <div
        className={`h-full bg-gradient-to-r ${gradient}`}
        style={{ width: `${pct}%` }}
      />
    </div>
  );
}

function ThinkingIndicator() {
  return (
    <div className="flex justify-start">
      <div className="flex items-center gap-1.5 rounded-2xl border border-slate-200 bg-white px-4 py-3 shadow-sm">
        <Dot delay="0ms" />
        <Dot delay="150ms" />
        <Dot delay="300ms" />
      </div>
    </div>
  );
}

function Dot({ delay }: { delay: string }) {
  return (
    <span
      className="inline-block h-1.5 w-1.5 animate-bounce rounded-full bg-slate-400"
      style={{ animationDelay: delay }}
    />
  );
}

function EmptyState({ onPick }: { onPick: (q: string) => void }) {
  return (
    <div className="mx-auto max-w-md py-12 text-center">
      <div className="mx-auto mb-5 flex h-12 w-12 items-center justify-center rounded-xl bg-slate-900 shadow-sm">
        <svg
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.6"
          strokeLinecap="round"
          strokeLinejoin="round"
          className="h-5 w-5 text-white"
        >
          <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" />
        </svg>
      </div>
      <h3 className="text-[15px] font-semibold tracking-tight text-slate-900">
        How can I help analyse your documents?
      </h3>
      <p className="mx-auto mt-1.5 max-w-xs text-[12px] leading-relaxed text-slate-500">
        The agent picks a retrieval strategy per query and streams its
        reasoning live.
      </p>

      <div className="mt-6 space-y-1.5 text-left">
        {SUGGESTIONS.map((s) => (
          <button
            key={s}
            onClick={() => onPick(s)}
            className="group flex w-full items-center justify-between gap-3 rounded-lg border border-slate-200/70 bg-white px-3.5 py-2.5 text-[12px] text-slate-700 transition-all hover:border-slate-300 hover:bg-slate-50 hover:shadow-sm"
          >
            <span className="leading-snug">{s}</span>
            <span className="shrink-0 text-slate-300 transition-all group-hover:translate-x-0.5 group-hover:text-slate-600">
              →
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}
