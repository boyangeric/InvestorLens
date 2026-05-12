/**
 * AgentTrace — live visualization of the agent's reasoning path.
 *
 * Each entry in `liveTrace` is one node firing in the LangGraph (moderator,
 * rewriter, adaptive_router, retriever, grader, generator, disclosure_analyzer).
 * Order is the order of execution — including any retry loops, which the
 * grader→rewriter cycle naturally shows as the same node appearing twice.
 *
 * This is the "transparency" pillar of the demo: the user sees *exactly*
 * what the agent did to produce its answer.
 */

import type { TraceEntry } from "../hooks/useWebSocket";

type Props = {
  liveTrace: TraceEntry[];
  isThinking: boolean;
};

const NODE_META: Record<
  string,
  { label: string; accent: string; tone: string }
> = {
  moderator: {
    label: "Moderator",
    accent: "bg-slate-400",
    tone: "text-slate-700",
  },
  rewriter: {
    label: "Query Rewriter",
    accent: "bg-blue-500",
    tone: "text-blue-700",
  },
  adaptive_router: {
    label: "Adaptive Router",
    accent: "bg-purple-500",
    tone: "text-purple-700",
  },
  retriever: {
    label: "Retriever",
    accent: "bg-cyan-500",
    tone: "text-cyan-700",
  },
  grader: {
    label: "Relevance Grader",
    accent: "bg-amber-500",
    tone: "text-amber-700",
  },
  generator: {
    label: "Generator",
    accent: "bg-emerald-500",
    tone: "text-emerald-700",
  },
  faithfulness: {
    label: "Faithfulness Audit",
    accent: "bg-teal-500",
    tone: "text-teal-700",
  },
  general_generator: {
    label: "General Knowledge Fallback",
    accent: "bg-slate-400",
    tone: "text-slate-700",
  },
  disclosure_analyzer: {
    label: "Disclosure Analyzer",
    accent: "bg-rose-500",
    tone: "text-rose-700",
  },
};

export function AgentTrace({ liveTrace, isThinking }: Props) {
  // Running session totals — recomputed every render from the live trace.
  // Each entry's `cost_usd` is set by call_llm, so the sum here is the
  // session-to-date cost without any extra plumbing from the backend.
  const sessionCost = liveTrace.reduce(
    (acc, e) => acc + (typeof e.cost_usd === "number" ? e.cost_usd : 0),
    0,
  );
  const sessionTokensIn = liveTrace.reduce(
    (acc, e) => acc + (typeof e.tokens_in === "number" ? e.tokens_in : 0),
    0,
  );
  const sessionTokensOut = liveTrace.reduce(
    (acc, e) => acc + (typeof e.tokens_out === "number" ? e.tokens_out : 0),
    0,
  );

  return (
    <div className="flex h-full flex-col">
      <div className="border-b border-slate-200/70 bg-white/70 px-6 py-4 backdrop-blur">
        <div className="flex items-start justify-between gap-3">
          <div>
            <h2 className="text-[13px] font-semibold tracking-tight text-slate-900">
              Reasoning trace
            </h2>
            <p className="mt-0.5 text-[11px] text-slate-500">
              Live LangGraph execution
            </p>
          </div>
          {liveTrace.length > 0 && (
            <div className="text-right">
              <div className="font-mono text-[11px] tabular-nums font-semibold text-slate-900">
                ${sessionCost.toFixed(4)}
              </div>
              <div className="font-mono text-[10px] tabular-nums text-slate-400">
                {sessionTokensIn.toLocaleString()}↓{" "}
                {sessionTokensOut.toLocaleString()}↑
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="flex-1 overflow-y-auto px-6 py-5">
        {liveTrace.length === 0 && !isThinking && (
          <div className="mt-16 text-center">
            <div className="mx-auto mb-3 flex h-10 w-10 items-center justify-center rounded-full bg-slate-100">
              <svg
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.6"
                strokeLinecap="round"
                strokeLinejoin="round"
                className="h-5 w-5 text-slate-400"
              >
                <circle cx="12" cy="12" r="10" />
                <polyline points="12 6 12 12 16 14" />
              </svg>
            </div>
            <p className="text-[11px] text-slate-400">
              Send a query to watch the agent reason
            </p>
          </div>
        )}

        <ol className="space-y-2.5">
          {liveTrace.map((entry, i) => (
            <TraceNode
              key={i}
              entry={entry}
              index={i}
              isLast={i === liveTrace.length - 1}
            />
          ))}

          {isThinking && (
            <li className="flex items-center gap-2.5 rounded-lg border border-slate-200/70 bg-white px-3.5 py-2.5 text-[11px] font-medium text-slate-600 shadow-sm">
              <span className="relative flex h-1.5 w-1.5">
                <span className="absolute inset-0 animate-ping rounded-full bg-indigo-400 opacity-75" />
                <span className="relative h-1.5 w-1.5 rounded-full bg-indigo-500" />
              </span>
              <span>Working</span>
            </li>
          )}
        </ol>
      </div>
    </div>
  );
}

function TraceNode({
  entry,
  index,
  isLast,
}: {
  entry: TraceEntry;
  index: number;
  isLast: boolean;
}) {
  const meta = NODE_META[entry.node] ?? {
    label: entry.node,
    accent: "bg-slate-400",
    tone: "text-slate-700",
  };

  return (
    <li className="relative animate-slide-in">
      {/* Connector line down to next node */}
      {!isLast && (
        <span
          aria-hidden
          className="absolute left-[10px] top-full h-2.5 w-px bg-slate-200"
        />
      )}

      <div className="group relative overflow-hidden rounded-lg border border-slate-200/70 bg-white shadow-sm transition-shadow hover:shadow-md">
        {/* Coloured accent stripe — left edge */}
        <span className={`absolute inset-y-0 left-0 w-[3px] ${meta.accent}`} />

        <div className="px-3.5 py-2.5 pl-4">
          <div className="flex items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <span
                className={`flex h-5 w-5 items-center justify-center rounded-full bg-slate-100 text-[10px] font-mono font-semibold ${meta.tone}`}
              >
                {index + 1}
              </span>
              <span className={`text-[12px] font-semibold ${meta.tone}`}>
                {meta.label}
              </span>
            </div>
            {typeof entry.duration_ms === "number" && (
              <span className="font-mono text-[10px] tabular-nums text-slate-400">
                {entry.duration_ms}ms
              </span>
            )}
          </div>

          {entry.status && (
            <p className="mt-1 text-[11px] leading-relaxed text-slate-600">
              {String(entry.status)}
            </p>
          )}

          <TraceDetails entry={entry} />
        </div>
      </div>
    </li>
  );
}

function TraceDetails({ entry }: { entry: TraceEntry }) {
  const details: { label: string; value: string }[] = [];

  if (entry.node === "adaptive_router" && entry.strategy) {
    details.push({ label: "strategy", value: String(entry.strategy) });
  }
  if (entry.node === "retriever" && typeof entry.chunks_retrieved === "number") {
    details.push({ label: "chunks", value: String(entry.chunks_retrieved) });
  }
  if (
    entry.node === "grader" &&
    typeof entry.relevant_count === "number" &&
    typeof entry.total_count === "number"
  ) {
    details.push({
      label: "relevant",
      value: `${entry.relevant_count}/${entry.total_count}`,
    });
  }
  if (entry.node === "generator" && typeof entry.confidence === "number") {
    details.push({
      label: "confidence",
      value: `${Math.round((entry.confidence as number) * 100)}%`,
    });
  }
  if (entry.node === "general_generator") {
    details.push({ label: "source", value: "general knowledge" });
  }
  if (entry.node === "faithfulness" && typeof entry.faithful === "boolean") {
    details.push({
      label: "verdict",
      value: entry.faithful ? "faithful" : "unsupported",
    });
  }
  if (typeof entry.tokens_in === "number" && typeof entry.tokens_out === "number") {
    details.push({
      label: "tokens",
      value: `${entry.tokens_in}↓ ${entry.tokens_out}↑`,
    });
  }
  if (typeof entry.cost_usd === "number" && entry.cost_usd > 0) {
    // Sub-cent costs need 4 decimals to be readable; cents+ get 3.
    const cost = entry.cost_usd as number;
    const formatted = cost < 0.01 ? cost.toFixed(4) : cost.toFixed(3);
    details.push({ label: "cost", value: `$${formatted}` });
  }

  if (details.length === 0) return null;

  return (
    <div className="mt-2 flex flex-wrap gap-1.5">
      {details.map((d) => (
        <span
          key={d.label}
          className="inline-flex items-center gap-1 rounded-md bg-slate-50 px-1.5 py-0.5 text-[10px] tabular-nums"
        >
          <span className="text-slate-400">{d.label}</span>
          <span className="font-mono font-medium text-slate-700">
            {d.value}
          </span>
        </span>
      ))}
    </div>
  );
}
