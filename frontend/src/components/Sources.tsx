/**
 * Sources — citations panel for the most recent agent answer.
 *
 * Pulls `sources` arrays out of trace entries (the generator writes them).
 * Dedupes by source+page so the same passage isn't shown twice.
 */

import type { ChatMessage, TraceEntry } from "../hooks/useWebSocket";

type Source = { source: string; page: number };

export function Sources({ messages }: { messages: ChatMessage[] }) {
  const lastAnswer = [...messages].reverse().find((m) => m.role === "assistant");
  const sources = extractSources(lastAnswer?.trace);

  return (
    <div className="flex h-full flex-col">
      <div className="border-b border-slate-200/70 bg-white/70 px-6 py-4 backdrop-blur">
        <h2 className="text-[13px] font-semibold tracking-tight text-slate-900">
          Sources
        </h2>
        <p className="mt-0.5 text-[11px] text-slate-500">
          Citations from the last answer
        </p>
      </div>

      <div className="flex-1 overflow-y-auto px-5 py-4">
        {sources.length === 0 ? (
          <div className="mt-12 text-center">
            <div className="mx-auto mb-3 flex h-9 w-9 items-center justify-center rounded-full bg-slate-100">
              <svg
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.6"
                strokeLinecap="round"
                strokeLinejoin="round"
                className="h-4 w-4 text-slate-400"
              >
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                <polyline points="14 2 14 8 20 8" />
              </svg>
            </div>
            <p className="text-[11px] leading-relaxed text-slate-400">
              Citations will appear here after the agent answers.
            </p>
          </div>
        ) : (
          <ul className="space-y-1.5">
            {sources.map((s, i) => (
              <li
                key={i}
                className="group flex items-center gap-2.5 rounded-lg border border-slate-200/70 bg-white px-2.5 py-2 transition-all hover:border-slate-300 hover:shadow-sm"
              >
                <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md border border-slate-200 bg-slate-50">
                  <svg
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="1.6"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    className="h-3.5 w-3.5 text-slate-500"
                  >
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                    <polyline points="14 2 14 8 20 8" />
                  </svg>
                </div>
                <div className="min-w-0 flex-1">
                  <div
                    className="truncate text-[11.5px] font-medium text-slate-900"
                    title={s.source}
                  >
                    {s.source}
                  </div>
                  <div className="font-mono text-[10px] tabular-nums text-slate-500">
                    p.{s.page}
                  </div>
                </div>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}

function extractSources(trace: TraceEntry[] | undefined): Source[] {
  if (!trace) return [];

  const all: Source[] = [];
  for (const entry of trace) {
    const raw = entry.sources;
    if (!Array.isArray(raw)) continue;
    for (const s of raw) {
      if (
        s &&
        typeof s === "object" &&
        "source" in s &&
        "page" in s &&
        typeof (s as Source).source === "string"
      ) {
        all.push({
          source: (s as Source).source,
          page: Number((s as Source).page) || 0,
        });
      }
    }
  }

  const seen = new Set<string>();
  return all.filter((s) => {
    const key = `${s.source}::${s.page}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}
