/**
 * Sources — citations panel for the most recent agent answer.
 *
 * Pulls `sources` arrays out of trace entries (the generator writes them).
 * The generator's `sources` field is `list[str]` (see GeneratorResponse) —
 * each entry is a citation string in one of these shapes:
 *   - Document    → "filename.pdf, Page X"
 *   - Market data → "yfinance, as_of YYYY-MM-DD"
 *   - News        → "domain.com, published YYYY-MM-DD"
 * We parse them into a tagged union so doc vs live can be styled differently.
 */

import type { ChatMessage, TraceEntry } from "../hooks/useWebSocket";

type DocSource = { kind: "doc"; source: string; page: string };
type LiveSource = { kind: "live"; source: string; meta: string };
type CitationSource = DocSource | LiveSource;

export function Sources({
  messages,
  liveTrace,
}: {
  messages: ChatMessage[];
  liveTrace?: TraceEntry[];
}) {
  const lastAnswer = [...messages].reverse().find((m) => m.role === "assistant");
  // During HITL pause and mid-generation, there's no completed assistant
  // message yet — fall back to the live trace so citations appear as soon
  // as a node emits them (e.g. metric_extractor when the approval modal opens).
  const sources = extractSources(lastAnswer?.trace ?? liveTrace);

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
              <SourceRow key={i} source={s} />
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}

function SourceRow({ source }: { source: CitationSource }) {
  const isLive = source.kind === "live";
  return (
    <li className="group flex items-center gap-2.5 rounded-lg border border-slate-200/70 bg-white px-2.5 py-2 transition-all hover:border-slate-300 hover:shadow-sm">
      <div
        className={`flex h-7 w-7 shrink-0 items-center justify-center rounded-md border ${
          isLive
            ? "border-sky-200 bg-sky-50"
            : "border-slate-200 bg-slate-50"
        }`}
      >
        {isLive ? (
          <svg
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.6"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="h-3.5 w-3.5 text-sky-600"
          >
            <circle cx="12" cy="12" r="10" />
            <line x1="2" y1="12" x2="22" y2="12" />
            <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
          </svg>
        ) : (
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
        )}
      </div>
      <div className="min-w-0 flex-1">
        <div
          className="truncate text-[11.5px] font-medium text-slate-900"
          title={source.source}
        >
          {source.source}
        </div>
        <div className="font-mono text-[10px] tabular-nums text-slate-500">
          {source.kind === "doc" ? (source.page ? `p.${source.page}` : "—") : source.meta}
        </div>
      </div>
      {isLive && (
        <span className="shrink-0 rounded-md bg-sky-50 px-1.5 py-0.5 text-[9.5px] font-semibold uppercase tracking-wide text-sky-700">
          Live
        </span>
      )}
    </li>
  );
}

function parseCitation(raw: string): CitationSource | null {
  if (typeof raw !== "string") return null;
  // Strip any leading "[Source: " / "[Live: " prefix and trailing "]" the
  // model may have left in — the documented format is bare, but be lenient.
  const clean = raw
    .trim()
    .replace(/^\[?\s*(Source|Live)\s*:\s*/i, "")
    .replace(/\]\s*$/, "")
    .trim();
  if (!clean) return null;

  // Live data: "yfinance, as_of 2026-05-17" or "afr.com, published 2026-05-15"
  const liveMatch = clean.match(/^(.+?),\s*(as_of|published)\s+(.+)$/i);
  if (liveMatch) {
    return {
      kind: "live",
      source: liveMatch[1].trim(),
      meta: `${liveMatch[2].toLowerCase()} ${liveMatch[3].trim()}`,
    };
  }

  // Document: "filename.pdf, Page 12" (page may be "1-14" range)
  const docMatch = clean.match(/^(.+?),\s*Page\s+([\w-]+)\s*$/i);
  if (docMatch) {
    return { kind: "doc", source: docMatch[1].trim(), page: docMatch[2].trim() };
  }

  // Fallback — treat as a doc with no page number rather than dropping it.
  return { kind: "doc", source: clean, page: "" };
}

function extractSources(trace: TraceEntry[] | undefined): CitationSource[] {
  if (!trace) return [];

  const all: CitationSource[] = [];
  for (const entry of trace) {
    const raw = entry.sources;
    if (!Array.isArray(raw)) continue;
    for (const s of raw) {
      // Backend currently emits strings; tolerate legacy {source, page}
      // objects so old checkpointed traces still render.
      if (typeof s === "string") {
        const parsed = parseCitation(s);
        if (parsed) all.push(parsed);
      } else if (s && typeof s === "object" && "source" in s) {
        const obj = s as { source: unknown; page?: unknown };
        if (typeof obj.source === "string") {
          all.push({
            kind: "doc",
            source: obj.source,
            page: obj.page != null ? String(obj.page) : "",
          });
        }
      }
    }
  }

  const seen = new Set<string>();
  return all.filter((s) => {
    const key =
      s.kind === "doc" ? `doc::${s.source}::${s.page}` : `live::${s.source}::${s.meta}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}
