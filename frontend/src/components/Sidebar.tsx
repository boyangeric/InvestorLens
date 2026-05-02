/**
 * Sidebar — left-rail navigation.
 *
 * Two destinations: Documents (corpus management) and Chat (the agent).
 * Uses simple parent-driven page state instead of react-router — for two
 * pages, a router would be over-engineered.
 */

import type { Page } from "../App";

type Props = {
  page: Page;
  onChange: (page: Page) => void;
  status: "connecting" | "open" | "closed";
  sessionId: string;
};

const NAV: { key: Page; label: string; hint: string; icon: JSX.Element }[] = [
  {
    key: "chat",
    label: "Chat",
    hint: "Ask the agent",
    icon: (
      <svg
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
        className="h-4 w-4"
      >
        <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
      </svg>
    ),
  },
  {
    key: "documents",
    label: "Documents",
    hint: "Manage corpus",
    icon: (
      <svg
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
        className="h-4 w-4"
      >
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
        <polyline points="14 2 14 8 20 8" />
        <line x1="9" y1="13" x2="15" y2="13" />
        <line x1="9" y1="17" x2="15" y2="17" />
      </svg>
    ),
  },
];

export function Sidebar({ page, onChange, status, sessionId }: Props) {
  return (
    <aside className="flex h-full w-60 shrink-0 flex-col border-r border-slate-200/80 bg-white/60 backdrop-blur">
      {/* Brand */}
      <div className="flex items-center gap-2.5 px-5 pb-5 pt-6">
        <div className="relative h-8 w-8 overflow-hidden rounded-lg bg-slate-900 shadow-sm ring-1 ring-slate-900/5">
          <div className="absolute inset-0 bg-gradient-to-br from-indigo-500/30 via-transparent to-fuchsia-500/20" />
          <div className="relative flex h-full items-center justify-center text-[11px] font-bold tracking-tight text-white">
            iL
          </div>
        </div>
        <div className="leading-tight">
          <div className="text-[13px] font-semibold tracking-tight text-slate-900">
            InvestorLens
          </div>
          <div className="text-[10px] font-medium text-slate-500">
            Agentic analyst
          </div>
        </div>
      </div>

      {/* Section label */}
      <div className="px-5 pb-2 pt-2 text-[10px] font-semibold uppercase tracking-[0.08em] text-slate-400">
        Workspace
      </div>

      {/* Nav */}
      <nav className="flex-1 space-y-0.5 px-3">
        {NAV.map((item) => {
          const active = item.key === page;
          return (
            <button
              key={item.key}
              onClick={() => onChange(item.key)}
              className={`group relative flex w-full items-center gap-3 rounded-md px-3 py-2 text-left transition-colors ${
                active
                  ? "bg-slate-900 text-white shadow-sm"
                  : "text-slate-600 hover:bg-slate-100 hover:text-slate-900"
              }`}
            >
              <span
                className={
                  active
                    ? "text-white"
                    : "text-slate-400 group-hover:text-slate-600"
                }
              >
                {item.icon}
              </span>
              <div className="flex-1 leading-tight">
                <div className="text-[13px] font-medium">{item.label}</div>
                <div
                  className={`text-[10px] ${
                    active ? "text-slate-300" : "text-slate-400"
                  }`}
                >
                  {item.hint}
                </div>
              </div>
            </button>
          );
        })}
      </nav>

      {/* Footer status */}
      <div className="border-t border-slate-200/80 px-5 py-4">
        <div className="flex items-center gap-2 text-[11px]">
          <span
            className={`relative h-1.5 w-1.5 rounded-full ${
              status === "open"
                ? "bg-emerald-500"
                : status === "connecting"
                  ? "bg-amber-500"
                  : "bg-red-500"
            }`}
          >
            {status === "open" && (
              <span className="absolute inset-0 animate-ping rounded-full bg-emerald-500 opacity-50" />
            )}
          </span>
          <span className="font-medium text-slate-700">
            {status === "open"
              ? "Connected"
              : status === "connecting"
                ? "Connecting"
                : "Disconnected"}
          </span>
        </div>
        <div className="mt-1 truncate font-mono text-[10px] text-slate-400">
          {sessionId.slice(0, 8)}
        </div>
      </div>
    </aside>
  );
}
