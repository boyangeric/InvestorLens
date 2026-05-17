/**
 * ApprovalModal — human-in-the-loop gate for high-stakes metric extraction.
 *
 * The graph pauses at `interrupt_before=["finalize_extraction"]`. By the time
 * this modal renders, the figures have already been extracted — the
 * reviewer's job is to verify them before they propagate into downstream
 * comparison/report generation.
 *
 * Three primary actions (matches the spec):
 *   - Approve all  — figures pass through unchanged, tagged "verified"
 *   - Edit values  — each metric becomes an input; corrected values are
 *                    written back to state and tagged "edited"
 *   - Skip         — proceed without verifying; output is tagged
 *                    "unverified" and the chat panel renders a warning
 *
 * Reject (abort entirely with a reason) is kept as a secondary action — it's
 * a structurally distinct operation from skip (halt vs proceed-with-warning).
 */

import { useState } from "react";

import type { ExtractedMetric, ReviewState } from "../hooks/useWebSocket";

type Props = {
  review: NonNullable<ReviewState>;
  onApprove: () => void;
  onApproveEdits: (metrics: ExtractedMetric[]) => void;
  onSkip: () => void;
  onReject: (reason: string) => void;
};

type Mode = "review" | "edit" | "reject";

export function ApprovalModal({
  review,
  onApprove,
  onApproveEdits,
  onSkip,
  onReject,
}: Props) {
  const [mode, setMode] = useState<Mode>("review");
  const [edited, setEdited] = useState<ExtractedMetric[]>(
    review.extracted_metrics,
  );
  const [reason, setReason] = useState("");

  const updateMetric = (i: number, patch: Partial<ExtractedMetric>) => {
    setEdited((prev) =>
      prev.map((m, idx) => (idx === i ? { ...m, ...patch } : m)),
    );
  };

  const hasMetrics = review.extracted_metrics.length > 0;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/40 p-6 backdrop-blur-sm">
      <div className="flex max-h-[90vh] w-full max-w-2xl flex-col rounded-xl bg-white shadow-2xl">
        <div className="border-b border-slate-200 px-6 py-4">
          <div className="flex items-start gap-3">
            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-amber-100">
              <span className="text-lg">⏸️</span>
            </div>
            <div>
              <h2 className="text-base font-semibold text-slate-900">
                {mode === "edit"
                  ? "Edit extracted figures"
                  : mode === "reject"
                    ? "Reject extraction"
                    : "Verify extracted figures"}
              </h2>
              <p className="mt-0.5 text-xs text-slate-500">
                {mode === "edit"
                  ? "Correct any wrong values. Empty fields will reset to the original."
                  : "One wrong number can move investment decisions. Verify before these figures flow into the final answer."}
              </p>
            </div>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto px-6 py-4">
          <dl className="mb-4 space-y-2.5 rounded-lg border border-slate-200 bg-slate-50 p-3.5 text-sm">
            <div>
              <dt className="text-[11px] font-semibold uppercase tracking-wide text-slate-500">
                Query
              </dt>
              <dd className="mt-0.5 text-slate-900">{review.query}</dd>
            </div>
            {review.reasoning && (
              <div>
                <dt className="text-[11px] font-semibold uppercase tracking-wide text-slate-500">
                  Routing reasoning
                </dt>
                <dd className="mt-0.5 text-slate-700">{review.reasoning}</dd>
              </div>
            )}
          </dl>

          {mode === "reject" ? (
            <div className="space-y-3">
              <textarea
                autoFocus
                value={reason}
                onChange={(e) => setReason(e.target.value)}
                placeholder="Reason for rejection (shown to the user)…"
                rows={3}
                className="w-full resize-none rounded-lg border border-slate-300 px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
              />
            </div>
          ) : hasMetrics ? (
            <div className="overflow-hidden rounded-lg border border-slate-200">
              <div className="border-b border-slate-200 bg-slate-50 px-3 py-2 text-[11px] font-semibold uppercase tracking-wide text-slate-500">
                Extracted metrics ({review.extracted_metrics.length})
              </div>
              {mode === "edit" ? (
                <div className="divide-y divide-slate-100">
                  {edited.map((m, i) => (
                    <MetricEditRow
                      key={i}
                      metric={m}
                      onChange={(patch) => updateMetric(i, patch)}
                    />
                  ))}
                </div>
              ) : (
                <table className="w-full text-sm">
                  <thead className="bg-white text-left text-[11px] uppercase tracking-wide text-slate-500">
                    <tr>
                      <th className="px-3 py-2 font-medium">Metric</th>
                      <th className="px-3 py-2 font-medium">Value</th>
                      <th className="px-3 py-2 font-medium">Period</th>
                      <th className="px-3 py-2 font-medium">Page</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {review.extracted_metrics.map((m, i) => (
                      <tr key={i} className="hover:bg-slate-50">
                        <td className="px-3 py-2 font-medium text-slate-800">
                          {m.name}
                        </td>
                        <td className="px-3 py-2 font-mono tabular-nums text-slate-900">
                          {m.value}
                        </td>
                        <td className="px-3 py-2 text-slate-600">{m.period}</td>
                        <td className="px-3 py-2 font-mono text-slate-500">
                          p.{m.page}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          ) : (
            <div className="rounded-lg border border-amber-200 bg-amber-50 p-3 text-xs text-amber-800">
              No metrics were extracted from the retrieved chunks. You can
              still skip (proceed unverified) or reject this extraction.
            </div>
          )}
        </div>

        <div className="flex items-center justify-between gap-2 border-t border-slate-200 bg-slate-50 px-6 py-3">
          {mode === "review" ? (
            <>
              <button
                onClick={() => setMode("reject")}
                className="text-xs font-medium text-slate-500 hover:text-red-600"
              >
                Reject extraction
              </button>
              <div className="flex gap-2">
                <button
                  onClick={onSkip}
                  className="rounded-lg border border-slate-300 bg-white px-3.5 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50"
                >
                  Skip
                </button>
                <button
                  onClick={() => setMode("edit")}
                  disabled={!hasMetrics}
                  className="rounded-lg border border-slate-300 bg-white px-3.5 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  Edit values
                </button>
                <button
                  onClick={onApprove}
                  disabled={!hasMetrics}
                  className="rounded-lg bg-emerald-600 px-3.5 py-2 text-sm font-medium text-white shadow-sm hover:bg-emerald-700 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  Approve all
                </button>
              </div>
            </>
          ) : mode === "edit" ? (
            <>
              <button
                onClick={() => {
                  setEdited(review.extracted_metrics);
                  setMode("review");
                }}
                className="text-xs font-medium text-slate-500 hover:text-slate-900"
              >
                Cancel edits
              </button>
              <div className="flex gap-2">
                <button
                  onClick={() => setMode("review")}
                  className="rounded-lg border border-slate-300 bg-white px-3.5 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50"
                >
                  Back
                </button>
                <button
                  onClick={() => onApproveEdits(edited)}
                  className="rounded-lg bg-indigo-600 px-3.5 py-2 text-sm font-medium text-white shadow-sm hover:bg-indigo-700"
                >
                  Apply edits
                </button>
              </div>
            </>
          ) : (
            <>
              <span />
              <div className="flex gap-2">
                <button
                  onClick={() => setMode("review")}
                  className="rounded-lg border border-slate-300 bg-white px-3.5 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50"
                >
                  Back
                </button>
                <button
                  onClick={() =>
                    onReject(reason.trim() || "Figures rejected by reviewer.")
                  }
                  className="rounded-lg bg-red-600 px-3.5 py-2 text-sm font-medium text-white shadow-sm hover:bg-red-700"
                >
                  Confirm rejection
                </button>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function MetricEditRow({
  metric,
  onChange,
}: {
  metric: ExtractedMetric;
  onChange: (patch: Partial<ExtractedMetric>) => void;
}) {
  return (
    <div className="grid grid-cols-12 gap-2 px-3 py-2.5">
      <div className="col-span-4">
        <label className="text-[10px] font-medium uppercase tracking-wide text-slate-500">
          Metric
        </label>
        <div className="mt-0.5 truncate text-[12px] font-medium text-slate-800">
          {metric.name}
        </div>
      </div>
      <div className="col-span-4">
        <label className="text-[10px] font-medium uppercase tracking-wide text-slate-500">
          Value
        </label>
        <input
          type="text"
          value={metric.value}
          onChange={(e) => onChange({ value: e.target.value })}
          className="mt-0.5 w-full rounded border border-slate-300 px-2 py-1 font-mono text-[12px] tabular-nums text-slate-900 focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
        />
      </div>
      <div className="col-span-2">
        <label className="text-[10px] font-medium uppercase tracking-wide text-slate-500">
          Period
        </label>
        <input
          type="text"
          value={metric.period}
          onChange={(e) => onChange({ period: e.target.value })}
          className="mt-0.5 w-full rounded border border-slate-300 px-2 py-1 text-[12px] text-slate-700 focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
        />
      </div>
      <div className="col-span-2">
        <label className="text-[10px] font-medium uppercase tracking-wide text-slate-500">
          Page
        </label>
        <input
          type="number"
          min={1}
          value={metric.page}
          onChange={(e) =>
            onChange({ page: parseInt(e.target.value, 10) || metric.page })
          }
          className="mt-0.5 w-full rounded border border-slate-300 px-2 py-1 font-mono text-[12px] text-slate-700 focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
        />
      </div>
    </div>
  );
}
