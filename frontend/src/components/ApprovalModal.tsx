/**
 * ApprovalModal — human-in-the-loop gate for disclosure analysis.
 *
 * Renders when the LangGraph hits `interrupt_before=["disclosure_analyzer"]`
 * and the backend sends `{type: "review_required"}`. The reviewer either
 * approves (resume) or rejects (abort with a reason).
 *
 * In a regulated financial setting this would also show the retrieved
 * source chunks for inspection — we surface the rewriter's reasoning as a
 * lighter-weight version of the same idea.
 */

import { useState } from "react";

import type { ReviewState } from "../hooks/useWebSocket";

type Props = {
  review: NonNullable<ReviewState>;
  onApprove: () => void;
  onReject: (reason: string) => void;
};

export function ApprovalModal({ review, onApprove, onReject }: Props) {
  const [rejectMode, setRejectMode] = useState(false);
  const [reason, setReason] = useState("");

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/40 backdrop-blur-sm">
      <div className="w-full max-w-lg rounded-xl bg-white p-6 shadow-2xl">
        <div className="mb-4 flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-full bg-amber-100">
            <span className="text-lg">⏸️</span>
          </div>
          <div>
            <h2 className="text-base font-semibold text-slate-900">
              Human review required
            </h2>
            <p className="text-xs text-slate-500">
              The agent has paused before running{" "}
              <span className="font-mono">{review.next_node}</span>
            </p>
          </div>
        </div>

        <dl className="mb-5 space-y-3 rounded-lg border border-slate-200 bg-slate-50 p-4 text-sm">
          <div>
            <dt className="text-xs font-semibold uppercase tracking-wide text-slate-500">
              Query
            </dt>
            <dd className="mt-1 text-slate-900">{review.query}</dd>
          </div>
          {review.reasoning && (
            <div>
              <dt className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                Routing reasoning
              </dt>
              <dd className="mt-1 text-slate-700">{review.reasoning}</dd>
            </div>
          )}
        </dl>

        {rejectMode ? (
          <div className="space-y-3">
            <textarea
              autoFocus
              value={reason}
              onChange={(e) => setReason(e.target.value)}
              placeholder="Reason for rejection (shown to the user)…"
              rows={2}
              className="w-full resize-none rounded-lg border border-slate-300 px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
            />
            <div className="flex justify-end gap-2">
              <button
                onClick={() => setRejectMode(false)}
                className="rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-sm font-medium text-slate-700 hover:bg-slate-50"
              >
                Back
              </button>
              <button
                onClick={() =>
                  onReject(reason.trim() || "Query rejected by reviewer.")
                }
                className="rounded-lg bg-red-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-red-700"
              >
                Confirm rejection
              </button>
            </div>
          </div>
        ) : (
          <div className="flex justify-end gap-2">
            <button
              onClick={() => setRejectMode(true)}
              className="rounded-lg border border-slate-300 bg-white px-4 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50"
            >
              Reject
            </button>
            <button
              onClick={onApprove}
              className="rounded-lg bg-emerald-600 px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-emerald-700"
            >
              Approve & continue
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
