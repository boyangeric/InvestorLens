"""
Evaluation runner — runs the full agent graph against test_questions.json,
applies metrics, and emits a markdown report.

Usage (from project root):

    .venv/bin/python -m backend.eval.run_eval

Optional flags:
    --questions PATH      Override fixture file (default: backend/eval/test_questions.json)
    --threshold FLOAT     Pass-rate gate, exit non-zero if below (default: 0.8)
    --output PATH         Write the markdown report to a file (default: stdout only)
    --filter CATEGORY     Run only one category (factual | adversarial | …)
    --verbose             Print per-question details inline

Prerequisites:
    1. Backend env vars set (.env with OPENAI_API_KEY)
    2. Qdrant running and the sample corpus ingested:
         docker compose up -d qdrant
         curl -F "file=@sample_docs/03061502.pdf" http://localhost:8000/documents/upload
    3. Run from the project root, not from inside backend/eval/

Exit codes:
    0   pass-rate ≥ threshold
    1   pass-rate < threshold, or fixture / setup error
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

from backend.agent.graph import app as agent_graph
from backend.agent.utils import aggregate_usage
from backend.eval.metrics import BINARY_METRICS, grade_question

logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_FIXTURE = PROJECT_ROOT / "backend" / "eval" / "test_questions.json"


# ---------------------------------------------------------------------------
# Run one question through the graph
# ---------------------------------------------------------------------------

async def run_one(question: dict) -> dict[str, Any]:
    """
    Invoke the agent graph for a single question and return a dict snapshot
    of the final state plus a flat `trace` list (matches metrics.py contract).

    Each question runs in its own thread_id so state never bleeds between
    cases. We catch all exceptions so a single bad question can't tank the
    suite — they're reported as failures.
    """
    initial_state = {
        "query": question["query"],
        "original_query": question["query"],
        "chat_history": [],
        "retry_count": 0,
        "node_trace": [],
    }
    config = {"configurable": {"thread_id": f"eval-{uuid.uuid4()}"}}

    start = time.perf_counter()
    try:
        # ainvoke runs to completion (or to an interrupt). Disclosure queries
        # would pause here — none in the current fixture, but we'd handle
        # them by checking get_state(config).next.
        final_state = await agent_graph.ainvoke(initial_state, config=config)
    except Exception as e:
        logger.exception("Question %s crashed", question["id"])
        return {
            "error": str(e),
            "trace": [],
            "generation": "",
            "grounded": True,  # default — metrics will see no fallback fired
            "duration_s": time.perf_counter() - start,
            "usage": aggregate_usage([]),
        }
    duration_s = time.perf_counter() - start

    trace = final_state.get("node_trace", [])
    return {
        "trace": trace,
        "generation": final_state.get("generation", ""),
        "grounded": final_state.get("grounded", True),
        "relevant_docs": final_state.get("relevant_docs", []),
        "retrieved_docs": final_state.get("retrieved_docs", []),
        "duration_s": duration_s,
        "usage": aggregate_usage(trace),
    }


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def _emoji(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def build_report(results: list[dict], threshold: float) -> tuple[str, bool]:
    """
    Build a markdown report from per-question results.

    Returns (report, gate_passed). `gate_passed` is True iff the overall
    pass rate is ≥ threshold.
    """
    total = len(results)
    passed_count = sum(1 for r in results if r["grade"]["passed"])
    pass_rate = passed_count / total if total else 0.0
    gate_passed = pass_rate >= threshold

    lines: list[str] = []
    lines.append("# InvestorLens — Evaluation Report\n")
    lines.append(f"- **Questions:** {total}")
    lines.append(f"- **Passed:** {passed_count} ({pass_rate:.0%})")
    lines.append(f"- **Threshold:** {threshold:.0%}")
    lines.append(f"- **Gate:** {'PASS' if gate_passed else 'FAIL'}\n")

    # Per-category breakdown
    by_category: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_category[r["question"]["category"]].append(r)

    lines.append("## By category\n")
    lines.append("| Category | Pass | Total | Rate |")
    lines.append("|---|---:|---:|---:|")
    for cat in sorted(by_category):
        rs = by_category[cat]
        p = sum(1 for r in rs if r["grade"]["passed"])
        lines.append(f"| {cat} | {p} | {len(rs)} | {p / len(rs):.0%} |")
    lines.append("")

    # Per-metric pass rate (only counting non-skipped runs)
    lines.append("## By metric\n")
    lines.append("| Metric | Pass | Checked | Rate |")
    lines.append("|---|---:|---:|---:|")
    for metric_name in BINARY_METRICS:
        checked = 0
        passed = 0
        for r in results:
            entry = r["grade"]["metrics"].get(metric_name, {})
            if entry.get("detail", "").startswith("skipped"):
                continue
            checked += 1
            if entry.get("passed"):
                passed += 1
        rate = (passed / checked) if checked else 1.0
        lines.append(f"| {metric_name} | {passed} | {checked} | {rate:.0%} |")
    lines.append("")

    # Latency stats
    durations = sorted(r["duration_s"] for r in results)
    if durations:
        p50 = durations[len(durations) // 2]
        p95 = durations[min(len(durations) - 1, int(len(durations) * 0.95))]
        lines.append("## Latency\n")
        lines.append(f"- p50: {p50:.2f}s")
        lines.append(f"- p95: {p95:.2f}s")
        lines.append(f"- max: {durations[-1]:.2f}s\n")

    # Cost — totals + per-node breakdown across the whole run
    total_cost = sum(r["actual"].get("usage", {}).get("cost_usd", 0.0) for r in results)
    total_in = sum(r["actual"].get("usage", {}).get("tokens_in", 0) for r in results)
    total_out = sum(r["actual"].get("usage", {}).get("tokens_out", 0) for r in results)
    if total_cost or total_in:
        avg = total_cost / total if total else 0.0
        lines.append("## Cost\n")
        lines.append(f"- **Total:** ${total_cost:.4f} ({total_in:,} in / {total_out:,} out)")
        lines.append(f"- **Per question (avg):** ${avg:.4f}\n")

        # Per-node aggregate across the full run
        by_node: dict[str, dict] = {}
        for r in results:
            for node, b in r["actual"].get("usage", {}).get("by_node", {}).items():
                agg = by_node.setdefault(
                    node, {"calls": 0, "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0}
                )
                agg["calls"] += b["calls"]
                agg["tokens_in"] += b["tokens_in"]
                agg["tokens_out"] += b["tokens_out"]
                agg["cost_usd"] += b["cost_usd"]

        if by_node:
            lines.append("| Node | Calls | Tokens in | Tokens out | Cost |")
            lines.append("|---|---:|---:|---:|---:|")
            for node, b in sorted(by_node.items(), key=lambda kv: -kv[1]["cost_usd"]):
                lines.append(
                    f"| {node} | {b['calls']} | {b['tokens_in']:,} | {b['tokens_out']:,} | ${b['cost_usd']:.4f} |"
                )
            lines.append("")

    # Question-by-question
    lines.append("## Questions\n")
    for r in results:
        q = r["question"]
        g = r["grade"]
        status = _emoji(g["passed"])
        lines.append(f"### {status} `{q['id']}` ({q['category']})")
        lines.append(f"> {q['query']}")
        if "error" in r["actual"]:
            lines.append(f"\n**Crashed:** `{r['actual']['error']}`\n")
            continue
        lines.append("")
        lines.append("| Metric | Result | Detail |")
        lines.append("|---|---|---|")
        for name, m in g["metrics"].items():
            mark = _emoji(m["passed"]) if not m["detail"].startswith("skipped") else "skip"
            lines.append(f"| {name} | {mark} | {m['detail']} |")
        lines.append(f"| latency | — | {r['duration_s']:.2f}s |")
        usage = r["actual"].get("usage", {})
        if usage.get("cost_usd"):
            lines.append(
                f"| cost | — | ${usage['cost_usd']:.4f} "
                f"({usage['tokens_in']:,} in / {usage['tokens_out']:,} out, {usage['calls']} calls) |"
            )
        excerpt = r["actual"].get("generation", "")[:200].replace("\n", " ")
        if excerpt:
            lines.append(f"\n_Answer excerpt:_ {excerpt}{'…' if len(r['actual']['generation']) > 200 else ''}\n")

    return "\n".join(lines), gate_passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main_async(args: argparse.Namespace) -> int:
    fixture_path = Path(args.questions)
    if not fixture_path.exists():
        print(f"ERROR: fixture not found: {fixture_path}", file=sys.stderr)
        return 1

    data = json.loads(fixture_path.read_text())
    questions = data.get("questions", [])
    if args.filter:
        questions = [q for q in questions if q["category"] == args.filter]

    if not questions:
        print("ERROR: no questions to run (check --filter)", file=sys.stderr)
        return 1

    print(f"Running {len(questions)} questions…", file=sys.stderr)
    results: list[dict] = []

    for i, q in enumerate(questions, 1):
        print(f"  [{i}/{len(questions)}] {q['id']}: {q['query'][:60]}…", file=sys.stderr)
        actual = await run_one(q)
        grade = grade_question(actual, q.get("expected", {}))
        results.append({
            "question": q,
            "actual": actual,
            "grade": grade,
            "duration_s": actual.get("duration_s", 0.0),
        })
        if args.verbose:
            print(f"      → {_emoji(grade['passed'])} ({actual.get('duration_s', 0):.1f}s)", file=sys.stderr)

    report, gate_passed = build_report(results, args.threshold)
    print(report)
    if args.output:
        Path(args.output).write_text(report)
        print(f"\nReport written to {args.output}", file=sys.stderr)

    return 0 if gate_passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Run InvestorLens evaluation suite.")
    parser.add_argument("--questions", default=str(DEFAULT_FIXTURE), help="Fixture path")
    parser.add_argument("--threshold", type=float, default=0.8, help="Pass-rate gate (0.0–1.0)")
    parser.add_argument("--output", help="Write report to this file in addition to stdout")
    parser.add_argument("--filter", help="Only run this category")
    parser.add_argument("--verbose", action="store_true", help="Per-question progress to stderr")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING,  # quiet — eval prints its own structured output
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
