"""
Eval metrics — pure functions that grade a single agent run against expectations.

Each function takes:
  * `actual`   — a dict snapshot of the final AgentState plus a `trace` list of
                 node trace entries (i.e. what the eval runner captured).
  * `expected` — the `expected` block from a test_questions.json entry.

Each function returns either:
  * a tuple `(passed: bool, detail: str)` for binary metrics, or
  * a numeric value (e.g. latency_ms).

Returning a `detail` string makes failures self-explanatory in the eval report
so a reviewer can see *why* a question failed without re-running it.
"""

from __future__ import annotations

import re
from typing import Any

# Citation regex — matches [Source: filename.pdf, Page 42] or [Source: filename, Page 42]
# Tolerant of optional whitespace and case.
CITATION_RE = re.compile(
    r"\[Source:\s*(?P<source>[^,\]]+),\s*Page\s*(?P<page>\d+)\s*\]",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trace_nodes(actual: dict[str, Any]) -> list[str]:
    """Names of every node that fired in this run, in execution order."""
    return [entry.get("node", "") for entry in actual.get("trace", [])]


def _moderator_decision(actual: dict[str, Any]) -> str:
    """Read the moderator's pass/block decision out of the trace."""
    for entry in actual.get("trace", []):
        if entry.get("node") == "moderator":
            return entry.get("status", "")
    return "missing"


def _router_strategy(actual: dict[str, Any]) -> str | None:
    """The strategy the adaptive router chose, or None if the router didn't fire."""
    for entry in actual.get("trace", []):
        if entry.get("node") == "router":
            return entry.get("status")
    return None


def _extract_citations(answer: str) -> list[tuple[str, int]]:
    """Pull every (source, page) tuple out of the answer text."""
    out: list[tuple[str, int]] = []
    for m in CITATION_RE.finditer(answer or ""):
        out.append((m.group("source").strip(), int(m.group("page"))))
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def moderation_correct(actual: dict, expected: dict) -> tuple[bool, str]:
    """Did the moderator block/pass as expected?"""
    if "moderator" not in expected:
        return True, "skipped"
    want = expected["moderator"]
    got = _moderator_decision(actual)
    return got == want, f"want={want}, got={got}"


def routing_correct(actual: dict, expected: dict) -> tuple[bool, str]:
    """
    Did the router pick a strategy from the allowed list?

    Multiple strategies are acceptable for a single question — e.g. "what was
    underlying profit?" can reasonably be answered by either keyword_search or
    semantic_search. The fixture gives an allow-list, not a single answer.
    """
    if "strategy_in" not in expected:
        return True, "skipped"
    allowed = set(expected["strategy_in"])
    got = _router_strategy(actual)
    if got is None:
        return False, "router did not fire"
    return got in allowed, f"want one of {sorted(allowed)}, got={got}"


def citation_grounding(actual: dict, expected: dict) -> tuple[bool, str]:
    """
    Every [Source: X, Page Y] in the answer must point to a chunk that was
    actually retrieved. This catches the citation-hallucination failure mode
    where the LLM invents a citation for a fact pulled from training data.

    Only checked when the answer is grounded. Only checks (source, page)
    membership — not whether the cited chunk literally contains the claim
    (that's the faithfulness audit's job).
    """
    if not expected.get("grounded", False):
        return True, "skipped (not a grounded question)"

    answer = actual.get("generation", "")
    citations = _extract_citations(answer)
    if not citations:
        # Grounded answer with zero citations is a separate failure mode
        # caught by `min_citations_met`. Skip here.
        return True, "no citations to check"

    relevant_docs = actual.get("relevant_docs", []) or []
    retrieved_pairs = {
        (str(d.get("source", "")), int(d.get("page", 0)))
        for d in relevant_docs
    }

    invalid = [c for c in citations if c not in retrieved_pairs]
    if invalid:
        return False, f"{len(invalid)}/{len(citations)} citations not in relevant_docs: {invalid[:3]}"
    return True, f"all {len(citations)} citations grounded"


def min_citations_met(actual: dict, expected: dict) -> tuple[bool, str]:
    """The answer must contain at least N distinct (source, page) citations."""
    if "min_citations" not in expected:
        return True, "skipped"
    want = expected["min_citations"]
    citations = set(_extract_citations(actual.get("generation", "")))
    return len(citations) >= want, f"want>={want}, got={len(citations)}"


def faithfulness_passed(actual: dict, expected: dict) -> tuple[bool, str]:
    """
    Did the post-gen faithfulness audit pass? Only meaningful for grounded
    answers — general-knowledge answers skip the audit by design.
    """
    if not expected.get("faithful", False):
        return True, "skipped"

    # Find the faithfulness trace entry
    for entry in actual.get("trace", []):
        if entry.get("node") == "faithfulness":
            verdict = entry.get("status", "")
            return verdict == "faithful", f"verdict={verdict}"

    # No faithfulness entry means generator wasn't reached (e.g. fell back
    # before generation). For a question expecting `faithful: true`, that's a
    # failure — we expected a grounded path.
    return False, "faithfulness node did not fire"


def grounded_correct(actual: dict, expected: dict) -> tuple[bool, str]:
    """Did the run produce a grounded answer when expected (and vice versa)?"""
    if "grounded" not in expected:
        return True, "skipped"
    want = expected["grounded"]
    got = bool(actual.get("grounded", True))
    return want == got, f"want grounded={want}, got grounded={got}"


def fallback_correct(actual: dict, expected: dict) -> tuple[bool, str]:
    """For out-of-scope questions, verify the expected fallback node fired."""
    if "fallback_node" not in expected:
        return True, "skipped"
    want = expected["fallback_node"]
    nodes = _trace_nodes(actual)
    if want in nodes:
        return True, f"{want} fired"
    return False, f"want {want}, trace={nodes}"


def latency_ms(actual: dict) -> int:
    """Total run wall-clock — sum of per-node duration_ms in the trace."""
    return sum(int(e.get("duration_ms", 0)) for e in actual.get("trace", []))


# ---------------------------------------------------------------------------
# Aggregator — runs all binary metrics for one question
# ---------------------------------------------------------------------------

BINARY_METRICS = {
    "moderation": moderation_correct,
    "routing": routing_correct,
    "grounded": grounded_correct,
    "min_citations": min_citations_met,
    "citation_grounding": citation_grounding,
    "faithfulness": faithfulness_passed,
    "fallback": fallback_correct,
}


def grade_question(actual: dict, expected: dict) -> dict[str, Any]:
    """
    Run every metric for one question. Returns a dict with per-metric results
    plus an overall pass flag (= all non-skipped metrics passed).
    """
    results: dict[str, Any] = {"metrics": {}, "latency_ms": latency_ms(actual)}

    overall = True
    for name, fn in BINARY_METRICS.items():
        passed, detail = fn(actual, expected)
        results["metrics"][name] = {"passed": passed, "detail": detail}
        # Skipped metrics return passed=True by convention, so this still works.
        if not passed:
            overall = False

    results["passed"] = overall
    return results
