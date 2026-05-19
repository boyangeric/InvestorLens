"""
Pydantic response schemas for every LLM-using node.

These are passed to OpenAI's Structured Outputs API (`response_format=Schema`)
so the model's JSON output is constrained by the schema at decode time. Invalid
JSON, missing fields, and unknown enum values become impossible at the API
boundary — nodes can therefore use typed attribute access (`result.confidence`)
instead of defensive `.get("confidence", 0.0)` calls.

Constraints honoured for Structured Outputs:
  - Every field is required (no Optional defaults at schema level)
  - additionalProperties is forbidden (OpenAI strict mode default)
  - Enums are expressed as Literal[...] — translates to JSON schema `enum`
  - No `minimum`/`maximum`/`pattern` — Structured Outputs ignores those;
    bounds (e.g. confidence in [0, 1]) are enforced by the prompt + a runtime
    clamp where it matters
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    """Base for every response model — forbids unknown fields."""

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Moderator
# ---------------------------------------------------------------------------
class ModeratorResponse(_Strict):
    decision: Literal["pass", "block"]
    reason: str


# ---------------------------------------------------------------------------
# Rewriter
# ---------------------------------------------------------------------------
class RewriterResponse(_Strict):
    rewritten_query: str
    was_rewritten: bool
    # Set when the query is genuinely vague AND chat history offers no
    # resolvable subject. The rewriter ends the turn with `clarification_question`
    # instead of continuing to the router — same short-circuit pattern as the
    # moderator's `block` decision. `clarification_question` is empty when
    # needs_clarification is False (Structured Outputs requires all fields).
    needs_clarification: bool
    clarification_question: str


# ---------------------------------------------------------------------------
# Adaptive Router
# ---------------------------------------------------------------------------
RouterStrategy = Literal[
    "semantic_search",
    "keyword_search",
    "direct_extract",
    "compare",
    "extract_metrics",
    "hybrid_live",
]


class RouterResponse(_Strict):
    strategy: RouterStrategy
    reasoning: str


# ---------------------------------------------------------------------------
# Grader
# ---------------------------------------------------------------------------
class GraderResponse(_Strict):
    # Three-way verdict instead of a forced binary. "uncertain" gives the
    # model a syntactic way to say "I can't confidently decide" — without it,
    # Structured Outputs forces a coin-flipped yes/no and the trace looks
    # confident even when the model wasn't. The grader node treats
    # "uncertain" as "do not ground on this chunk" (same effect as "no") but
    # counts it separately in the trace so the signal isn't lost.
    relevant: Literal["yes", "no", "uncertain"]
    reason: str


# ---------------------------------------------------------------------------
# Generator (grounded answer)
# ---------------------------------------------------------------------------
class GeneratorResponse(_Strict):
    answer: str
    # Confidence is unconstrained at the schema level (Structured Outputs
    # doesn't support numeric bounds). The prompt asks for [0, 1] and the
    # node clamps on read.
    confidence: float
    reasoning: str
    sources: list[str]


# ---------------------------------------------------------------------------
# General Generator (CRAG fallback)
# ---------------------------------------------------------------------------
class GeneralGeneratorResponse(_Strict):
    answer: str
    reasoning: str


# ---------------------------------------------------------------------------
# Faithfulness Auditor
# ---------------------------------------------------------------------------
class FaithfulnessResponse(_Strict):
    verdict: Literal["faithful", "unsupported"]
    unsupported_claims: list[str]
    reasoning: str


# ---------------------------------------------------------------------------
# Metric Extractor (HITL gate — the human verifies these numbers before they
# propagate into a comparison or grounded answer)
# ---------------------------------------------------------------------------
class ExtractedMetric(_Strict):
    name: str = Field(description='Standardised metric name, e.g. "revenue", "net_profit_margin"')
    value: str = Field(description='Exact figure as it appears in the document, e.g. "$52.3 billion", "23.4%"')
    period: str = Field(description='Reporting period the figure applies to, e.g. "FY2024", "HY26"')
    source: str = Field(description="Source filename")
    page: int = Field(description="1-based page number where the figure was found")


class MetricExtractorResponse(_Strict):
    document: str
    period: str
    metrics: list[ExtractedMetric]
    key_highlights: list[str]
