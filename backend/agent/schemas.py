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


# ---------------------------------------------------------------------------
# Adaptive Router
# ---------------------------------------------------------------------------
RouterStrategy = Literal[
    "semantic_search",
    "keyword_search",
    "direct_extract",
    "compare",
    "analyze_disclosures",
]


class RouterResponse(_Strict):
    strategy: RouterStrategy
    reasoning: str


# ---------------------------------------------------------------------------
# Grader
# ---------------------------------------------------------------------------
class GraderResponse(_Strict):
    # 1 = relevant, 0 = not relevant. Literal[0, 1] becomes integer + enum
    # in the JSON schema, so the API cannot return e.g. "yes" or 2.
    relevant: Literal[0, 1]
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
# Disclosure Analyzer
# ---------------------------------------------------------------------------
RiskSeverity = Literal["high", "medium", "low"]


class DisclosedRisk(_Strict):
    title: str
    severity: RiskSeverity
    summary: str
    passage: str
    source: str
    page: int = Field(description="1-based page number in the source document")


class DisclosureAnalyzerResponse(_Strict):
    risks: list[DisclosedRisk]
    overall_risk_assessment: str
