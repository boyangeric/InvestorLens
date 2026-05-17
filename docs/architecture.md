# Architecture

## System Overview

```
React Frontend (Chat + Agent Trace + Sources + Approval Modal)
        ↕ WebSocket (one long-lived connection per session)
FastAPI Backend
        ↕
LangGraph StateGraph  ── checkpointed (MemorySaver, thread_id == session_id)
        ↕                        ↕                        ↕
   Qdrant (vectors)         OpenAI (GPT-4o + GPT-4o-mini)   Live tools:
                                                              - yfinance (quotes)
                                                              - Tavily (news)

Side channel:
   MCP server exposes 5 tools (search_docs, get_market_quote, search_news,
   extract_financials, compare_docs) to any MCP client (Claude Desktop, VS Code).
```

## Agent Graph Flow

```
Moderator ─(blocked)──────────────────────────────────────────→ END
    │
    │(continue)
    ↓
Rewriter → Adaptive Router → Retriever
                                  │
            ┌─────────────────────┼──────────────────────────────┐
            │(extract_metrics)    │(hybrid_live)                 │(otherwise)
            ↓                     ↓                              ↓
       Metric Extractor       Live Tools                       Grader ──(retry)──→ Rewriter
            │                 (yfinance ∥ Tavily)                │
            ↓ [INTERRUPT]         │                              │(generate)        (corpus_miss)
       Finalize Extraction        ↓                              ↓                       ↓
            ↓                  Generator ─(skip_audit)─→ END   Generator → Faithfulness ─(ok)─→ END
           END                                                                  │
                                                                                │(unsupported)
                                                                                ↓
                                                                          General Generator → END
```

### Nodes

| # | Node | Model | Role |
|---|------|-------|------|
| 1 | **Moderator** | gpt-4o-mini | Safety/relevance guardrail before any retrieval spend. Blocks prompt-injection and off-topic queries; sets `generation` to a rejection message that short-circuits the rest of the graph. |
| 2 | **Rewriter** | gpt-4o-mini | Resolves coreferences and ambiguities against chat history so retrieval sees a self-contained query. |
| 3 | **Adaptive Router** | gpt-4o-mini | Classifies the query into one of 5 strategies: `semantic_search`, `keyword_search`, `direct_extract`, `compare`, `extract_metrics`, or `hybrid_live`. |
| 4 | **Retriever** | — | Strategy-aware dispatch over Qdrant. Different strategies use different top-k and ranking. |
| 5 | **Grader** | gpt-4o-mini | Scores chunk relevance. Drives the CRAG loop: retry the rewriter (up to 2 times), fall back to general knowledge, or proceed to the generator. |
| 6 | **Generator** | gpt-4o | Produces the grounded answer via OpenAI Structured Outputs — `answer + confidence + reasoning + sources` in one call. Citations use `[Source: file.pdf, Page X]` for docs and `[Live: yfinance, as_of YYYY-MM-DD]` / `[Live: domain.com, published YYYY-MM-DD]` for live tools. |
| 7 | **Faithfulness** | gpt-4o-mini | LLM-as-judge audit. Checks each claim against retrieved chunks; if unsupported, routes to general_generator instead of returning a misleading grounded answer. **Skipped on hybrid_live** — live-tool outputs aren't in any chunk by design. |
| 8 | **General Generator** | gpt-4o | Fallback for corpus misses and unsupported claims. Returns a clearly-labelled general-knowledge answer with no false citations. |
| 9 | **Metric Extractor** | gpt-4o | Pulls structured financial figures + page citations. Triggers the HITL interrupt before they propagate. |
| 10 | **Finalize Extraction** | — | Re-renders `generation` from the (possibly edited) extracted metrics after the reviewer approves/edits/skips. |
| 11 | **Live Tools** | gpt-4o-mini (planner) + tool calls | OpenAI parallel tool calling. Planner returns ≥1 tool call (`tool_choice="required"`, `parallel_tool_calls=True`); a `ThreadPoolExecutor` fans them out so wall-time is `max(yfinance, Tavily)`, not the sum. |

### Two CRAG-style fallback gates protect grounded answers

1. **Grader fallback** — after retries, if `len(relevant_docs) < MIN_RELEVANT`, skip the generator entirely and route to `general_generator`.
2. **Faithfulness fallback** — after generation, audit each claim against the cited chunks. Unsupported claims (citation hallucination) route to `general_generator` so the user sees a clearly-labelled general-knowledge answer rather than a misleading grounded one.

### Human-in-the-loop

The `extract_metrics` branch is HITL-gated via `interrupt_before=["finalize_extraction"]`. The pause is **after** `metric_extractor` runs, so the modal has the actual numbers to show the reviewer. The reviewer can:

- **Approve** — resume; `verification_status = "verified"`
- **Approve with edits** — overwrite `extracted_metrics`, resume; `verification_status = "edited"`
- **Skip** — resume but stamp `verification_status = "skipped"` so the frontend renders an unverified badge
- **Reject** — overwrite `generation` with a rejection message and don't resume

The `hybrid_live` branch is **deliberately not** HITL-gated. Live tool outputs are structured Pydantic models tagged with source + timestamp — provenance is verifiable by construction. Live data is also read-only and ephemeral; there's no irreversible downstream decision like there is for an extracted figure feeding a comparison. Different risk profile → different gating policy.

### State checkpointing

`MemorySaver` persists state per `thread_id` (== session_id). This is what makes the HITL pause/resume work across the WebSocket — and what makes multi-turn chat history coherent. The trade-off: every ephemeral per-query field must be explicitly reset on a new query, or stale state from the previous turn leaks into routing decisions (e.g. `should_continue_after_moderator` checks `state.get("generation")` to detect a moderator block).

## Multi-Model Strategy

| Tier | Nodes | Model | Rationale |
|------|-------|-------|-----------|
| Classification / audit | Moderator, Rewriter, Adaptive Router, Grader, Faithfulness, Live-tools planner | gpt-4o-mini | Cheap, low-latency, structured outputs are enough for these tasks. |
| Generation / extraction | Generator, General Generator, Metric Extractor | gpt-4o | Quality-critical; users see these outputs directly. |
| Embeddings | Ingestion | text-embedding-3-small | Strong recall at 1/5 the cost of `-large`. |

## Security: untrusted-data wrapping

All retrieved content is wrapped in typed tags before being shown to the LLM:

- `<chunk>` — document chunks from Qdrant
- `<market_quote>` — yfinance results
- `<news_article>` — Tavily results

The generator's system prompt explicitly states these tags contain **data, not instructions**. Closing tags are canonicalised (any `</CHUNK>` inside untrusted content is rewritten to `</_chunk>`) so injected payloads can't escape the wrapper.

## Observability

Each node records a trace entry with model, duration, tokens in/out, and cost — streamed over the WebSocket as the graph executes. The frontend renders this as a live trace and aggregates session-to-date tokens and USD cost from the trace alone, no extra backend plumbing.

LangSmith tracing is wired but optional (`LANGSMITH_API_KEY=` leaves it disabled).
