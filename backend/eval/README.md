# Evaluation Suite

End-to-end evaluation of the InvestorLens agent against a fixed question set,
with a markdown report and a pass-rate gate suitable for CI.

This is a **behavioural** eval, not a unit-test suite — every question runs
the full LangGraph (moderator → rewriter → router → retriever → grader →
generator → faithfulness, with the CRAG fallback to `general_generator`).
Real LLM calls; no mocks.

## Why this exists

Banks and enterprise buyers don't ship RAG without numbers. A demo answers
one question well; an eval suite proves the system behaves correctly across
the failure modes that matter in production:

- prompt injection,
- citation hallucination,
- corpus misses,
- routing errors,
- regression after prompt edits.

Each of the five categories below maps to a specific design decision in the
agent graph, so the eval doubles as a justification for the architecture.

## Question categories

| Category | What it tests | Graph branch |
|---|---|---|
| `factual` | Direct lookups against the corpus | grader → generator → faithfulness |
| `comparative` | Cross-document analysis | router=`compare`, multi-source citations |
| `extraction` | Structured extraction over many chunks | router=`direct_extract` |
| `adversarial` | Prompt-injection / off-domain | moderator must BLOCK |
| `out_of_scope` | Corpus miss → general-knowledge fallback | grader < threshold → `general_generator` |

## Metrics

Pure functions in `metrics.py`. Each is computed per question; aggregates are
shown in the report.

| Metric | What it checks | Why it matters |
|---|---|---|
| `moderation` | Adversarial queries are blocked, on-topic queries pass | First-line defence; cheap to test, high signal |
| `routing` | Adaptive router picks an allowed strategy | Validates the "adaptive" claim — without this, it's just generic RAG |
| `grounded` | `grounded` flag matches expectation | Ensures CRAG fallback fires for corpus misses |
| `min_citations` | Answer has ≥ N distinct `[Source: …, Page …]` cites | Catches "looks confident, no sources" answers |
| `citation_grounding` | Every cited (source, page) is in `relevant_docs` | Catches **citation hallucination** — fact from training data with a real cite slapped on |
| `faithfulness` | Post-gen audit returns `faithful` | Independent check that load-bearing claims are supported by the chunks |
| `fallback` | Out-of-scope queries route to `general_generator` | Validates the second CRAG gate end-to-end |

A question passes iff every non-skipped metric passes. Metrics that don't
apply to a question (e.g. `routing` for a blocked adversarial query) are
skipped, not failed.

## Running it

From the project root, with backend deps installed:

```bash
# 1. Make sure Qdrant is up and the sample corpus is ingested
docker compose up -d qdrant
.venv/bin/uvicorn backend.api.main:app --port 8000 &
curl -F "file=@sample_docs/03061502.pdf" http://localhost:8000/documents/upload

# 2. Run the eval
.venv/bin/python -m backend.eval.run_eval --verbose
```

Optional flags:

```
--questions PATH       Override fixture file
--threshold FLOAT      Pass-rate gate (default 0.8)
--output PATH          Save the markdown report to a file
--filter CATEGORY      Run only one category
--verbose              Per-question progress to stderr
```

## Cost & runtime

Each question runs the full graph — typically 5–10 LLM calls (router,
grader-per-chunk, generator, faithfulness). With the current 13-question
fixture and `gpt-4o-mini` for graders + `gpt-4o` for generation, expect
roughly **\$0.10–0.20 and 60–120s** per full run at 2026 prices.

A full run is cheap enough to use as a CI gate on PRs that touch prompts or
graph edges.

## Extending the suite

Add a question by appending to `test_questions.json`:

```json
{
  "id": "factual_004",
  "category": "factual",
  "query": "What is the gearing ratio?",
  "expected": {
    "moderator": "pass",
    "strategy_in": ["keyword_search", "semantic_search"],
    "grounded": true,
    "min_citations": 1,
    "faithful": true
  }
}
```

Skip any field you don't want to assert — missing fields are skipped, not
failed. New metrics go in `metrics.py`; register them in `BINARY_METRICS` and
they appear in the per-metric report automatically.

## Future work

- Hook to LangSmith so each eval run produces a comparable dataset run, with
  prompt versions baked into the run name.
- Add a "regression" mode that diffs against a previous report and fails if
  any individual question went from PASS to FAIL.
- Add token-cost-per-question to the report (`tokens_in`/`tokens_out` is
  already captured per node — sum and price it).
