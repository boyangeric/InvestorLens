# InvestorLens

An agentic financial document analyst powered by LangGraph, OpenAI, and Qdrant.

Upload financial documents (annual reports, earnings transcripts, ETF factsheets) and interact with them through an intelligent agent that adaptively chooses the best retrieval strategy for each query — and reaches out to live market and news data when the question needs current state the report can't answer.

## Features

- **Adaptive RAG**: Agent selects semantic search, keyword search, structured extraction, multi-document comparison, or hybrid-live per query
- **Live tool calling**: OpenAI native parallel tool calling (`parallel_tool_calls=True`) fans out to yfinance + Tavily news concurrently, so wall-time is `max(latencies)` not the sum
- **Human-in-the-loop**: LangGraph `interrupt_after` on the metric extractor — analysts approve / edit / skip extracted figures before they propagate, and the verdict is stamped on every answer for audit
- **Citation provenance**: Distinct `[Source: file.pdf, Page X]` vs `[Live: yfinance, as_of YYYY-MM-DD]` markers so readers can tell document-grounded claims from current external state
- **Faithfulness audit**: Post-generation LLM-as-judge check on doc-grounded answers; skipped on hybrid-live branches whose Pydantic-typed tool outputs are verifiable by construction
- **MCP server**: 5 tools (`search_docs`, `get_market_quote`, `search_news`, `extract_financials`, `compare_docs`) exposed via Model Context Protocol — usable from Claude Desktop, VS Code, any MCP client
- **Real-time trace**: Frontend streams each LangGraph node as it fires, with per-node tokens / cost / latency and session-to-date spend
- **Multi-model routing**: GPT-4o for generation/extraction, GPT-4o-mini for moderation/routing/grading/tool-planning (cost optimisation)

## Quick Start

### First-time setup

```bash
# 1. Configure secrets
cp .env.example .env
# Fill in OPENAI_API_KEY (required) and LANGSMITH_API_KEY (optional, for tracing).
# TAVILY_API_KEY is optional — leave blank to disable the news tool;
# the agent will still answer doc-only and hybrid-quote-only questions.

# 2. Backend dependencies (one-off)
python -m venv .venv
.venv/bin/pip install -r backend/requirements.txt

# 3. Frontend dependencies (one-off)
cd frontend && npm install && cd ..
```

### Running the project

Run each command in its own terminal from the project root

```bash
# 1. Qdrant (vector DB)
docker compose up -d qdrant

# 2. Backend (FastAPI + LangGraph, :8000)
.venv/bin/uvicorn backend.api.main:app --port 8000 --reload

# 3. Frontend (Vite, :5173)
cd frontend && npm run dev
```

Then open **http://localhost:5173**.

To stop: `Ctrl-C` in the backend and frontend terminals, then
`docker compose down`.

## Architecture

See [docs/architecture.md](docs/architecture.md) for the full system design.

## Evaluation

End-to-end behavioural eval suite — 16 questions across 6 categories
(factual, comparative, extraction, adversarial, out-of-scope, hybrid-live)
with binary metrics including citation grounding, faithfulness, and
expected routing strategy. CI-hookable via a pass-rate threshold.

```bash
# After ingesting the sample corpus
.venv/bin/python -m backend.eval.run_eval --verbose
```

See [backend/eval/README.md](backend/eval/README.md) for details.
