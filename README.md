# InvestorLens

An agentic financial document analyst powered by LangGraph, Azure OpenAI, and Qdrant.

Upload financial documents (annual reports, earnings transcripts, ETF factsheets) and interact with them through an intelligent agent that adaptively chooses the best retrieval strategy for each query.

## Features

- **Adaptive RAG**: Agent selects semantic search, keyword search, structured extraction, or multi-document comparison per query
- **LangGraph Agent**: StateGraph with conditional routing, cyclic self-correction, and human-in-the-loop
- **MCP Server**: Tools exposed via Model Context Protocol for interoperability
- **Real-time Trace**: Frontend shows the agent's reasoning path as it executes
- **Multi-model Routing**: GPT-4o for generation, GPT-4o-mini for routing/grading (cost optimization)

## Quick Start

### First-time setup

```bash
# 1. Configure secrets
cp .env.example .env
# Fill in your Azure OpenAI and LangSmith credentials

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

End-to-end behavioural eval suite — 13 questions across 5 categories
(factual, comparative, extraction, adversarial, out-of-scope) with 7 binary
metrics including citation grounding and faithfulness. CI-hookable via a
pass-rate threshold.

```bash
# After ingesting the sample corpus
.venv/bin/python -m backend.eval.run_eval --verbose
```

See [backend/eval/README.md](backend/eval/README.md) for details.
