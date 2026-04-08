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

```bash
# 1. Clone and configure
cp .env.example .env
# Fill in your Azure OpenAI and LangSmith credentials

# 2. Start Qdrant
docker compose up -d

# 3. Install backend dependencies
cd backend
pip install -r requirements.txt

# 4. Install frontend dependencies
cd frontend
npm install

# 5. Run backend
uvicorn backend.main:app --reload

# 6. Run frontend
cd frontend
npm run dev
```

## Architecture

See [docs/architecture.md](docs/architecture.md) for the full system design.
