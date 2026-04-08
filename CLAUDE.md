# InvestorLens

## Project Overview

InvestorLens is an agentic financial document analyst. Users upload financial documents (annual reports, earnings transcripts, ETF factsheets, fund PDS documents) and interact with them through an intelligent single-agent system that decides *how* to answer each query — semantic search, keyword search, structured extraction, or multi-document comparison.

The agent is built with LangGraph, exposed as an MCP server, includes human-in-the-loop for risk assessments, and has a React frontend showing the agent's reasoning path in real time.

## Architecture

```
React Frontend (Chat + Agent Trace + Info Panel)
        ↕ WebSocket / REST
FastAPI Backend
        ↕
LangGraph StateGraph (single agent, multiple tools)
  Nodes: Moderator → Rewriter → Router → Retriever → Grader → Generator → Self-Assessment
  Tools: search_docs, extract_financials, compare_docs, flag_risks
        ↕                    ↕
   Qdrant (vectors)    Azure OpenAI (GPT-4o + GPT-4o-mini)
```

## Tech Stack

- **LLM**: Azure OpenAI — GPT-4o for generation/extraction, GPT-4o-mini for routing/grading/moderation
- **Orchestration**: LangGraph (StateGraph with conditional edges and cycles)
- **RAG Framework**: LangChain
- **Vector Store**: Qdrant (local via Docker)
- **Embeddings**: Azure OpenAI text-embedding-ada-002
- **Backend**: FastAPI with WebSocket support
- **Frontend**: React + TypeScript + Tailwind CSS + shadcn/ui
- **MCP**: MCP Python SDK (official)
- **Observability**: LangSmith (free tier)
- **Deployment**: Docker Compose locally, optionally Azure Container Apps
- **PDF Parsing**: pdfplumber

## Coding Conventions

- **Python**: 3.11+, type hints everywhere, Pydantic for all data models
- **Async**: Use async/await for all LLM calls and API endpoints
- **Prompts**: Store in `/backend/prompts/` as YAML files, never hardcode in Python
- **Error handling**: LLM calls wrapped in try/except, structured output parsing has fallback recovery
- **Logging**: Use Python logging module, log every node entry/exit with timing
- **Testing**: Each node should be independently testable with mock LLM responses
- **Environment variables**: All secrets (Azure keys, LangSmith key) via .env file, never committed
