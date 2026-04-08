# Architecture

## System Overview

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

## Agent Graph Flow

1. **Moderator** — Validates query safety and relevance
2. **Rewriter** — Resolves ambiguous/follow-up queries using chat history
3. **Router** — Classifies query into retrieval strategy (semantic/keyword/extract/compare)
4. **Retriever** — Executes the chosen strategy against Qdrant
5. **Grader** — Scores chunk relevance; loops back to rewriter if insufficient (max 2 retries)
6. **Generator** — Produces grounded answer with citations and confidence score
7. **Risk Flagger** — Identifies risk disclosures with human-in-the-loop approval

## Multi-Model Strategy

| Node | Model | Rationale |
|------|-------|-----------|
| Moderator, Rewriter, Router, Grader | GPT-4o-mini | Simple classification tasks, cost-efficient |
| Generator, Extractor, Compare, Risk Flagger | GPT-4o | Quality-critical outputs requiring accuracy |
