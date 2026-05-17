"""
External-data tools — fetched live at query time, not from the document corpus.

These complement the offline retriever by giving the agent a way to answer
questions about *current* state (prices, market cap, recent headlines) that
a static PDF corpus cannot. The agent's adaptive router decides per-query
whether to call them.

Each tool is a pure function returning a Pydantic model so the generator can
weave the result into an answer with structured provenance ("🌐 live data
from yfinance" vs "📄 from report p.12").
"""
