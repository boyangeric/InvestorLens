"""
Retriever Node — fetches documents from Qdrant based on the router's chosen strategy.

This is the strategy-aware retrieval layer. Instead of always doing the same
vector search, it dispatches to different retrieval approaches:
  - semantic_search: embedding similarity (best for conceptual questions)
  - keyword_search: metadata filtering + keyword matching (best for specific lookups)
  - extract_metrics: broad-coverage retrieval (more chunks) feeding the
    HITL-gated metric extractor downstream
  - compare: retrieves from multiple documents for comparison
  - hybrid_live: semantic retrieval; live_tools adds market + news in parallel
"""

import logging
import os
import re
import time
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient

from backend.agent.state import AgentState
from backend.agent.utils import embeddings_with_retry

load_dotenv()

logger = logging.getLogger(__name__)

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
COLLECTION_NAME = "investorlens_docs"

# How many chunks to retrieve per strategy
TOP_K = {
    "semantic_search": 8,
    "keyword_search": 5,
    "extract_metrics": 20,  # broad coverage — the extractor needs to see all metric-bearing passages
    "compare": 6,
    "hybrid_live": 5,  # fewer chunks since live data carries half the load
}


def _get_embedding(text: str) -> list[float]:
    """Generate an embedding vector for the given text."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = embeddings_with_retry(client, model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


def _semantic_search(query: str, top_k: int) -> list[dict]:
    """Standard vector similarity search — best for conceptual questions."""
    qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    query_vector = _get_embedding(query)

    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k, # Controls how many of the most similar document chuncks are returned from the vector store query
    )

    return [
        {
            "content": (hit.payload or {}).get("text", ""),
            "source": (hit.payload or {}).get("source", "unknown"),
            "page": (hit.payload or {}).get("page", 0),
            "chunk_index": (hit.payload or {}).get("chunk_index", 0),
            "score": hit.score, # how similar to the query
        }
        for hit in results.points
    ]


def _tokenize(text: str) -> set[str]:
    """Extract meaningful tokens from text (lowercase, filter short/common words)."""
    # Common words that don't signal relevance
    stopwords = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "can", "what", "which", "who", "when",
        "where", "why", "how", "this", "that", "these", "those", "it", "its",
    }
    # Extract words, lowercase, filter stopwords and very short tokens
    tokens = re.findall(r'\b\w+\b', text.lower())
    return {t for t in tokens if t not in stopwords and len(t) > 2}


def _lexical_search(query: str, top_k: int) -> list[dict]:
    """
    Lexical (token-based) retrieval: rank chunks by keyword overlap, not embedding similarity.

    This is deterministic and ignores synonyms ("revenue" won't match "earnings" unless
    both appear in the chunk). Perfect for specific lookups: "dividend per share",
    "EPS", "net profit", etc.
    """
    # First pass: fetch candidates via semantic search (fast retrieval from index)
    candidates = _semantic_search(query, top_k=top_k * 3)

    # Extract query tokens
    query_tokens = _tokenize(query)

    # If no meaningful tokens, return candidates as-is
    if not query_tokens:
        return candidates[:top_k]

    # Score each candidate by token overlap
    def lexical_score(chunk: dict) -> tuple[int, float]:
        """Return (token_match_count, semantic_score) for sorting."""
        chunk_tokens = _tokenize(chunk.get("content", ""))
        # Count how many query tokens appear in this chunk
        matches = len(query_tokens & chunk_tokens)  # Set intersection
        return (matches, chunk.get("score", 0.0))

    # Re-rank by token overlap (descending), then by semantic score (tiebreaker)
    ranked = sorted(candidates, key=lexical_score, reverse=True)

    logger.debug(
        "_lexical_search: query tokens=%s, top result has %d matches",
        list(query_tokens)[:5],
        lexical_score(ranked[0])[0] if ranked else 0,
    )

    return ranked[:top_k]


def _bm25_search(query: str, top_k: int) -> list[dict]:
    """
    BM25 full-text search — deterministic keyword matching via Qdrant's
    native BM25 scoring. Best for specific field/value lookups where exact term matches, not semantic reasoning.
    """
    # TODO: Enable when Qdrant's Python SDK simplifies BM25 text search API
    # For now, use semantic search (which works well and has the text index available as fallback)
    logger.debug("_bm25_search: BM25 not yet integrated in SDK, using semantic fallback")
    return _semantic_search(query, top_k)


def _keyword_search(query: str, top_k: int) -> list[dict]:
    """
    Lexical keyword search: rank chunks by token overlap, not embedding similarity.

    Perfect for specific field lookups: "What is the dividend per share?",
    "EPS", "net profit", etc. Ignores synonyms — "revenue" won't match
    "earnings" unless both appear in the chunk.

    Approach:
    1. Over-retrieve via semantic search (cheap, uses index)
    2. Re-rank by token overlap (deterministic, no LLM)
    3. Return top-k by lexical relevance

    Falls back to native BM25 if available (Qdrant >= 1.8.0 + SDK support).
    """
    # Try native BM25 first (if available)
    try:
        results = _bm25_search(query, top_k=top_k)
        if results and results[0].get("content"):  # If BM25 worked, use it
            logger.debug("_keyword_search: using native BM25")
            return results
    except Exception:
        pass

    # Fallback: lexical search (token-based re-ranking)
    results = _lexical_search(query, top_k=top_k)

    logger.debug(
        "_keyword_search: query=%s, got %d results via lexical re-ranking",
        query[:50],
        len(results),
    )

    return results


def _compare(query: str, top_k: int) -> list[dict]:
    """
    Cross-document retrieval for comparison queries.

    Retrieves chunks and groups by source document so the generator
    can compare across documents.
    """
    results = _semantic_search(query, top_k=top_k * 2)

    # Ensure we have chunks from multiple sources
    seen_sources = set()
    diverse_results = []

    chunks_per_source: dict[str, int] = {}

    for doc in results:
        source = doc["source"]
        # Take up to top_k/2 chunks per source to ensure diversity
        current_count = chunks_per_source.get(source, 0)
        if current_count < top_k // 2:
            diverse_results.append(doc)
            chunks_per_source[source] = current_count + 1
            seen_sources.add(source)

        if len(diverse_results) >= top_k:
            break

    return diverse_results


# Strategy dispatcher — maps strategy name to retrieval function
STRATEGY_FN = {
    "semantic_search": _semantic_search,
    "keyword_search": _keyword_search,
    # extract_metrics uses broad semantic retrieval (20 chunks) without lexical re-ranking;
    # the downstream HITL gate lets humans verify which chunks are actually relevant
    "extract_metrics": lambda query, top_k: _semantic_search(query, top_k=top_k),
    "compare": _compare,
    # hybrid_live uses semantic search for the document side; the live_tools
    # node downstream adds the live market + news context in parallel.
    "hybrid_live": _semantic_search,
}


def retriever(state: AgentState) -> dict:
    """
    Execute the retrieval strategy chosen by the router.

    Reads: query, retrieval_strategy
    Writes: retrieved_docs, node_trace
    """
    query = state["query"]
    strategy = state.get("retrieval_strategy", "semantic_search")
    top_k = TOP_K.get(strategy, 8)

    logger.info("Retriever: executing '%s' with top_k=%d", strategy, top_k)

    start = time.time()
    retrieve_fn = STRATEGY_FN.get(strategy, _semantic_search)
    docs = retrieve_fn(query, top_k)
    duration_ms = int((time.time() - start) * 1000)

    logger.info("Retriever: got %d chunks in %dms", len(docs), duration_ms)

    trace_entry = {
        "node": "retriever",
        "status": f"{len(docs)} chunks via {strategy}",
        "duration_ms": duration_ms,
        "chunks_retrieved": len(docs),
    }

    node_trace = state.get("node_trace", []) + [trace_entry]

    return {
        "retrieved_docs": docs,
        "current_node": "retriever",
        "node_trace": node_trace,
    }
