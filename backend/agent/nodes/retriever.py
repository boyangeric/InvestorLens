"""
Retriever Node — fetches documents from Qdrant based on the router's chosen strategy.

This is the strategy-aware retrieval layer. Instead of always doing the same
vector search, it dispatches to different retrieval approaches:
  - semantic_search: embedding similarity (best for conceptual questions)
  - keyword_search: metadata filtering + keyword matching (best for specific lookups)
  - direct_extract: fetches all chunks from a document (best for structured extraction)
  - compare: retrieves from multiple documents for comparison
"""

import logging
import os
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
    "direct_extract": 20,
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


def _keyword_search(query: str, top_k: int) -> list[dict]:
    """
    Keyword-aware search — uses embedding search but with a higher bar.
    """
    results = _semantic_search(query, top_k=top_k * 2)

    # Apply a stricter score threshold for keyword queries
    # The embedding captures the general concept of "revenue" and "2023" — but it might also match chunks about "income", "earnings", "fiscal year", or other financially-related text that doesn't actually contain those exact terms.
    # Short, specific keyword queries produce less distinctive embeddings compared to full natural language questions like "What was the company's total revenue in 2023?", which give the model more semantic signal to work with.
    filtered = [doc for doc in results if doc["score"] >= 0.3]
    return filtered[:top_k]


def _direct_extract(query: str, top_k: int) -> list[dict]:
    """
    Fetch a large number of chunks for extraction tasks.

    For queries like "extract all financial metrics", we need broad coverage
    rather than pinpoint relevance, so we retrieve more chunks.
    """
    return _semantic_search(query, top_k=top_k)


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
    "direct_extract": _direct_extract,
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
