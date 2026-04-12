"""
Embedder — converts text chunks into vectors and stores them in Qdrant.

Uses OpenAI text-embedding-ada-002 for embeddings and Qdrant as the
vector store. Each chunk is stored with its full metadata (source, page,
chunk_index) so retrieval results can produce proper citations.
"""

import logging
import os
import uuid

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

load_dotenv()

logger = logging.getLogger(__name__)

# OpenAI embedding config
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL","text-embedding-3-small")
EMBEDDING_DIMENSION = 1536 

# Qdrant config
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = "investorlens_docs"

# Batch size for embedding API calls (ada-002 supports up to 2048 inputs)
EMBED_BATCH_SIZE = 100


def get_openai_client() -> OpenAI:
    """Create an OpenAI client for embeddings."""
    return OpenAI(api_key=OPENAI_API_KEY)


def get_qdrant_client() -> QdrantClient:
    """Create a Qdrant client."""
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def ensure_collection(client: QdrantClient) -> None:
    """Create the Qdrant collection if it doesn't exist."""
    collections = [c.name for c in client.get_collections().collections]

    if COLLECTION_NAME not in collections:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=EMBEDDING_DIMENSION,
                distance=Distance.COSINE,
            ),
        )
        logger.info("Created Qdrant collection: %s", COLLECTION_NAME)
    else:
        logger.info("Qdrant collection already exists: %s", COLLECTION_NAME)


def embed_texts(client: OpenAI, texts: list[str]) -> list[list[float]]:
    """
    Embed a list of texts in batches using OpenAI.

    Args:
        client: OpenAI client.
        texts: List of text strings to embed.

    Returns:
        List of embedding vectors (each is a list of 1536 floats).
    """
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        logger.info("Embedding batch %d–%d of %d", i, i + len(batch), len(texts))

        response = client.embeddings.create(
            input=batch,
            model=EMBEDDING_MODEL,
        )
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    return all_embeddings


def store_chunks(chunks: list[dict]) -> int:
    """
    Embed chunks and store them in Qdrant with metadata.

    This is the main entry point — call it with the output of chunker.chunk_pages().

    Args:
        chunks: List of chunk dicts from chunker: [{"text", "source", "page", "chunk_index"}]

    Returns:
        Number of chunks stored.
    """
    if not chunks:
        logger.warning("No chunks to store")
        return 0

    openai_client = get_openai_client()
    qdrant_client = get_qdrant_client()

    # Ensure collection exists
    ensure_collection(qdrant_client)

    # Generate embeddings
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embed_texts(openai_client, texts)

    # Build Qdrant points with metadata
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={
                "text": chunk["text"],
                "source": chunk["source"],
                "page": chunk["page"],
                "chunk_index": chunk["chunk_index"],
            },
        )
        for chunk, embedding in zip(chunks, embeddings)
    ]

    # Upsert into Qdrant in batches
    UPSERT_BATCH_SIZE = 100
    for i in range(0, len(points), UPSERT_BATCH_SIZE):
        batch = points[i : i + UPSERT_BATCH_SIZE]
        qdrant_client.upsert(collection_name=COLLECTION_NAME, points=batch)

    logger.info("Stored %d chunks in Qdrant collection '%s'", len(points), COLLECTION_NAME)
    return len(points)
