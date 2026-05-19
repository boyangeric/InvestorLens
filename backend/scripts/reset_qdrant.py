#!/usr/bin/env python
"""
Reset Qdrant collection to enable text indexing for BM25 search.

This script:
1. Deletes the existing 'investorlens_docs' collection (if it exists)
2. The next document ingest will recreate it with text indexing enabled

Run after Phase 2 code changes, before uploading documents.
"""

import os
import sys
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = "investorlens_docs"


def main():
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    # Check if collection exists
    collections = [c.name for c in client.get_collections().collections]

    if COLLECTION_NAME in collections:
        print(f"Deleting collection '{COLLECTION_NAME}'...")
        client.delete_collection(collection_name=COLLECTION_NAME)
        print("✓ Collection deleted.")
        print("\nNext time you upload a document, the collection will be recreated")
        print("with text indexing enabled for BM25 search.")
    else:
        print(f"Collection '{COLLECTION_NAME}' does not exist. Nothing to delete.")
        sys.exit(1)


if __name__ == "__main__":
    main()
