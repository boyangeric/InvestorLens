#!/usr/bin/env python
"""
Re-ingest sample documents from sample_docs/ directory.

After running reset_qdrant.py, run this to upload all sample PDFs.
This will recreate the collection with text indexing enabled.

Run from project root: .venv/bin/python backend/scripts/reingest_samples.py
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.ingestion.parser import parse_pdf
from backend.ingestion.chunker import chunk_pages
from backend.ingestion.embedder import store_chunks

SAMPLE_DOCS_DIR = Path(__file__).parent.parent.parent / "sample_docs"


def main():
    if not SAMPLE_DOCS_DIR.exists():
        print(f"Sample docs directory not found: {SAMPLE_DOCS_DIR}")
        return

    pdf_files = list(SAMPLE_DOCS_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {SAMPLE_DOCS_DIR}")
        return

    print(f"Found {len(pdf_files)} PDF file(s). Re-ingesting...\n")

    for pdf_path in pdf_files:
        print(f"Ingesting: {pdf_path.name}")

        try:
            pages = parse_pdf(str(pdf_path))
            # Use original filename for source
            for page in pages:
                page["source"] = pdf_path.name

            chunks = chunk_pages(pages)
            count = store_chunks(chunks)

            print(f"  ✓ {len(pages)} pages → {count} chunks\n")
        except Exception as e:
            print(f"  ✗ Failed: {e}\n")

    print("Re-ingestion complete. Collection now has text indexing enabled for BM25.")


if __name__ == "__main__":
    main()
