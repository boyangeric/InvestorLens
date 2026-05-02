"""
Document endpoints — ingest, list, and inspect indexed PDFs.

Runs parse → chunk → embed → store inline (a real deployment would queue
this to a worker; here we keep dev UX synchronous).
"""

import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File

from backend.ingestion.chunker import chunk_pages
from backend.ingestion.embedder import (
    COLLECTION_NAME,
    get_qdrant_client,
    store_chunks,
)
from backend.ingestion.parser import parse_pdf

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("")
async def list_documents() -> dict:
    """
    Return every document currently indexed, with chunk + page counts.

    Scrolls the Qdrant collection (payload only — no vectors) and aggregates
    by `source`. Cheap for demo-scale corpora; for production you'd cache or
    maintain a dedicated documents table.
    """
    client = get_qdrant_client()

    # If the collection doesn't exist yet, return empty list rather than 500.
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION_NAME not in existing:
        return {"documents": [], "total_chunks": 0}

    by_source: dict[str, dict] = {}
    next_offset = None

    while True:
        points, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            with_payload=True,
            with_vectors=False,
            limit=256,
            offset=next_offset,
        )
        if not points:
            break

        for p in points:
            payload = p.payload or {}
            source = payload.get("source", "unknown")
            page = int(payload.get("page", 0))

            doc = by_source.setdefault(
                source, {"source": source, "chunks": 0, "pages": set()}
            )
            doc["chunks"] += 1
            doc["pages"].add(page)

        if next_offset is None:
            break

    documents = [
        {
            "source": d["source"],
            "chunks": d["chunks"],
            "pages": len(d["pages"]),
        }
        for d in sorted(by_source.values(), key=lambda x: x["source"])
    ]

    return {
        "documents": documents,
        "total_chunks": sum(d["chunks"] for d in documents),
    }


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)) -> dict:
    """
    Upload a PDF, parse it, chunk it, embed it, and store in Qdrant.

    Returns counts for quick UX feedback.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Write upload to a temp file so pdfplumber (which wants a path) can read it
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    try:
        logger.info("Ingesting: %s", file.filename)

        pages = parse_pdf(str(tmp_path))
        # Override source: the parser uses the temp filename, but we want
        # the original upload name so citations and filtering work correctly.
        for page in pages:
            page["source"] = file.filename

        chunks = chunk_pages(pages)
        store_chunks(chunks)

        logger.info(
            "Ingested '%s': %d pages → %d chunks",
            file.filename,
            len(pages),
            len(chunks),
        )

        return {
            "status": "ok",
            "filename": file.filename,
            "pages": len(pages),
            "chunks": len(chunks),
        }
    except Exception as e:
        logger.exception("Ingestion failed for %s", file.filename)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}") from e
    finally:
        tmp_path.unlink(missing_ok=True)
