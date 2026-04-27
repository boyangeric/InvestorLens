"""
Document upload endpoint — ingests a PDF into Qdrant.

Runs the parse → chunk → embed → store pipeline in a background-safe way.
For a real deployment you'd queue this to a worker (Celery, RQ); here we run
it inline because the dev UX benefits from synchronous feedback.
"""

import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File

from backend.ingestion.chunker import chunk_pages
from backend.ingestion.embedder import store_chunks
from backend.ingestion.parser import parse_pdf

logger = logging.getLogger(__name__)

router = APIRouter()


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
