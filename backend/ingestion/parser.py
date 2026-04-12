"""
PDF Parser — extracts text from financial documents using pdfplumber.

Each page is returned as a separate dict with page number and text content,
preserving page boundaries for downstream citation (e.g., "Source: page 5").
"""

import logging
from pathlib import Path

import pdfplumber

logger = logging.getLogger(__name__)


def parse_pdf(file_path: str) -> list[dict]:
    """
    Extract text from a PDF file, returning one entry per page.

    Args:
        file_path: Path to the PDF file.

    Returns:
        List of dicts: [{"page": 1, "text": "...", "source": "filename.pdf"}, ...]
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    filename = path.name
    pages: list[dict] = []

    logger.info("Parsing PDF: %s", filename)

    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            text = text.strip()

            if not text:
                logger.debug("Page %d is empty, skipping", i + 1)
                continue

            pages.append({
                "page": i + 1,
                "text": text,
                "source": filename,
            })

    logger.info("Extracted %d non-empty pages from %s", len(pages), filename)
    return pages
