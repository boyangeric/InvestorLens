"""
Chunker — splits parsed pages into overlapping text chunks for embedding.

Uses LangChain's RecursiveCharacterTextSplitter which splits on natural
boundaries (paragraphs → sentences → words) rather than blindly at character N.
Overlap ensures context isn't lost at chunk boundaries.
"""

import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# Defaults tuned for financial documents:
# - 1000 chars ≈ ~200 tokens (fits well within ada-002's context)
# - 200 char overlap keeps sentences intact across boundaries
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200


def chunk_pages(
    pages: list[dict],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict]:
    """
    Split parsed pages into overlapping chunks, preserving metadata.

    Args:
        pages: Output from parser.parse_pdf — [{"page", "text", "source"}]
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Characters of overlap between consecutive chunks.

    Returns:
        List of dicts: [{"text": "...", "source": "file.pdf", "page": 3, "chunk_index": 0}, ...]
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[dict] = []
    global_index = 0

    for page in pages:
        page_chunks = splitter.split_text(page["text"])

        for text in page_chunks:
            chunks.append({
                "text": text,
                "source": page["source"],
                "page": page["page"],
                "chunk_index": global_index,
            })
            global_index += 1

    logger.info(
        "Chunked %d pages into %d chunks (size=%d, overlap=%d)",
        len(pages), len(chunks), chunk_size, chunk_overlap,
    )
    return chunks
