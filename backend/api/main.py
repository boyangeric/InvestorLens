"""
FastAPI entrypoint for InvestorLens.

Exposes:
  GET  /health                   — liveness probe
  POST /documents/upload         — ingest a PDF into Qdrant
  WS   /chat/ws/{session_id}     — stream agent execution in real time

Uses CORS to allow the Vite dev server (localhost:5173) to talk to the API.
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.chat import router as chat_router
from backend.api.documents import router as documents_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="InvestorLens API",
    description="Agentic financial document analyst",
    version="0.1.0",
)

# CORS — allow the React dev server during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router, prefix="/chat", tags=["chat"])
app.include_router(documents_router, prefix="/documents", tags=["documents"])


@app.get("/health")
async def health() -> dict:
    """Liveness probe — returns 200 if the app is running."""
    return {"status": "ok"}
