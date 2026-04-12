"""
serve.py
Phase 3: FastAPI server exposing the RAG pipeline as a REST API.

Resume connection (Uber):
'Developed RESTful APIs and data processing pipelines using Python'
Same FastAPI pattern - endpoint validates input, calls pipeline, returns JSON.
"""

import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from src.pipeline.rag_pipeline import (
    load_vectorstore, build_llm, build_rag_chain, query_rag
)

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# Global chain variable - loaded once at startup
_chain = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load RAG chain when server starts. Release when server stops."""
    global _chain
    logger.info("Loading RAG chain at startup...")
    vs     = load_vectorstore()
    llm    = build_llm()
    _chain = build_rag_chain(vs, llm)
    logger.info("RAG chain ready. Server accepting requests.")
    yield
    logger.info("Server shutting down.")


app = FastAPI(
    title="Uber RAG Pipeline API",
    description="Production RAG system for Uber support documentation",
    version="1.0.0",
    lifespan=lifespan
)


# ── Request / Response schemas ────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Question to answer from Uber documentation"
    )
    top_k: int = Field(
        default=4,
        ge=1,
        le=10,
        description="Number of chunks to retrieve"
    )


class SourceDoc(BaseModel):
    source:   str
    category: str
    snippet:  str


class QueryResponse(BaseModel):
    question:   str
    answer:     str
    sources:    list[SourceDoc]
    latency_ms: float
    top_k_used: int


# ── Endpoints ─────────────────────────────────────────────────────────
@app.get("/health")
def health_check():
    """Liveness probe — Kubernetes calls this every 30s."""
    return {
        "status":       "healthy",
        "chain_loaded": _chain is not None
    }


@app.get("/info")
def info():
    """Return metadata about the running RAG system."""
    return {
        "model":       "sentence-transformers/all-MiniLM-L6-v2",
        "llm":         "google/flan-t5-base",
        "collection":  "uber_support_docs",
        "api_version": "1.0.0"
    }


@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    """
    Main RAG query endpoint.
    POST a question, receive answer + source citations + latency.
    """
    if _chain is None:
        raise HTTPException(status_code=503, detail="RAG chain not loaded")

    result = query_rag(_chain, request.question, verbose=False)

    return QueryResponse(
        question=result["question"],
        answer=result["answer"],
        sources=[SourceDoc(**s) for s in result["sources"]],
        latency_ms=result["latency_ms"],
        top_k_used=request.top_k
    )