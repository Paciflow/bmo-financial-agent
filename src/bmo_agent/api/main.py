from __future__ import annotations

import io
import json
import logging
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response

from bmo_agent.agent.graph import BMOAgent
from bmo_agent.agent.llm import LLMNotConfiguredError
from bmo_agent.api.schemas import (
    ChatRequest,
    ChatResponse,
    DocumentStatus,
    RetrievedPassage,
    RetrieveRequest,
    RetrieveResponse,
)
from bmo_agent.config import get_settings
from bmo_agent.retrieval.hybrid import HybridRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="BMO Financial Intelligence Agent", version="0.1.0")

settings = get_settings()


@lru_cache(maxsize=1)
def _get_retriever() -> HybridRetriever:
    return HybridRetriever(settings)


@lru_cache(maxsize=1)
def _get_agent() -> BMOAgent:
    return BMOAgent(settings, retriever=_get_retriever())


@app.get("/health")
def health():
    checks = {
        "indexes_present": (settings.processed_dir / "chunks.json").exists() and (settings.processed_dir / "headings.json").exists(),
        "anthropic_configured": settings.anthropic_configured,
        "s3_configured": settings.s3_configured,
    }
    status = "ok" if checks["indexes_present"] else "degraded"
    return {"status": status, **checks}


@app.get("/documents", response_model=list[DocumentStatus])
def list_documents():
    return [_document_status(settings.document_id)]


@app.get("/documents/{document_id}/status", response_model=DocumentStatus)
def document_status(document_id: str):
    if document_id != settings.document_id:
        raise HTTPException(status_code=404, detail="Unknown document_id")
    return _document_status(document_id)


def _document_status(document_id: str) -> DocumentStatus:
    meta_path = settings.raw_dir / f"{document_id}.ingestion.json"
    ingested = meta_path.exists()
    s3_uploaded = None
    if ingested:
        meta = json.loads(meta_path.read_text())
        s3_uploaded = meta.get("s3_uploaded")

    pages_path = settings.processed_dir / "raw_pages.json"
    headings_path = settings.processed_dir / "headings.json"
    chunks_path = settings.processed_dir / "chunks.json"
    tables_path = settings.processed_dir / "tables.json"

    pages = len(json.loads(pages_path.read_text())) if pages_path.exists() else None
    headings = len(json.loads(headings_path.read_text())) if headings_path.exists() else None
    chunks = len(json.loads(chunks_path.read_text())) if chunks_path.exists() else None
    tables = len(json.loads(tables_path.read_text())) if tables_path.exists() else None

    return DocumentStatus(
        document_id=document_id,
        ingested=ingested,
        pages=pages,
        headings=headings,
        chunks=chunks,
        tables=tables,
        s3_uploaded=s3_uploaded,
        indexed=chunks_path.exists(),
    )


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        agent = _get_agent()
    except LLMNotConfiguredError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    try:
        result = agent.ask(req.question)
    except LLMNotConfiguredError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return ChatResponse(**result)


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest):
    retriever = _get_retriever()
    result = retriever.retrieve(req.query, use_rerank=req.use_rerank)
    evidence = [
        RetrievedPassage(
            chunk_id=e.chunk_id,
            text=e.text,
            content_type=e.content_type,
            heading_path=e.heading_path,
            page=e.page_index + 1,
            table_id=e.table_id,
            dense_rank=e.dense_rank,
            bm25_rank=e.bm25_rank,
            rrf_score=e.rrf_score,
            rerank_score=e.rerank_score,
        )
        for e in result.evidence
    ]
    return RetrieveResponse(query=result.query, evidence=evidence, diagnostics=result.diagnostics)


@app.get("/sources/{chunk_id}")
def get_source(chunk_id: str):
    retriever = _get_retriever()
    chunk = retriever.chunks_by_id.get(chunk_id)
    if chunk is None:
        raise HTTPException(status_code=404, detail="Unknown chunk_id")
    return {
        "chunk_id": chunk.chunk_id,
        "content_type": chunk.content_type,
        "heading_path": chunk.heading_path,
        "page": chunk.page_index + 1,
        "table_id": chunk.table_id,
        "text": chunk.text,
    }


@app.get("/sources/{chunk_id}/page")
def get_source_page(chunk_id: str, dpi: int = 150):
    """Render the PDF page this chunk came from as a PNG, for the Streamlit source
    preview. Bounded to this document only -- no arbitrary file paths accepted.
    """
    retriever = _get_retriever()
    chunk = retriever.chunks_by_id.get(chunk_id)
    if chunk is None:
        raise HTTPException(status_code=404, detail="Unknown chunk_id")

    import pymupdf

    pdf_path = settings.raw_dir / f"{settings.document_id}.pdf"
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="Source PDF not found on this server")

    doc = pymupdf.open(str(pdf_path))
    if not (0 <= chunk.page_index < doc.page_count):
        raise HTTPException(status_code=404, detail="Page index out of range")
    page = doc[chunk.page_index]
    pix = page.get_pixmap(dpi=dpi)
    buf = io.BytesIO(pix.tobytes("png"))
    return Response(content=buf.getvalue(), media_type="image/png")
