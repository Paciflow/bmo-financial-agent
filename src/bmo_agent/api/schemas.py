from __future__ import annotations

from pydantic import BaseModel


class ChatRequest(BaseModel):
    question: str


class Citation(BaseModel):
    chunk_id: str
    page: int
    heading_path: str | None
    content_type: str
    excerpt: str


class ChatResponse(BaseModel):
    answer: str
    response_type: str
    category: str | None
    citations: list[Citation]
    retrieval_diagnostics: dict
    latency_ms: float


class RetrieveRequest(BaseModel):
    query: str
    use_rerank: bool = True


class RetrievedPassage(BaseModel):
    chunk_id: str
    text: str
    content_type: str
    heading_path: str | None
    page: int
    table_id: str | None
    dense_rank: int | None
    bm25_rank: int | None
    rrf_score: float
    rerank_score: float | None


class RetrieveResponse(BaseModel):
    query: str
    evidence: list[RetrievedPassage]
    diagnostics: dict


class DocumentStatus(BaseModel):
    document_id: str
    ingested: bool
    pages: int | None
    headings: int | None
    chunks: int | None
    tables: int | None
    s3_uploaded: bool | None
    indexed: bool
