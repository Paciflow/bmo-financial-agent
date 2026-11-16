from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from functools import lru_cache

from qdrant_client import QdrantClient

from bmo_agent.config import Settings, get_settings
from bmo_agent.embeddings.embedder import Embedder
from bmo_agent.indexing.bm25_index import load_bm25_index, tokenize
from bmo_agent.indexing.qdrant_index import get_client
from bmo_agent.models import Chunk, TableBlock
from bmo_agent.retrieval.rrf import rrf_fuse_scores
from bmo_agent.retrieval.section_match import SectionMatcher

logger = logging.getLogger(__name__)


@dataclass
class EvidencePassage:
    chunk_id: str
    text: str
    content_type: str
    heading_id: str | None
    heading_path: str | None
    page_index: int
    table_id: str | None
    dense_rank: int | None = None
    bm25_rank: int | None = None
    rrf_score: float = 0.0
    rerank_score: float | None = None


@dataclass
class RetrievalResult:
    query: str
    evidence: list[EvidencePassage]
    diagnostics: dict = field(default_factory=dict)


@lru_cache(maxsize=1)
def _load_chunks_by_id(settings_key: str) -> dict[str, Chunk]:
    settings = get_settings()
    chunks = [Chunk(**c) for c in json.loads((settings.processed_dir / "chunks.json").read_text())]
    return {c.chunk_id: c for c in chunks}


@lru_cache(maxsize=1)
def _load_tables_by_id(settings_key: str) -> dict[str, TableBlock]:
    settings = get_settings()
    path = settings.processed_dir / "tables.json"
    if not path.exists():
        return {}
    tables = [TableBlock(**t) for t in json.loads(path.read_text())]
    return {t.table_id: t for t in tables}


class HybridRetriever:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.embedder = Embedder(self.settings)
        self.client: QdrantClient = get_client(self.settings)
        self.bm25, self.bm25_chunk_ids = load_bm25_index(self.settings)
        self.chunks_by_id = _load_chunks_by_id("v1")
        self.tables_by_id = _load_tables_by_id("v1")
        self.section_matcher = SectionMatcher(self.settings)
        self._cross_encoder = None

    def _dense_search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        vec = self.embedder.embed_one(query, is_query=True)
        hits = self.client.query_points(
            collection_name=self.settings.qdrant_chunks_collection,
            query=vec,
            limit=top_k,
        ).points
        return [(h.payload["chunk_id"], h.score) for h in hits]

    def _bm25_search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        tokens = tokenize(query)
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(zip(self.bm25_chunk_ids, scores), key=lambda x: -x[1])[:top_k]
        return [(cid, float(score)) for cid, score in ranked if score > 0]

    def _rrf_fuse(self, dense: list[tuple[str, float]], bm25: list[tuple[str, float]], k: int) -> dict[str, EvidencePassage]:
        fused_scores = rrf_fuse_scores(dense, bm25, k)
        dense_ranks = {cid: rank for rank, (cid, _score) in enumerate(dense, start=1)}
        bm25_ranks = {cid: rank for rank, (cid, _score) in enumerate(bm25, start=1)}

        fused: dict[str, EvidencePassage] = {}
        for cid, score in fused_scores.items():
            chunk = self.chunks_by_id.get(cid)
            if chunk is None:
                continue
            fused[cid] = EvidencePassage(
                chunk_id=cid,
                text=chunk.text,
                content_type=chunk.content_type,
                heading_id=chunk.heading_id,
                heading_path=chunk.heading_path,
                page_index=chunk.page_index,
                table_id=chunk.table_id,
                dense_rank=dense_ranks.get(cid),
                bm25_rank=bm25_ranks.get(cid),
                rrf_score=score,
            )
        return fused

    def _apply_section_boost(self, query: str, fused: dict[str, EvidencePassage], bonus: float = 0.02) -> str | None:
        """Boost chunks from the section SectionMatcher confidently matched (BM25
        title + dense section search agree). Returns the matched heading_id (or
        None) for diagnostics. See section_match.py for why this needs both
        signals to agree -- a dense-only version of this regressed a different
        query class and was reverted before this version replaced it.
        """
        matched = self.section_matcher.matched_heading_id(query, self.embedder, self.client)
        if matched:
            for passage in fused.values():
                if passage.heading_id == matched:
                    passage.rrf_score += bonus
        return matched

    def _rerank(self, query: str, passages: list[EvidencePassage]) -> list[EvidencePassage]:
        if self._cross_encoder is None:
            from sentence_transformers import CrossEncoder

            self._cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        pairs = [(query, p.text) for p in passages]
        scores = self._cross_encoder.predict(pairs)
        for p, s in zip(passages, scores):
            p.rerank_score = float(s)
        return sorted(passages, key=lambda p: -p.rerank_score)

    def _expand_tables(self, passages: list[EvidencePassage]) -> list[EvidencePassage]:
        """For table-chunk evidence, append the full original table so numeric
        answers aren't based on a partial row-group slice.
        """
        seen_tables: set[str] = set()
        for p in passages:
            if p.table_id and p.table_id not in seen_tables:
                seen_tables.add(p.table_id)
                table = self.tables_by_id.get(p.table_id)
                if table:
                    full_rows = "\n".join(" | ".join((c or "") for c in row) for row in table.raw_cells)
                    p.text = f"{p.text}\n\n[Full original table {table.table_id}]\n{full_rows}"
        return passages

    def retrieve(self, query: str, use_rerank: bool = True) -> RetrievalResult:
        s = self.settings
        dense = self._dense_search(query, s.dense_top_k)
        bm25 = self._bm25_search(query, s.bm25_top_k)
        fused = self._rrf_fuse(dense, bm25, s.rrf_k)
        matched_section = self._apply_section_boost(query, fused)
        ranked = sorted(fused.values(), key=lambda p: -p.rrf_score)

        candidates = ranked[: s.rerank_top_k]
        if use_rerank and candidates:
            candidates = self._rerank(query, candidates)

        evidence = candidates[: s.evidence_top_k]
        evidence = self._expand_tables(evidence)

        diagnostics = {
            "dense_candidates": len(dense),
            "bm25_candidates": len(bm25),
            "fused_candidates": len(fused),
            "reranked": use_rerank,
            "matched_section": matched_section,
        }
        return RetrievalResult(query=query, evidence=evidence, diagnostics=diagnostics)

    def retrieve_variant(self, query: str, mode: str, top_k: int = 10) -> list[EvidencePassage]:
        """Retrieval under a single named strategy, for comparative evaluation
        (Phase 17: dense-only vs BM25-only vs hybrid vs hybrid+rerank).
        """
        s = self.settings
        if mode == "dense":
            dense = self._dense_search(query, max(top_k, s.dense_top_k))
            fused = self._rrf_fuse(dense, [], s.rrf_k)
            ranked = sorted(fused.values(), key=lambda p: -p.rrf_score)
            return ranked[:top_k]
        if mode == "bm25":
            bm25 = self._bm25_search(query, max(top_k, s.bm25_top_k))
            fused = self._rrf_fuse([], bm25, s.rrf_k)
            ranked = sorted(fused.values(), key=lambda p: -p.rrf_score)
            return ranked[:top_k]
        if mode == "hybrid":
            dense = self._dense_search(query, s.dense_top_k)
            bm25 = self._bm25_search(query, s.bm25_top_k)
            fused = self._rrf_fuse(dense, bm25, s.rrf_k)
            self._apply_section_boost(query, fused)
            ranked = sorted(fused.values(), key=lambda p: -p.rrf_score)
            return ranked[:top_k]
        if mode == "hybrid_rerank":
            dense = self._dense_search(query, s.dense_top_k)
            bm25 = self._bm25_search(query, s.bm25_top_k)
            fused = self._rrf_fuse(dense, bm25, s.rrf_k)
            self._apply_section_boost(query, fused)
            ranked = sorted(fused.values(), key=lambda p: -p.rrf_score)[: s.rerank_top_k]
            ranked = self._rerank(query, ranked) if ranked else ranked
            return ranked[:top_k]
        raise ValueError(f"Unknown retrieval mode: {mode}")
