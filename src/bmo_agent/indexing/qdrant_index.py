from __future__ import annotations

import json
import logging
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from bmo_agent.config import Settings, get_settings
from bmo_agent.embeddings.embedder import Embedder
from bmo_agent.models import Chunk, HeadingNode

logger = logging.getLogger(__name__)


def get_client(settings: Settings | None = None) -> QdrantClient:
    settings = settings or get_settings()
    Path(settings.qdrant_local_path).mkdir(parents=True, exist_ok=True)
    # Embedded local mode: persists to disk, no server/Docker required.
    return QdrantClient(path=settings.qdrant_local_path)


def _ensure_collection(client: QdrantClient, name: str, dim: int) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if name in existing:
        return
    client.create_collection(collection_name=name, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))
    logger.info("Created Qdrant collection '%s' (dim=%d)", name, dim)


def _stable_id(text_id: str) -> int:
    import hashlib

    return int(hashlib.sha1(text_id.encode("utf-8")).hexdigest()[:16], 16)


def index_sections(client: QdrantClient, headings: list[HeadingNode], section_texts: dict[str, str], embedder: Embedder, settings: Settings) -> int:
    _ensure_collection(client, settings.qdrant_sections_collection, settings.embedding_dim)
    texts = [section_texts.get(h.id, h.title) for h in headings]
    vectors = embedder.embed(texts)
    points = []
    for h, vec in zip(headings, vectors):
        points.append(
            PointStruct(
                id=_stable_id(f"section::{h.id}"),
                vector=vec,
                payload={
                    "heading_id": h.id,
                    "title": h.title,
                    "level": h.level,
                    "path": h.path,
                    "start_page": h.start_page,
                    "end_page": h.end_page,
                    "parent_id": h.parent_id,
                },
            )
        )
    client.upsert(collection_name=settings.qdrant_sections_collection, points=points)
    logger.info("Indexed %d sections into '%s'", len(points), settings.qdrant_sections_collection)
    return len(points)


def index_chunks(client: QdrantClient, chunks: list[Chunk], embedder: Embedder, settings: Settings, batch_size: int = 32) -> int:
    _ensure_collection(client, settings.qdrant_chunks_collection, settings.embedding_dim)
    total = 0
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        vectors = embedder.embed([c.text for c in batch], batch_size=batch_size)
        points = [
            PointStruct(
                id=_stable_id(c.chunk_id),
                vector=vec,
                payload={
                    "chunk_id": c.chunk_id,
                    "document_id": c.document_id,
                    "content_type": c.content_type,
                    "heading_id": c.heading_id,
                    "heading_path": c.heading_path,
                    "page_index": c.page_index,
                    "table_id": c.table_id,
                    "text": c.text,
                },
            )
            for c, vec in zip(batch, vectors)
        ]
        client.upsert(collection_name=settings.qdrant_chunks_collection, points=points)
        total += len(points)
    logger.info("Indexed %d chunks into '%s'", total, settings.qdrant_chunks_collection)
    return total


def build_section_texts(headings: list[HeadingNode], pages_text_by_index: dict[int, str], max_chars: int = 1500) -> dict[str, str]:
    """Lightweight section representation for the sections collection: title + lead
    text from the section's page range. (Swap in Claude-generated summaries later --
    see interpretation/semantic.py -- once an Anthropic key is configured.)
    """
    out: dict[str, str] = {}
    for h in headings:
        text_parts = [h.title]
        for p in range(h.start_page, min(h.end_page, h.start_page + 2) + 1):
            text_parts.append(pages_text_by_index.get(p, ""))
        combined = " ".join(text_parts)
        out[h.id] = combined[:max_chars]
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    settings = get_settings()
    headings = [HeadingNode(**h) for h in json.loads((settings.processed_dir / "headings.json").read_text())]
    chunks = [Chunk(**c) for c in json.loads((settings.processed_dir / "chunks.json").read_text())]
    pages = json.loads((settings.processed_dir / "raw_pages.json").read_text())
    pages_by_index = {p["page_index"]: p["text"] for p in pages}

    embedder = Embedder(settings)
    client = get_client(settings)

    section_texts = build_section_texts(headings, pages_by_index)
    index_sections(client, headings, section_texts, embedder, settings)
    index_chunks(client, chunks, embedder, settings)
