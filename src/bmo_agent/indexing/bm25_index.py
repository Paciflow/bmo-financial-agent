from __future__ import annotations

import json
import logging
import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from bmo_agent.config import Settings, get_settings
from bmo_agent.models import Chunk

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z]+\d*(?:'[A-Za-z]+)?|\d[\d,.]*%?")


def tokenize(text: str) -> list[str]:
    """Tokenizer that keeps financial terminology, acronyms, numbers (with commas/
    decimals/percent signs) and table identifiers intact, lowercased for matching.
    """
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def build_bm25_index(chunks: list[Chunk], settings: Settings | None = None) -> BM25Okapi:
    settings = settings or get_settings()
    out_dir = Path(settings.bm25_index_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus_tokens = [tokenize(c.text) for c in chunks]
    bm25 = BM25Okapi(corpus_tokens)

    chunk_ids = [c.chunk_id for c in chunks]
    with (out_dir / "bm25.pkl").open("wb") as f:
        pickle.dump(bm25, f)
    (out_dir / "chunk_ids.json").write_text(json.dumps(chunk_ids), encoding="utf-8")

    logger.info("Built BM25 index over %d chunks -> %s", len(chunks), out_dir)
    return bm25


def load_bm25_index(settings: Settings | None = None) -> tuple[BM25Okapi, list[str]]:
    settings = settings or get_settings()
    out_dir = Path(settings.bm25_index_path)
    with (out_dir / "bm25.pkl").open("rb") as f:
        bm25 = pickle.load(f)
    chunk_ids = json.loads((out_dir / "chunk_ids.json").read_text())
    return bm25, chunk_ids


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    settings = get_settings()
    chunks = [Chunk(**c) for c in json.loads((settings.processed_dir / "chunks.json").read_text())]
    build_bm25_index(chunks, settings)
