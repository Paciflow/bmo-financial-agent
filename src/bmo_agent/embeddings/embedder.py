from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np

from bmo_agent.config import Settings, get_settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    logger.info("Loading embedding model %s (local, no network dependency after first download)", model_name)
    return SentenceTransformer(model_name)


class Embedder:
    """Wraps a local sentence-transformers model so the agent's dense retrieval path
    has zero network dependency at query time -- important for a live demo where an
    external embeddings API could add latency or fail.

    Swappable: production deployments can point this at Bedrock Titan Text Embeddings V2
    instead (same interface), per the target architecture in the slides.
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.model = _load_model(self.settings.embedding_model)

    def embed(self, texts: list[str], batch_size: int = 32, is_query: bool = False) -> list[list[float]]:
        if not texts:
            return []
        if is_query and self.settings.embedding_query_prefix:
            # Asymmetric retrieval models (e.g. BGE) are trained to prepend an
            # instruction to the query side only -- never to indexed passages, or
            # the asymmetry that makes the model work is lost. embedding_query_prefix
            # is "" for models (like the previous all-MiniLM-L6-v2) that don't use one.
            texts = [self.settings.embedding_query_prefix + t for t in texts]
        vectors = self.model.encode(texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)
        return np.asarray(vectors).tolist()

    def embed_one(self, text: str, is_query: bool = False) -> list[float]:
        return self.embed([text], is_query=is_query)[0]
