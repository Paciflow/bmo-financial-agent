from __future__ import annotations


def rrf_fuse_scores(dense: list[tuple[str, float]], bm25: list[tuple[str, float]], k: int) -> dict[str, float]:
    """Pure Reciprocal Rank Fusion: combine two ranked (id, score) lists into one
    fused score per id, score(id) = sum(1 / (k + rank)) over lists where id appears.
    Kept dependency-free (no Qdrant/embedder) so it's cheap to unit test.
    """
    scores: dict[str, float] = {}
    for rank, (item_id, _score) in enumerate(dense, start=1):
        scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    for rank, (item_id, _score) in enumerate(bm25, start=1):
        scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    return scores
