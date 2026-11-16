from __future__ import annotations


def recall_at_k(ranked_pages: list[int], expected_pages: list[int], k: int) -> float:
    if not expected_pages:
        return float("nan")
    top = set(ranked_pages[:k])
    hit = any(p in top for p in expected_pages)
    return 1.0 if hit else 0.0


def reciprocal_rank(ranked_pages: list[int], expected_pages: list[int]) -> float:
    if not expected_pages:
        return float("nan")
    for rank, page in enumerate(ranked_pages, start=1):
        if page in expected_pages:
            return 1.0 / rank
    return 0.0
