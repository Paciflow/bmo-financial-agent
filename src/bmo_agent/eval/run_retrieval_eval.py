"""Phase 17 retrieval comparison: dense-only vs BM25-only vs hybrid vs hybrid+rerank.

Measures Recall@5 and MRR against manually-verified expected pages, plus average
per-query latency for each strategy. Does not require an Anthropic API key --
retrieval-only, no LLM calls.

Usage:
    PYTHONPATH=src python -m bmo_agent.eval.run_retrieval_eval
"""

from __future__ import annotations

import json
import logging
import time

from bmo_agent.config import get_settings
from bmo_agent.eval.metrics import reciprocal_rank, recall_at_k
from bmo_agent.eval.questions import QUESTIONS
from bmo_agent.retrieval.hybrid import HybridRetriever

logging.basicConfig(level=logging.WARNING)

MODES = ["dense", "bm25", "hybrid", "hybrid_rerank"]
RECALL_K = 5


def ranked_pages_from_evidence(evidence) -> list[int]:
    pages: list[int] = []
    for e in evidence:
        p = e.page_index + 1
        if p not in pages:
            pages.append(p)
    return pages


def main() -> None:
    settings = get_settings()
    retriever = HybridRetriever(settings)

    retrieval_questions = [q for q in QUESTIONS if q.expected_pages]
    print(f"Evaluating {len(retrieval_questions)} retrieval questions across modes: {MODES}\n")

    summary = {}
    per_question_rows = []

    for mode in MODES:
        recalls, mrrs, latencies = [], [], []
        for q in retrieval_questions:
            start = time.perf_counter()
            evidence = retriever.retrieve_variant(q.question, mode=mode, top_k=10)
            latency_ms = (time.perf_counter() - start) * 1000
            ranked_pages = ranked_pages_from_evidence(evidence)

            r = recall_at_k(ranked_pages, q.expected_pages, RECALL_K)
            mrr = reciprocal_rank(ranked_pages, q.expected_pages)
            recalls.append(r)
            mrrs.append(mrr)
            latencies.append(latency_ms)

            per_question_rows.append(
                {
                    "id": q.id,
                    "mode": mode,
                    "category": q.category,
                    f"recall@{RECALL_K}": r,
                    "mrr": round(mrr, 3),
                    "latency_ms": round(latency_ms, 1),
                    "top_pages": ranked_pages[:RECALL_K],
                    "expected_pages": q.expected_pages,
                }
            )

        summary[mode] = {
            f"recall@{RECALL_K}": round(sum(recalls) / len(recalls), 3),
            "mrr": round(sum(mrrs) / len(mrrs), 3),
            "avg_latency_ms": round(sum(latencies) / len(latencies), 1),
            "n_questions": len(retrieval_questions),
        }

    print("=" * 70)
    print(f"{'Mode':<16}{'Recall@' + str(RECALL_K):<12}{'MRR':<10}{'Avg latency (ms)':<18}")
    print("-" * 70)
    for mode in MODES:
        s = summary[mode]
        print(f"{mode:<16}{s[f'recall@{RECALL_K}']:<12}{s['mrr']:<10}{s['avg_latency_ms']:<18}")
    print("=" * 70)

    out_dir = settings.processed_dir / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "retrieval_eval_summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "retrieval_eval_detail.json").write_text(json.dumps(per_question_rows, indent=2))
    print(f"\nWrote {out_dir / 'retrieval_eval_summary.json'} and retrieval_eval_detail.json")


if __name__ == "__main__":
    main()
