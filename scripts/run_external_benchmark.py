"""Run a stratified sample of the user-supplied 300-question benchmark
(bmo_300_question_benchmark.json) through the live agent and grade it.

Design notes (read before trusting the numbers):
- The benchmark's `printed_page` values follow the FULL annual report's internal
  pagination, not our MD&A-only PDF's own page numbers. Verified empirically: our
  PDF's page footers read "<N> BMO Financial Group 208th Annual Report 2025" where
  N = our_pdf_1based_page + 14, consistently across 4 independently checked pages.
  So printed_page -> our_pdf_page via `page - 14`.
- The 300 questions contain heavy duplication -- many of the same ~10-15 underlying
  facts are asked 3x (Easy/Medium/Hard) with only the phrasing changed. Running all
  300 would mostly re-test identical retrieval for near-identical questions. This
  script dedupes by (reference_answer, source) before sampling, so the sample
  actually covers distinct facts/behaviors rather than restating the same ones.
- Grading is a heuristic substring/number match against reference_answer, same
  caveat as eval/run_answer_eval.py: it's a lightweight check, not full semantic
  grading. Questions with reference_answer=None are logged for manual review, not
  auto-graded pass/fail -- the benchmark's own notes field says as much.

Usage:
    PYTHONPATH=src python scripts/run_external_benchmark.py <path_to_benchmark.json> [--per-bucket N]
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import time
from collections import defaultdict

from bmo_agent.agent.graph import BMOAgent
from bmo_agent.config import get_settings

logging.basicConfig(level=logging.WARNING)

PAGE_OFFSET = 14  # our_pdf_page = printed_page - PAGE_OFFSET
RANDOM_SEED = 42


def _expected_pdf_pages(source: dict | None) -> list[int] | None:
    if not source:
        return None
    pages = []
    for key in ("printed_page", "printed_page_start"):
        val = source.get(key)
        if val is not None:
            try:
                pages.append(int(val) - PAGE_OFFSET)
            except (TypeError, ValueError):
                pass
    return pages or None


def _grade_reference(reference: str, answer_text: str) -> bool:
    ref = reference.lower().strip()
    ans = answer_text.lower()
    ref_numbers = re.findall(r"\d[\d.]*", ref)
    if ref_numbers:
        ans_numbers = set(re.findall(r"\d[\d.]*", ans))
        if all(n in ans_numbers for n in ref_numbers):
            return True
    core = ref.replace("approximately", "").replace("$", "").strip()
    return core in ans or ref in ans


def stratified_sample(questions: list[dict], per_bucket: int, seed: int = RANDOM_SEED) -> list[dict]:
    rng = random.Random(seed)
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    seen_facts: set[tuple] = set()

    for q in questions:
        key = (q["difficulty"], q["type"])
        fact_key = (q.get("reference_answer"), json.dumps(q.get("source"), sort_keys=True))
        if q.get("reference_answer") is not None and fact_key in seen_facts:
            continue  # skip near-duplicate phrasing of an already-covered fact
        seen_facts.add(fact_key)
        buckets[key].append(q)

    sample: list[dict] = []
    for key, items in buckets.items():
        rng.shuffle(items)
        sample.extend(items[:per_bucket])
    return sample


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark_path")
    parser.add_argument("--per-bucket", type=int, default=12, help="max questions sampled per (difficulty,type) bucket")
    args = parser.parse_args()

    data = json.loads(open(args.benchmark_path).read())
    all_questions = data["questions"]
    sample = stratified_sample(all_questions, args.per_bucket)
    print(f"Sampled {len(sample)} of {len(all_questions)} questions (deduped + stratified by difficulty x type)")

    settings = get_settings()
    agent = BMOAgent(settings)

    results = []
    start_all = time.perf_counter()
    for i, q in enumerate(sample, 1):
        t0 = time.perf_counter()
        try:
            r = agent.ask(q["question"])
        except Exception as exc:  # noqa: BLE001
            results.append({**q, "error": str(exc)})
            continue
        latency = (time.perf_counter() - t0) * 1000

        expected_pages = _expected_pdf_pages(q.get("source"))
        citation_pages = {c["page"] for c in r.get("citations", [])}
        page_hit = bool(expected_pages and citation_pages & set(expected_pages)) if expected_pages else None

        value_correct = None
        if q.get("reference_answer"):
            value_correct = _grade_reference(q["reference_answer"], r["answer"])

        fabrication_check = None
        if q["type"] == "general":
            fabrication_check = len(r.get("citations", [])) == 0

        results.append(
            {
                "id": q["id"],
                "difficulty": q["difficulty"],
                "type": q["type"],
                "question": q["question"],
                "reference_answer": q.get("reference_answer"),
                "expected_pages": expected_pages,
                "response_type": r.get("response_type"),
                "citation_pages": sorted(citation_pages),
                "page_hit": page_hit,
                "value_correct": value_correct,
                "no_fabricated_citation": fabrication_check,
                "answer": r["answer"],
                "latency_ms": round(latency, 1),
            }
        )
        print(f"[{i}/{len(sample)}] {q['id']} ({q['difficulty']}/{q['type']}) "
              f"value_correct={value_correct} page_hit={page_hit} ({latency/1000:.1f}s)")

    total_time = time.perf_counter() - start_all
    print(f"\nTotal: {len(results)} questions in {total_time/60:.1f} min")

    out_path = settings.processed_dir / "eval" / "external_benchmark_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
