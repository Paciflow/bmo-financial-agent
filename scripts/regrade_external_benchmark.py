"""Re-grade run_external_benchmark.py's saved results with a better heuristic for
definitional/conceptual questions, without re-spending API calls.

The original grader required reference_answer to appear as a near-exact substring,
which works for figures ("$11.44") but produces false negatives on short canonical
definitions ("reduction of an asset carrying value") compared against a longer
paraphrased explanation. This adds a word-overlap check as a second path: if most
of the reference's meaningful (non-stopword) words appear anywhere in the answer,
count it correct. Still a heuristic, not semantic grading -- flagged results that
flip from fail to pass are worth spot-checking, not blindly trusted either.

Usage:
    python scripts/regrade_external_benchmark.py <results.json>
"""

from __future__ import annotations

import json
import re
import sys

STOPWORDS = {
    "a", "an", "the", "of", "to", "in", "on", "for", "and", "or", "is", "are", "was",
    "were", "be", "that", "this", "with", "as", "by", "from", "its", "it", "or",
    "other", "than", "not",
}


def content_words(text: str) -> set[str]:
    words = re.findall(r"[a-z]+", text.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 2}


def word_overlap_pass(reference: str, answer: str, threshold: float = 0.6) -> bool:
    ref_words = content_words(reference)
    if not ref_words:
        return False
    ans_words = content_words(answer)
    hit = len(ref_words & ans_words) / len(ref_words)
    return hit >= threshold


def main() -> None:
    path = sys.argv[1]
    results = json.loads(open(path).read())

    flips = []
    for r in results:
        if r.get("value_correct") is False and r.get("reference_answer"):
            new_verdict = word_overlap_pass(r["reference_answer"], r["answer"])
            if new_verdict:
                flips.append(r["id"])
                r["value_correct"] = True
                r["value_correct_method"] = "word_overlap"
        elif r.get("value_correct") is True:
            r["value_correct_method"] = "exact_substring_or_number"

    print(f"Re-graded: {len(flips)} questions flipped from fail to pass via word-overlap: {flips}")

    by_bucket = {}
    for r in results:
        key = (r["difficulty"], r["type"])
        by_bucket.setdefault(key, {"total": 0, "graded": 0, "correct": 0, "page_checked": 0, "page_hit": 0})
        b = by_bucket[key]
        b["total"] += 1
        if r.get("value_correct") is not None:
            b["graded"] += 1
            if r["value_correct"]:
                b["correct"] += 1
        if r.get("page_hit") is not None:
            b["page_checked"] += 1
            if r["page_hit"]:
                b["page_hit"] += 1

    print(f"\n{'Difficulty/Type':<22}{'N':<5}{'Graded':<8}{'Correct':<9}{'Accuracy':<10}{'PageHit':<10}")
    print("-" * 70)
    totals = {"total": 0, "graded": 0, "correct": 0, "page_checked": 0, "page_hit": 0}
    for (diff, typ), b in sorted(by_bucket.items()):
        acc = f"{b['correct']}/{b['graded']}" if b["graded"] else "n/a"
        acc_pct = f"({100*b['correct']/b['graded']:.0f}%)" if b["graded"] else ""
        page = f"{b['page_hit']}/{b['page_checked']}" if b["page_checked"] else "n/a"
        print(f"{diff+'/'+typ:<22}{b['total']:<5}{b['graded']:<8}{acc+' '+acc_pct:<9}{'':<1}{page:<10}")
        for k in totals:
            totals[k] += b[k]

    overall_acc = f"{totals['correct']}/{totals['graded']} ({100*totals['correct']/totals['graded']:.0f}%)" if totals["graded"] else "n/a"
    overall_page = f"{totals['page_hit']}/{totals['page_checked']} ({100*totals['page_hit']/totals['page_checked']:.0f}%)" if totals["page_checked"] else "n/a"
    print("-" * 70)
    print(f"Overall value accuracy: {overall_acc}")
    print(f"Overall page-citation hit rate: {overall_page}")

    general_fab = [r for r in results if r["type"] == "general" and r.get("no_fabricated_citation") is not None]
    if general_fab:
        ok = sum(1 for r in general_fab if r["no_fabricated_citation"])
        print(f"No-fabricated-citation rate (general questions): {ok}/{len(general_fab)}")

    out_path = path.replace(".json", "_regraded.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
