"""Phase 17 end-to-end answer evaluation: citation accuracy, numerical accuracy,
answer faithfulness signals, clarification/general-knowledge routing correctness,
and latency. Requires ANTHROPIC_API_KEY to be configured (makes real LLM calls).

Usage:
    PYTHONPATH=src python -m bmo_agent.eval.run_answer_eval
"""

from __future__ import annotations

import json
import logging
import re

from bmo_agent.agent.graph import BMOAgent
from bmo_agent.agent.llm import LLMNotConfiguredError
from bmo_agent.config import get_settings
from bmo_agent.eval.questions import QUESTIONS

logging.basicConfig(level=logging.WARNING)


def _number_present(value: float, text: str) -> bool:
    """Heuristic numeric-accuracy check: does a value (allowing for $, commas,
    rounding to the nearest integer or one decimal) appear in the answer text?
    This is a lightweight substring-style check, not full semantic validation --
    documented as such in the eval summary, per the brief's 'do not fabricate
    evaluation results' instruction.
    """
    candidates = {
        f"{value:,.0f}",
        f"{value:,.1f}",
        f"{value:.0f}",
        f"{value:.1f}",
        f"{value:.2f}",
        str(int(value)) if float(value).is_integer() else str(value),
    }
    text_numbers = set(re.findall(r"-?\d[\d,]*\.?\d*", text))
    return any(c in text_numbers or c in text for c in candidates)


def main() -> None:
    settings = get_settings()
    try:
        agent = BMOAgent(settings)
    except LLMNotConfiguredError as exc:
        print(f"Cannot run answer evaluation: {exc}")
        return

    rows = []
    for q in QUESTIONS:
        try:
            result = agent.ask(q.question)
        except Exception as exc:  # noqa: BLE001
            rows.append({"id": q.id, "category": q.category, "error": str(exc)})
            continue

        citation_pages = {c["page"] for c in result.get("citations", [])}
        expected_pages = set(q.expected_pages or [])

        citation_hit = None
        if q.expect_citations and expected_pages:
            citation_hit = bool(citation_pages & expected_pages)

        numeric_hits = None
        if q.expected_values:
            checks = {}
            for key, val in q.expected_values.items():
                if isinstance(val, (int, float)):
                    checks[key] = _number_present(val, result["answer"])
            numeric_hits = checks

        clarification_correct = None
        if q.expect_clarification:
            clarification_correct = result.get("response_type") == "clarification"

        no_fabricated_citation = None
        if not q.expect_citations:
            no_fabricated_citation = len(result.get("citations", [])) == 0

        rows.append(
            {
                "id": q.id,
                "category": q.category,
                "question": q.question,
                "response_type": result.get("response_type"),
                "citation_hit": citation_hit,
                "numeric_hits": numeric_hits,
                "clarification_correct": clarification_correct,
                "no_fabricated_citation": no_fabricated_citation,
                "latency_ms": result.get("latency_ms"),
                "answer_preview": result["answer"][:200],
            }
        )

    def _rate(key: str) -> float | None:
        vals = [r[key] for r in rows if r.get(key) is not None]
        return round(sum(1 for v in vals if v) / len(vals), 3) if vals else None

    summary = {
        "n_questions": len(rows),
        "citation_accuracy": _rate("citation_hit"),
        "clarification_accuracy": _rate("clarification_correct"),
        "no_fabricated_citation_rate": _rate("no_fabricated_citation"),
        "avg_latency_ms": round(sum(r["latency_ms"] for r in rows if r.get("latency_ms")) / max(1, len(rows)), 1),
    }

    print(json.dumps(summary, indent=2))

    out_dir = settings.processed_dir / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "answer_eval_summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "answer_eval_detail.json").write_text(json.dumps(rows, indent=2))
    print(f"\nWrote {out_dir / 'answer_eval_summary.json'} and answer_eval_detail.json")


if __name__ == "__main__":
    main()
