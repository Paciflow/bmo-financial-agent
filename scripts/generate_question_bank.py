"""Generate a large (100+) grounded question bank from the already-extracted table
data, for broader (if shallower) accuracy sampling than the 20 hand-verified
eval/questions.py set.

Unlike questions.py (each one individually read and verified against the source
page by hand), these are generated deterministically FROM the structured
extraction itself: a (label, value, year, page) triple pulled straight out of
tables.json is correct by construction (it's the same data the chunker/index use),
not independently re-verified against the PDF page image. That's the honest
tradeoff for scale -- stated explicitly rather than implying hand-verification that
didn't happen.

Usage:
    PYTHONPATH=src python scripts/generate_question_bank.py
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict

from bmo_agent.chunking.chunker import _detect_header_row_count
from bmo_agent.chunking.text_cleanup import clean_table_cell
from bmo_agent.config import get_settings

_NUM_RE = re.compile(r"^\(?-?[\d,]+\.?\d*\)?$")


@dataclass
class GeneratedQuestion:
    id: str
    question: str
    expected_answer_text: str
    expected_value: float | None
    expected_page: int
    table_id: str
    label: str
    year: str | None


def _parse_number(raw: str) -> float | None:
    if not raw:
        return None
    s = raw.strip().rstrip(",.")
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()").replace(",", "")
    if not s or not re.match(r"^-?\d+\.?\d*$", s):
        return None
    val = float(s)
    return -val if neg else val


def _clean_label(raw: str) -> str | None:
    label = clean_table_cell(raw).strip()
    if len(label) < 4 or len(label) > 80:
        return None
    if not re.search(r"[A-Za-z]{3,}", label):
        return None
    return label


def generate_from_tables(max_questions: int = 200) -> list[GeneratedQuestion]:
    settings = get_settings()
    tables = json.loads((settings.processed_dir / "tables.json").read_text())

    questions: list[GeneratedQuestion] = []
    seen_questions: set[str] = set()  # global dedup on exact question text

    for table in tables:
        rows = table["raw_cells"]
        if len(rows) < 2:
            continue
        n_header = min(_detect_header_row_count(rows), max(1, len(rows) - 1))
        header_row = rows[0]
        data_rows = rows[n_header:]

        # Try to read two year labels from the header row (common BMO pattern:
        # cols are label, year_a, year_b, ...).
        year_cols: list[tuple[int, str]] = []
        for col_idx, cell in enumerate(header_row):
            if cell and re.match(r"^(19|20)\d{2}$", cell.strip()):
                year_cols.append((col_idx, cell.strip()))
        if not year_cols:
            continue

        for row in data_rows:
            if not row or not row[0]:
                continue
            label = _clean_label(row[0])
            if not label:
                continue

            for col_idx, year in year_cols:
                if col_idx >= len(row):
                    continue
                value = _parse_number(row[col_idx])
                if value is None:
                    continue
                question_text = f"What was BMO's {label.lower()} for fiscal {year}, per the MD&A?"
                if question_text in seen_questions:
                    continue
                seen_questions.add(question_text)
                qid = f"gen{len(questions) + 1:04d}"
                questions.append(
                    GeneratedQuestion(
                        id=qid,
                        question=question_text,
                        expected_answer_text=f"{label}: {value:,.2f} ({table['heading_path'] or 'N/A'}, page {table['page_index'] + 1})",
                        expected_value=value,
                        expected_page=table["page_index"] + 1,
                        table_id=table["table_id"],
                        label=label,
                        year=year,
                    )
                )

            if len(questions) >= max_questions:
                return questions
    return questions


def main() -> None:
    settings = get_settings()
    qs = generate_from_tables()
    out_path = settings.processed_dir / "eval" / "generated_question_bank.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps([asdict(q) for q in qs], indent=2))
    print(f"Generated {len(qs)} questions -> {out_path}")
    for q in qs[:5]:
        print(" -", q.question, "->", q.expected_answer_text)


if __name__ == "__main__":
    main()
