from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from bmo_agent.chunking.text_cleanup import clean_table_cell
from bmo_agent.config import Settings, get_settings
from bmo_agent.models import Chunk, HeadingNode, PageText, TableBlock

logger = logging.getLogger(__name__)

NARRATIVE_CHUNK_SIZE = 1200
NARRATIVE_OVERLAP = 200
MAX_HEADER_ROWS = 6
TABLE_DATA_ROWS_PER_CHUNK = 8

_HAS_LETTER_RE = re.compile(r"[A-Za-z]")


def _detect_header_row_count(rows: list[list[str | None]]) -> int:
    """Find where a table's real data starts, instead of assuming a fixed header
    row count -- tables on this document vary from 1 header row (simple two-column
    figures) to 4+ (multi-level column groupings like region x insured/uninsured x
    year). Row 0 is always treated as header (the top column-title row). From row 1
    on, the first row whose non-label cells are all non-empty and contain no letters
    (i.e. look like pure figures, not more column labels) marks the start of data.
    """
    if not rows:
        return 0
    for i in range(1, min(len(rows), MAX_HEADER_ROWS + 1)):
        value_cells = rows[i][1:]
        if not value_cells:
            continue
        looks_like_data = all(
            cell and cell.strip() and not _HAS_LETTER_RE.search(cell) for cell in value_cells
        )
        if looks_like_data:
            return i
    return min(1, len(rows) - 1)


def _leaf_headings(headings: list[HeadingNode]) -> list[HeadingNode]:
    parent_ids = {h.parent_id for h in headings if h.parent_id}
    return [h for h in headings if h.id not in parent_ids]


def _chunk_narrative_section(heading: HeadingNode, pages: list[PageText], document_id: str) -> list[Chunk]:
    section_pages = [p for p in pages if heading.start_page <= p.page_index <= heading.end_page]
    if not section_pages:
        return []

    # Concatenate section text, tracking (char_offset -> page_index) breakpoints.
    full_text = ""
    offsets: list[tuple[int, int]] = []  # (start_offset, page_index)
    for p in section_pages:
        offsets.append((len(full_text), p.page_index))
        full_text += p.text + "\n"

    full_text = full_text.strip()
    if len(full_text) < 40:
        return []

    chunks: list[Chunk] = []
    start = 0
    idx = 0
    n = len(full_text)
    while start < n:
        end = min(start + NARRATIVE_CHUNK_SIZE, n)
        piece = full_text[start:end].strip()
        if piece:
            # Attribute the chunk to whichever page its MIDPOINT falls in, not its
            # start offset. A chunk that starts at the tail of one page's footer and
            # is mostly the next page's content (common near page boundaries) was
            # previously mislabeled with the earlier page -- confirmed in testing:
            # a chunk starting one character into page 14's footer, whose actual
            # cited table was on page 15, was tagged "page 14".
            midpoint = (start + end) // 2
            page_index = section_pages[0].page_index
            for off, pidx in offsets:
                if off <= midpoint:
                    page_index = pidx
                else:
                    break
            chunks.append(
                Chunk(
                    chunk_id=f"c_{heading.id}_{idx:03d}",
                    document_id=document_id,
                    content_type="narrative",
                    heading_id=heading.id,
                    heading_path=heading.path,
                    page_index=page_index,
                    text=piece,
                )
            )
            idx += 1
        if end >= n:
            break
        start = end - NARRATIVE_OVERLAP
    return chunks


def _format_row(row: list[str | None]) -> str:
    return " | ".join(clean_table_cell(c) for c in row)


def _chunk_table(table: TableBlock, document_id: str) -> list[Chunk]:
    rows = table.raw_cells
    n_header = min(_detect_header_row_count(rows), max(1, len(rows) - 1))
    header_rows = rows[:n_header]
    data_rows = rows[n_header:]

    if not data_rows:
        text = "\n".join(_format_row(r) for r in rows)
        return [
            Chunk(
                chunk_id=f"tc_{table.table_id}_000",
                document_id=document_id,
                content_type="table",
                heading_id=table.heading_id,
                heading_path=table.heading_path,
                page_index=table.page_index,
                text=text,
                table_id=table.table_id,
            )
        ]

    chunks: list[Chunk] = []
    part = 0
    for i in range(0, len(data_rows), TABLE_DATA_ROWS_PER_CHUNK):
        group = data_rows[i : i + TABLE_DATA_ROWS_PER_CHUNK]
        header_text = "\n".join(_format_row(r) for r in header_rows)
        body_text = "\n".join(_format_row(r) for r in group)
        heading_prefix = f"[{table.heading_path}] " if table.heading_path else ""
        text = f"{heading_prefix}Table {table.table_id} (page {table.page_index + 1})\n{header_text}\n{body_text}"
        chunks.append(
            Chunk(
                chunk_id=f"tc_{table.table_id}_{part:03d}",
                document_id=document_id,
                content_type="table",
                heading_id=table.heading_id,
                heading_path=table.heading_path,
                page_index=table.page_index,
                text=text,
                table_id=table.table_id,
            )
        )
        part += 1
    return chunks


def build_chunks(
    pages: list[PageText],
    headings: list[HeadingNode],
    tables: list[TableBlock],
    document_id: str,
    settings: Settings | None = None,
) -> list[Chunk]:
    settings = settings or get_settings()
    chunks: list[Chunk] = []

    for heading in _leaf_headings(headings):
        chunks.extend(_chunk_narrative_section(heading, pages, document_id))

    for table in tables:
        chunks.extend(_chunk_table(table, document_id))

    out_path = settings.processed_dir / "chunks.json"
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps([c.model_dump() for c in chunks], indent=2), encoding="utf-8")
    logger.info(
        "Built %d chunks (%d narrative, %d table) -> %s",
        len(chunks),
        sum(1 for c in chunks if c.content_type == "narrative"),
        sum(1 for c in chunks if c.content_type == "table"),
        out_path,
    )
    return chunks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    settings = get_settings()
    pages = [PageText(**p) for p in json.loads((settings.processed_dir / "raw_pages.json").read_text())]
    headings = [HeadingNode(**h) for h in json.loads((settings.processed_dir / "headings.json").read_text())]
    tables = [TableBlock(**t) for t in json.loads((settings.processed_dir / "tables.json").read_text())]
    build_chunks(pages, headings, tables, settings.document_id, settings)
