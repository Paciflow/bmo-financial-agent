from __future__ import annotations

import json
import logging
from pathlib import Path

import pymupdf

from bmo_agent.config import Settings, get_settings
from bmo_agent.models import HeadingNode, TableBlock

logger = logging.getLogger(__name__)


def _heading_for_page(headings: list[HeadingNode], page_index: int) -> HeadingNode | None:
    """Return the most specific (deepest) heading whose range contains page_index."""
    candidates = [h for h in headings if h.start_page <= page_index <= h.end_page]
    if not candidates:
        return None
    return max(candidates, key=lambda h: h.level)


def extract_tables(pdf_path: Path, headings: list[HeadingNode], settings: Settings | None = None) -> list[TableBlock]:
    """Detect table structure per page with PyMuPDF's built-in table finder and link
    each table to the deepest enclosing heading, per PDF page.
    """
    settings = settings or get_settings()
    doc = pymupdf.open(str(pdf_path))

    tables: list[TableBlock] = []
    table_counter = 0
    for page_index, page in enumerate(doc):
        found = page.find_tables()
        for t in found.tables:
            try:
                cells = t.extract()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to extract table on page %d: %s", page_index, exc)
                continue
            if not cells or len(cells) < 2:
                continue
            heading = _heading_for_page(headings, page_index)
            table_counter += 1
            table_id = f"t{table_counter:04d}"
            tables.append(
                TableBlock(
                    table_id=table_id,
                    page_index=page_index,
                    bbox=tuple(t.bbox),
                    n_rows=len(cells),
                    n_cols=len(cells[0]) if cells else 0,
                    raw_cells=cells,
                    heading_id=heading.id if heading else None,
                    heading_path=heading.path if heading else None,
                )
            )

    out_path = settings.processed_dir / "tables.json"
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps([t.model_dump() for t in tables], indent=2), encoding="utf-8")
    logger.info("Detected %d tables across %d pages -> %s", len(tables), doc.page_count, out_path)
    return tables


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    settings = get_settings()
    pdf_path = Path(sys.argv[1]) if len(sys.argv) > 1 else settings.raw_dir / f"{settings.document_id}.pdf"
    headings_path = settings.processed_dir / "headings.json"
    headings = [HeadingNode(**h) for h in json.loads(headings_path.read_text())]
    extract_tables(pdf_path, headings, settings)
