"""Vision-based interpretation for the document's non-textual diagram content.

Scope note: reliably auto-detecting "this page has a diagram" vs. "this page has a
heavily-gridded table" turned out to need more than a simple heuristic -- vector
drawing count alone doesn't separate them (BMO's statistical tables use just as many
ruled lines as an org chart uses box outlines; see README). Rather than build a
classifier the remaining time doesn't justify, this script interprets a short,
explicitly-reviewed list of known diagram pages. DIAGRAM_PAGES was populated by
manually inspecting a page where the agent's own answer flagged garbled, column-like
evidence text (page 20's operating-segments organizational chart) -- a real signal
worth following up, not a blind guess. Extending DIAGRAM_PAGES after inspecting more
pages is the natural next step for broader coverage.

Usage:
    PYTHONPATH=src python scripts/interpret_diagrams.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from bmo_agent.config import get_settings
from bmo_agent.interpretation.diagrams import interpret_diagram_page
from bmo_agent.models import Chunk, HeadingNode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("interpret_diagrams")

DIAGRAM_PAGES = [19]  # 0-based page indices; page 20 (1-based) = BMO operating-segments org chart


def _heading_for_page(headings: list[HeadingNode], page_index: int) -> HeadingNode | None:
    candidates = [h for h in headings if h.start_page <= page_index <= h.end_page]
    return max(candidates, key=lambda h: h.level) if candidates else None


def main() -> None:
    settings = get_settings()
    pdf_path = settings.raw_dir / f"{settings.document_id}.pdf"
    headings = [HeadingNode(**h) for h in json.loads((settings.processed_dir / "headings.json").read_text())]

    diagram_chunks: list[Chunk] = []
    for page_index in DIAGRAM_PAGES:
        description = interpret_diagram_page(pdf_path, page_index, settings)
        heading = _heading_for_page(headings, page_index)
        diagram_chunks.append(
            Chunk(
                chunk_id=f"diag_p{page_index + 1:03d}_000",
                document_id=settings.document_id,
                content_type="diagram",
                heading_id=heading.id if heading else None,
                heading_path=heading.path if heading else None,
                page_index=page_index,
                text=f"[Diagram, page {page_index + 1}] {description}",
            )
        )

    chunks_path = settings.processed_dir / "chunks.json"
    existing = [Chunk(**c) for c in json.loads(chunks_path.read_text())]
    new_ids = {dc.chunk_id for dc in diagram_chunks}
    existing = [c for c in existing if c.chunk_id not in new_ids]
    merged = existing + diagram_chunks
    chunks_path.write_text(json.dumps([c.model_dump() for c in merged], indent=2))
    logger.info("Added %d diagram chunk(s); total chunks now %d -> %s", len(diagram_chunks), len(merged), chunks_path)


if __name__ == "__main__":
    main()
