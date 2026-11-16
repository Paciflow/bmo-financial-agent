from __future__ import annotations

import json
import logging
from pathlib import Path

import pymupdf

from bmo_agent.config import Settings, get_settings
from bmo_agent.models import PageText

logger = logging.getLogger(__name__)

PAGE_DELIMITER_TEMPLATE = "\n\n===== PAGE {page_num} (pdf_index={page_index}) =====\n\n"


def extract_native_text(pdf_path: Path, settings: Settings | None = None) -> list[PageText]:
    """Extract all pages with PyMuPDF, preserving PDF page index and printed label.

    Writes raw_text.txt (with explicit page delimiters) and raw_pages.json to
    settings.processed_dir.
    """
    settings = settings or get_settings()
    settings.processed_dir.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(str(pdf_path))

    pages: list[PageText] = []
    text_parts: list[str] = []
    for i, page in enumerate(doc):
        text = page.get_text()
        try:
            label = page.get_label()
            printed_label = label if label else None
        except Exception:  # noqa: BLE001
            printed_label = None
        pages.append(PageText(page_index=i, printed_label=printed_label, text=text, char_count=len(text)))
        text_parts.append(PAGE_DELIMITER_TEMPLATE.format(page_num=i + 1, page_index=i))
        text_parts.append(text)

    raw_text_path = settings.processed_dir / "raw_text.txt"
    raw_pages_path = settings.processed_dir / "raw_pages.json"
    raw_text_path.write_text("".join(text_parts), encoding="utf-8")
    raw_pages_path.write_text(json.dumps([p.model_dump() for p in pages], indent=2), encoding="utf-8")

    logger.info("Extracted %d pages (%d total chars) -> %s", len(pages), sum(p.char_count for p in pages), raw_text_path)
    return pages


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    settings = get_settings()
    pdf_path = Path(sys.argv[1]) if len(sys.argv) > 1 else settings.raw_dir / f"{settings.document_id}.pdf"
    extract_native_text(pdf_path, settings)
