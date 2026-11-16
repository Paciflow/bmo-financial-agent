from __future__ import annotations

import base64
import logging
from pathlib import Path

from bmo_agent.agent.llm import get_anthropic_client
from bmo_agent.config import Settings, get_settings

logger = logging.getLogger(__name__)

DIAGRAM_PROMPT = """This image is one page of BMO Financial Group's 2025 MD&A. It contains a \
non-textual element (an organizational diagram, chart, or similar visual) whose structure is lost \
when the page is extracted as plain text (boxes/columns get linearized into a flat word sequence).

Describe this visual element as a clear, retrieval-friendly paragraph of plain text: what it shows, \
its entities, and the relationships/hierarchy between them. Be literal -- transcribe labels exactly as \
shown, don't invent entities or relationships that aren't visually present. If there are multiple \
distinct visual elements on the page, describe each separately. If no non-textual diagram/chart is \
actually present on this page, say so explicitly instead of inventing one.
"""


def render_page_png(pdf_path: Path, page_index: int, dpi: int = 150) -> bytes:
    import pymupdf

    doc = pymupdf.open(str(pdf_path))
    page = doc[page_index]
    pix = page.get_pixmap(dpi=dpi)
    return pix.tobytes("png")


def interpret_diagram_page(pdf_path: Path, page_index: int, settings: Settings | None = None) -> str:
    """Render a page to an image and ask Claude's vision capability to describe its
    diagram/chart content as plain text -- the one case native text extraction
    structurally cannot recover (vector-drawn boxes/columns linearize into a flat,
    unreadable word sequence; see README for the concrete example found on this
    document's operating-segments organizational chart, page 20).
    """
    settings = settings or get_settings()
    client = get_anthropic_client(settings)
    png_bytes = render_page_png(pdf_path, page_index)
    b64 = base64.standard_b64encode(png_bytes).decode("ascii")

    resp = client.messages.create(
        model=settings.anthropic_interpretation_model,
        max_tokens=800,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                    {"type": "text", "text": DIAGRAM_PROMPT},
                ],
            }
        ],
    )
    text = "".join(block.text for block in resp.content if block.type == "text")
    logger.info("Interpreted diagram on page %d (%d chars)", page_index + 1, len(text))
    return text
