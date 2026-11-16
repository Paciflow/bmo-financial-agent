from __future__ import annotations

import json
import logging
from pathlib import Path

import pymupdf

from bmo_agent.config import Settings, get_settings
from bmo_agent.models import HeadingNode

logger = logging.getLogger(__name__)


def build_heading_tree(pdf_path: Path, settings: Settings | None = None) -> list[HeadingNode]:
    """Build a flat list of HeadingNode from the PDF's embedded outline/TOC.

    Each node's start_page/end_page (0-based, inclusive) is derived from where the
    next heading at the same-or-shallower level begins. This gives exact, structural
    (non-LLM-inferred) document segmentation, so chunks stay aligned with the
    document's own headings instead of an inferred structure.
    """
    settings = settings or get_settings()
    doc = pymupdf.open(str(pdf_path))
    toc = doc.get_toc()  # list of [level, title, page_1_based]
    if not toc:
        raise ValueError("No embedded table of contents found in this PDF; heading segmentation requires one.")

    last_page_index = doc.page_count - 1

    nodes: list[HeadingNode] = []
    parent_stack: list[tuple[int, str]] = []  # (level, id)

    for idx, (level, title, page_1_based) in enumerate(toc):
        start_page = max(0, page_1_based - 1)
        node_id = f"h{idx:03d}"

        while parent_stack and parent_stack[-1][0] >= level:
            parent_stack.pop()
        parent_id = parent_stack[-1][1] if parent_stack else None
        path = title if parent_id is None else f"{_path_for(nodes, parent_id)} > {title}"

        nodes.append(
            HeadingNode(
                id=node_id,
                title=title.strip(),
                level=level,
                start_page=start_page,
                end_page=last_page_index,  # placeholder; corrected by _fix_end_pages below
                parent_id=parent_id,
                path=path,
            )
        )
        parent_stack.append((level, node_id))

    _fix_end_pages(nodes, last_page_index)

    out_path = settings.processed_dir / "headings.json"
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps([n.model_dump() for n in nodes], indent=2), encoding="utf-8")
    logger.info("Built %d heading nodes -> %s", len(nodes), out_path)
    return nodes


def _path_for(nodes: list[HeadingNode], node_id: str) -> str:
    for n in nodes:
        if n.id == node_id:
            return n.path
    return ""


def _fix_end_pages(nodes: list[HeadingNode], last_page_index: int) -> None:
    """Recompute end_page per node as (start of next node with level<=this level) - 1,
    correctly handling nested children whose own end_page must not exceed the parent's.
    """
    n = len(nodes)
    for i, node in enumerate(nodes):
        end_page = last_page_index
        for j in range(i + 1, n):
            if nodes[j].level <= node.level:
                end_page = max(node.start_page, nodes[j].start_page - 1)
                break
        node.end_page = end_page


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    settings = get_settings()
    pdf_path = Path(sys.argv[1]) if len(sys.argv) > 1 else settings.raw_dir / f"{settings.document_id}.pdf"
    heads = build_heading_tree(pdf_path, settings)
    for h in heads:
        print(f"{'  ' * h.level}[{h.id}] p{h.start_page}-{h.end_page} {h.title}")
