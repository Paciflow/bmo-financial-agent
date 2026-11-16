from __future__ import annotations

import re

_HAS_LETTER_RE = re.compile(r"[A-Za-z]")


def _merge_lines(cell: str | None) -> list[str]:
    """Split a cell on its internal newlines and merge a lone "," or "." line back
    onto the number immediately before it -- PyMuPDF's table extraction on this PDF
    sometimes isolates a thousands-separator or decimal point onto its own line
    (e.g. "36274\\n," instead of "36,274"); that's unambiguous to repair.
    """
    if not cell:
        return []
    merged: list[str] = []
    for raw_line in cell.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        if line in (",", ".") and merged:
            merged[-1] = merged[-1] + line
        else:
            merged.append(line)
    return merged


def merge_number_fragments(cell: str | None) -> str:
    """Public, test-covered contract: merge split number fragments only, no word
    segmentation. Used wherever we want the safe/minimal cleanup.
    """
    return " ".join(_merge_lines(cell))


def _split_glued_words(token: str) -> str:
    """Best-effort de-gluing of concatenated label text (e.g. "Netinterestincome"
    -> "Net interest income") via dictionary-based word segmentation. Only ever
    applied to tokens that contain letters -- numeric tokens (including negative
    values like "(164)") are never touched, so signs/parens are preserved exactly.
    Not perfect (uncommon words like "divestitures" can mis-segment), which is why
    this is used for retrieval-text rendering only, never for the stored audit
    data in tables.json.
    """
    if not token or not _HAS_LETTER_RE.search(token):
        return token
    try:
        import wordninja

        words = wordninja.split(token)
        return " ".join(words) if words else token
    except Exception:  # noqa: BLE001
        return token


def clean_table_cell(cell: str | None) -> str:
    """Full rendering-time cleanup for a table cell: merge split number fragments,
    then de-glue concatenated label words token by token. The original raw_cells
    in tables.json are never modified -- this only affects display/retrieval text.
    """
    tokens = _merge_lines(cell)
    return " ".join(_split_glued_words(t) for t in tokens)
