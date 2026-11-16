from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class PageText(BaseModel):
    page_index: int  # 0-based PDF page index
    printed_label: str | None = None
    text: str
    char_count: int


class HeadingNode(BaseModel):
    id: str
    title: str
    level: int
    start_page: int  # 0-based, inclusive
    end_page: int  # 0-based, inclusive
    parent_id: str | None = None
    path: str  # e.g. "Enterprise-Wide Risk Management > Credit and Counterparty Risk"


class TableBlock(BaseModel):
    table_id: str
    page_index: int
    bbox: tuple[float, float, float, float]
    n_rows: int
    n_cols: int
    raw_cells: list[list[str | None]]
    heading_id: str | None = None
    heading_path: str | None = None


ContentType = Literal["narrative", "table", "diagram", "chart", "footnote"]


class FinancialFact(BaseModel):
    """A single extracted/interpreted numeric fact from a financial table."""

    metric: str
    value: float | None
    original_value_string: str
    unit: str | None = None
    year: str | None = None
    period: str | None = None
    operating_segment: str | None = None
    accounting_basis: Literal["reported", "adjusted", "unspecified"] = "unspecified"
    table_id: str
    source_page: int
    confidence: Literal["high", "medium", "low"] = "high"
    note: str | None = None


class ContentBlock(BaseModel):
    block_id: str
    document_id: str
    content_type: ContentType
    heading_id: str | None = None
    heading_path: str | None = None
    page_index: int
    text: str
    table_id: str | None = None
    facts: list[FinancialFact] = []


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    content_type: ContentType
    heading_id: str | None = None
    heading_path: str | None = None
    page_index: int
    page_label: str | None = None
    text: str
    table_id: str | None = None
    source_block_ids: list[str] = []
