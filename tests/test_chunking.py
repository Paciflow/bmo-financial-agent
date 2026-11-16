from bmo_agent.chunking.chunker import (
    NARRATIVE_CHUNK_SIZE,
    NARRATIVE_OVERLAP,
    _chunk_narrative_section,
    _chunk_table,
    _detect_header_row_count,
)
from bmo_agent.chunking.text_cleanup import merge_number_fragments
from bmo_agent.models import HeadingNode, PageText, TableBlock

DOC_ID = "test_doc"


def _heading(id_, title, level, start, end, parent=None):
    return HeadingNode(id=id_, title=title, level=level, start_page=start, end_page=end, parent_id=parent, path=title)


def test_narrative_chunking_respects_heading_boundaries_and_overlap():
    long_text_a = "Section A sentence. " * 100  # long enough to span multiple chunks
    long_text_b = "Section B sentence. " * 100

    heading_a = _heading("h0", "Section A", 1, 0, 0)
    heading_b = _heading("h1", "Section B", 1, 1, 1)
    pages = [
        PageText(page_index=0, text=long_text_a, char_count=len(long_text_a)),
        PageText(page_index=1, text=long_text_b, char_count=len(long_text_b)),
    ]

    a_chunks = _chunk_narrative_section(heading_a, pages, DOC_ID)
    b_chunks = _chunk_narrative_section(heading_b, pages, DOC_ID)

    assert len(a_chunks) > 1  # text was long enough to split
    assert all("Section B" not in c.text for c in a_chunks)  # no boundary crossing
    assert all("Section A" not in c.text for c in b_chunks)

    for c in a_chunks:
        assert len(c.text) <= NARRATIVE_CHUNK_SIZE

    # consecutive chunks should overlap
    overlap_tail = a_chunks[0].text[-NARRATIVE_OVERLAP:]
    assert overlap_tail[:50] in a_chunks[1].text


def test_table_chunking_repeats_header_and_splits_large_tables():
    header = ["Region", "2025", "2024"]
    data_rows = [[f"Row{i}", str(i * 10), str(i * 9)] for i in range(20)]
    table = TableBlock(
        table_id="t0001",
        page_index=5,
        bbox=(0, 0, 100, 100),
        n_rows=1 + len(data_rows),
        n_cols=3,
        raw_cells=[header] + data_rows,
        heading_id="h0",
        heading_path="Some Section",
    )

    chunks = _chunk_table(table, DOC_ID)

    assert len(chunks) > 1  # 20 data rows should be split across multiple chunks
    for c in chunks:
        assert c.table_id == "t0001"
        assert "Region" in c.text  # header repeated in every chunk


def test_small_table_is_single_chunk():
    table = TableBlock(
        table_id="t0002",
        page_index=1,
        bbox=(0, 0, 10, 10),
        n_rows=2,
        n_cols=2,
        raw_cells=[["Metric", "Value"], ["Revenue", "100"]],
        heading_id=None,
        heading_path=None,
    )
    chunks = _chunk_table(table, DOC_ID)
    assert len(chunks) == 1
    assert "Revenue" in chunks[0].text


def test_header_detection_does_not_swallow_data_rows():
    # Regression test: a single-header-row table whose section label row ("Reported
    # Results") has empty values must not have the first real data row ("Net interest
    # income") swallowed into a fixed 3-row header -- that caused "Adjusted Results"
    # figures to be mislabeled under a repeated "Reported Results" tag downstream.
    rows = [
        ["(Canadian $ in millions)", "2025", "2024"],
        ["Reported Results", "", ""],
        ["Net interest income", "21487", "19468"],
        ["Revenue", "36274", "32795"],
    ]
    assert _detect_header_row_count(rows) == 2


def test_header_detection_handles_multi_level_headers():
    rows = [
        ["", "", "As at October 31 2025"],
        ["(Canadian $ in millions)", "", "Outstanding balances"],
        ["Region", "Insured", "Uninsured"],
        ["Atlantic", "3274", "4024"],
    ]
    assert _detect_header_row_count(rows) == 3


def test_merge_number_fragments_joins_lone_separator_lines():
    assert merge_number_fragments("36274\n,") == "36274,"
    assert merge_number_fragments("1144\n.") == "1144."
    assert merge_number_fragments("Revenue") == "Revenue"
    assert merge_number_fragments("") == ""
    assert merge_number_fragments(None) == ""


def test_clean_table_cell_segments_glued_label_words():
    from bmo_agent.chunking.text_cleanup import clean_table_cell

    assert clean_table_cell("ReportedResults") == "Reported Results"
    assert clean_table_cell("Netinterestincome") == "Net interest income"


def test_clean_table_cell_preserves_negative_value_parens():
    from bmo_agent.chunking.text_cleanup import clean_table_cell

    # (164) is a negative value in accounting notation -- must never be stripped
    # to "164", which would silently flip its sign/meaning.
    assert clean_table_cell("(164)") == "(164)"


def test_clean_table_cell_merges_split_numbers_too():
    from bmo_agent.chunking.text_cleanup import clean_table_cell

    assert clean_table_cell("36274\n,") == "36274,"


def test_narrative_chunk_page_attribution_uses_midpoint_not_start():
    # Regression test: a chunk that starts one character into page 0 (its tail end)
    # and is almost entirely page 1's content must be attributed to page 1, not
    # page 0 -- a real citation bug found in testing (a chunk tagged "page 14" whose
    # actual cited table was on page 15).
    page0_text = "END"  # page 0 contributes only 3 chars before the chunk starts
    page1_text = "Revenue figures follow. " * 80  # long page 1 content
    heading = _heading("h0", "Section", 1, 0, 1)
    pages = [
        PageText(page_index=0, text=page0_text, char_count=len(page0_text)),
        PageText(page_index=1, text=page1_text, char_count=len(page1_text)),
    ]
    chunks = _chunk_narrative_section(heading, pages, DOC_ID)
    assert chunks[0].page_index == 1
