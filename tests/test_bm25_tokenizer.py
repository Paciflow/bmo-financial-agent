from bmo_agent.indexing.bm25_index import tokenize


def test_tokenize_preserves_numbers_with_commas_and_decimals():
    tokens = tokenize("Revenue was $36,274 million, up 10.6% from $32,795 million.")
    assert "36,274" in tokens
    assert "10.6%" in tokens
    assert "32,795" in tokens


def test_tokenize_preserves_acronyms_and_table_ids():
    tokens = tokenize("See TABLE 37 for OSFI's B-20 Guideline and CET1 ratio.")
    assert "table" in tokens
    assert "37" in tokens
    assert "osfi's" in tokens
    assert "cet1" in tokens


def test_tokenize_lowercases():
    tokens = tokenize("BMO Financial Group")
    assert tokens == ["bmo", "financial", "group"]
