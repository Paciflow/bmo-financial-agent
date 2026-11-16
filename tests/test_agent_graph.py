from bmo_agent.agent.graph import _referenced_pages


def test_referenced_pages_single_citation():
    assert _referenced_pages("Revenue grew 10.6% (p. 10).") == {10}


def test_referenced_pages_multiple_citations():
    text = "Reported revenue was $36,274M (p. 10) while net income was $8,725M (p. 10) and ROE fell (p. 23)."
    assert _referenced_pages(text) == {10, 23}


def test_referenced_pages_grouped_citation():
    assert _referenced_pages("See the summary table (pp. 10, 41).") == {10, 41}


def test_referenced_pages_none_when_no_citation_present():
    # This is the case that matters most: an honest "I can't find this" answer
    # must not look like it cited anything.
    text = "The evidence doesn't give BMO's closing share price on that date, so I can't answer that."
    assert _referenced_pages(text) == set()


def test_referenced_pages_catches_citation_not_at_start_of_parens():
    # Regression: found during benchmark review -- "(Table 21, p. 38)" was
    # silently dropped because the old regex required "(p." literally at the
    # start of the parenthetical, stripping a citation the model got right.
    text = "Reported net income by quarter (Table 21, p. 38) shows Q1 at $2,138 million."
    assert _referenced_pages(text) == {38}


def test_referenced_pages_without_parens():
    assert _referenced_pages("This is confirmed on p. 39 of the report.") == {39}


def test_referenced_pages_no_false_positive_on_abbreviations():
    text = "Corp. revenue rose and GAAP. figures were restated for the Group. 14 subsidiaries."
    assert _referenced_pages(text) == set()
