from unittest.mock import MagicMock

import pytest

from bmo_agent.config import Settings
from bmo_agent.retrieval.section_match import SectionMatcher


@pytest.fixture
def matcher(tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir()
    headings = [
        {"id": "h003", "title": "About BMO"},
        {"id": "h015", "title": "Capital Markets"},
        {"id": "h014", "title": "Wealth Management"},
    ]
    import json

    (processed / "headings.json").write_text(json.dumps(headings))

    class FakeSettings(Settings):
        @property
        def processed_dir(self):
            return processed

    return SectionMatcher(FakeSettings())


def _fake_embedder():
    e = MagicMock()
    e.embed_one.return_value = [0.0]
    return e


def _fake_client(top_heading_id: str | None):
    client = MagicMock()
    if top_heading_id is None:
        client.query_points.return_value.points = []
    else:
        hit = MagicMock()
        hit.payload = {"heading_id": top_heading_id}
        client.query_points.return_value.points = [hit]
    return client


def test_agreement_returns_matched_heading(matcher):
    # BM25 over titles will clearly favor "About BMO" for this query.
    client = _fake_client("h003")
    result = matcher.matched_heading_id("What is BMO total assets?", _fake_embedder(), client)
    assert result == "h003"


def test_disagreement_returns_none(matcher):
    # BM25-title will favor "Capital Markets" (literal keyword match); dense
    # (mocked) disagrees -- this is the exact case that caused the dense-only
    # version to regress, so disagreement must suppress the boost.
    client = _fake_client("h003")
    result = matcher.matched_heading_id("What is BMO Capital Markets revenue?", _fake_embedder(), client)
    assert result is None


def test_no_keyword_overlap_returns_none(matcher):
    client = _fake_client("h003")
    result = matcher.matched_heading_id("xyz unrelated gibberish query", _fake_embedder(), client)
    assert result is None
