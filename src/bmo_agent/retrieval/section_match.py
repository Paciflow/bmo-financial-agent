from __future__ import annotations

import json
import logging

from rank_bm25 import BM25Okapi

from bmo_agent.config import Settings, get_settings
from bmo_agent.indexing.bm25_index import tokenize

logger = logging.getLogger(__name__)


class SectionMatcher:
    """Identifies which document section a query is about, for boosting that
    section's chunks in retrieval -- built to fix a real failure (short company
    facts like "total assets" losing to unrelated table chunks on raw term
    frequency; see chunking/README notes) without the regression a dense-only
    version caused (it also scored an irrelevant section as a *better* match for
    a Capital Markets query than the Capital Markets section itself).

    Two independent signals, used only when they agree:
    - BM25 over the 55 section *titles* (cheap, literal keyword match --
      "Capital Markets" in the query matches the "Capital Markets" title
      directly, with none of an embedding model's topical drift).
    - Dense search over the bmo_sections collection (semantic, catches queries
      that don't share exact words with the title, e.g. "total assets").

    Verified empirically before shipping: for "total assets" and "client count"
    both signals agree on "About BMO" (safe to boost); for "Capital Markets
    revenue" they disagree (BM25 correctly says Capital Markets, dense still says
    About BMO) so no boost fires -- the exact case that caused the dense-only
    version to regress. The cost is conservatism: a query where BM25 is right but
    dense disagrees (observed once, for a Wealth Management query) gets no boost
    either. That's an acceptable trade for not re-introducing the regression.
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        headings = json.loads((self.settings.processed_dir / "headings.json").read_text())
        self.heading_ids = [h["id"] for h in headings]
        self.titles = [h["title"] for h in headings]
        self._bm25 = BM25Okapi([tokenize(t) for t in self.titles])

    def _top_bm25_title(self, query: str) -> str | None:
        scores = self._bm25.get_scores(tokenize(query))
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        if scores[best_idx] <= 0:
            return None
        return self.heading_ids[best_idx]

    def _top_dense_section(self, query: str, embedder, client) -> str | None:
        vec = embedder.embed_one(query, is_query=True)
        hits = client.query_points(collection_name=self.settings.qdrant_sections_collection, query=vec, limit=1).points
        return hits[0].payload["heading_id"] if hits else None

    def matched_heading_id(self, query: str, embedder, client) -> str | None:
        """Returns the heading_id to boost, or None if the two signals disagree
        (or neither found a confident match)."""
        bm25_match = self._top_bm25_title(query)
        if bm25_match is None:
            return None
        dense_match = self._top_dense_section(query, embedder, client)
        if bm25_match == dense_match:
            return bm25_match
        return None
