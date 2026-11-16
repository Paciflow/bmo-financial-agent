from bmo_agent.retrieval.rrf import rrf_fuse_scores


def test_rrf_combines_both_lists():
    dense = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
    bm25 = [("b", 5.0), ("d", 3.0)]
    scores = rrf_fuse_scores(dense, bm25, k=60)

    assert set(scores) == {"a", "b", "c", "d"}
    # "b" appears in both lists (dense rank 2, bm25 rank 1) so should outscore
    # items that only appear once.
    assert scores["b"] > scores["a"]
    assert scores["b"] > scores["d"]


def test_rrf_rank_one_scores_higher_than_rank_two():
    dense = [("a", 1.0), ("b", 0.5)]
    scores = rrf_fuse_scores(dense, [], k=60)
    assert scores["a"] > scores["b"]


def test_rrf_empty_inputs():
    assert rrf_fuse_scores([], [], k=60) == {}


def test_rrf_k_dampens_rank_difference():
    dense = [("a", 1.0), ("b", 0.9)]
    low_k = rrf_fuse_scores(dense, [], k=1)
    high_k = rrf_fuse_scores(dense, [], k=1000)
    # With a tiny k, rank-1 vs rank-2 differ a lot in relative terms; with a huge k
    # the gap nearly disappears.
    low_k_ratio = low_k["a"] / low_k["b"]
    high_k_ratio = high_k["a"] / high_k["b"]
    assert low_k_ratio > high_k_ratio > 1.0
