from unittest.mock import MagicMock, patch

from bmo_agent.config import Settings
from bmo_agent.embeddings.embedder import Embedder


def _fake_settings(**overrides):
    defaults = {
        "embedding_model": "fake-model",
        "embedding_dim": 8,
        "embedding_query_prefix": "Represent this sentence for searching relevant passages: ",
    }
    defaults.update(overrides)
    return Settings(**defaults)


def test_query_prefix_applied_only_when_is_query_true():
    fake_model = MagicMock()
    fake_model.encode.return_value = [[0.0] * 8]
    with patch("bmo_agent.embeddings.embedder._load_model", return_value=fake_model):
        embedder = Embedder(_fake_settings())

        embedder.embed_one("What was revenue?", is_query=True)
        texts_passed = fake_model.encode.call_args[0][0]
        assert texts_passed[0].startswith("Represent this sentence for searching relevant passages: ")
        assert texts_passed[0].endswith("What was revenue?")

        embedder.embed_one("What was revenue?", is_query=False)
        texts_passed = fake_model.encode.call_args[0][0]
        assert texts_passed[0] == "What was revenue?"


def test_no_prefix_configured_means_no_change_even_for_queries():
    fake_model = MagicMock()
    fake_model.encode.return_value = [[0.0] * 8]
    with patch("bmo_agent.embeddings.embedder._load_model", return_value=fake_model):
        embedder = Embedder(_fake_settings(embedding_query_prefix=""))
        embedder.embed_one("What was revenue?", is_query=True)
        texts_passed = fake_model.encode.call_args[0][0]
        assert texts_passed[0] == "What was revenue?"


def test_passages_never_get_the_query_prefix_even_in_batch():
    fake_model = MagicMock()
    fake_model.encode.return_value = [[0.0] * 8, [0.0] * 8]
    with patch("bmo_agent.embeddings.embedder._load_model", return_value=fake_model):
        embedder = Embedder(_fake_settings())
        embedder.embed(["chunk one", "chunk two"])  # is_query defaults to False
        texts_passed = fake_model.encode.call_args[0][0]
        assert texts_passed == ["chunk one", "chunk two"]
