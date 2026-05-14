"""
Unit tests for SlideEmbedder (sentence-transformers dense embeddings).

All tests patch sentence_transformers.SentenceTransformer so no model
download or GPU is required.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from propresenter_speech.slide_embedder import SlideEmbedder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _make_embedder(slide_texts: list[str], slide_vecs: list[np.ndarray], query_vec: np.ndarray) -> SlideEmbedder:
    """Build a SlideEmbedder with a mocked SentenceTransformer."""
    mock_model = MagicMock()
    # encode() is called twice: once during build (slide matrix), once per find_slide (query row)
    mock_model.encode.side_effect = [
        np.array(slide_vecs, dtype=np.float32),
        np.array([query_vec], dtype=np.float32),
    ]
    embedder = SlideEmbedder()
    with patch("sentence_transformers.SentenceTransformer", return_value=mock_model):
        embedder.load()
    embedder._model = mock_model
    embedder.build(slide_texts)
    return embedder


# ---------------------------------------------------------------------------
# build()
# ---------------------------------------------------------------------------

class TestBuild:
    def _mock_embedder(self, slide_texts: list[str]) -> SlideEmbedder:
        mock_model = MagicMock()
        mock_model.encode.return_value = np.ones((len(slide_texts), 4), dtype=np.float32)
        embedder = SlideEmbedder()
        with patch("sentence_transformers.SentenceTransformer", return_value=mock_model):
            embedder.load()
        embedder._model = mock_model
        embedder.build(slide_texts)
        return embedder

    def test_slide_count_matches_input(self):
        embedder = self._mock_embedder(["slide one", "slide two", "slide three"])
        assert embedder.slide_count == 3

    def test_embeddings_shape(self):
        vecs = np.array([np.ones(4, dtype=np.float32), np.zeros(4, dtype=np.float32)])
        mock_model = MagicMock()
        mock_model.encode.return_value = vecs
        embedder = SlideEmbedder()
        with patch("sentence_transformers.SentenceTransformer", return_value=mock_model):
            embedder.load()
        embedder._model = mock_model
        embedder.build(["a", "b"])
        assert embedder._embeddings.shape == (2, 4)

    def test_raises_on_empty_texts(self):
        embedder = SlideEmbedder()
        with pytest.raises(ValueError):
            embedder.build([])

    def test_raises_when_indices_length_mismatches_texts(self):
        embedder = SlideEmbedder()
        with pytest.raises(ValueError):
            embedder.build(["only one text"], slide_indices=[0, 1])

    def test_custom_slide_indices_stored(self):
        embedder = self._mock_embedder(["slide with text"])
        # override after build to check storage
        embedder2 = SlideEmbedder()
        mock_model = MagicMock()
        mock_model.encode.return_value = np.ones((1, 4), dtype=np.float32)
        with patch("sentence_transformers.SentenceTransformer", return_value=mock_model):
            embedder2.load()
        embedder2._model = mock_model
        embedder2.build(["slide with text"], slide_indices=[2])
        assert embedder2._slide_indices == [2]

    def test_default_indices_are_sequential(self):
        embedder = self._mock_embedder(["a", "b", "c"])
        assert embedder._slide_indices == [0, 1, 2]


# ---------------------------------------------------------------------------
# find_slide()
# ---------------------------------------------------------------------------

class TestFindSlide:
    def test_returns_minus_one_before_build(self):
        embedder = SlideEmbedder()
        idx, score = embedder.find_slide("anything")
        assert idx == -1
        assert score == 0.0

    def test_returns_minus_one_for_empty_query(self):
        mock_model = MagicMock()
        mock_model.encode.return_value = np.ones((1, 4), dtype=np.float32)
        embedder = SlideEmbedder()
        with patch("sentence_transformers.SentenceTransformer", return_value=mock_model):
            embedder.load()
        embedder._model = mock_model
        embedder.build(["hello world"])
        idx, _ = embedder.find_slide("   ")
        assert idx == -1

    def test_picks_closest_by_cosine(self):
        v0 = _unit(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        v1 = _unit(np.array([0.0, 1.0, 0.0], dtype=np.float32))
        query = _unit(np.array([0.1, 0.9, 0.0], dtype=np.float32))
        embedder = _make_embedder(["slide a", "slide b"], [v0, v1], query)
        idx, score = embedder.find_slide("query")
        assert idx == 1
        assert score > 0.8

    def test_identical_vector_scores_one(self):
        v = _unit(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        other = _unit(np.array([0.0, 1.0, 0.0], dtype=np.float32))
        embedder = _make_embedder(["slide a", "slide b"], [v, other], v)
        idx, score = embedder.find_slide("query")
        assert idx == 0
        assert pytest.approx(score, abs=1e-5) == 1.0

    def test_score_clipped_to_zero_one(self):
        v0 = _unit(np.random.rand(32).astype(np.float32))
        query = _unit(np.random.rand(32).astype(np.float32))
        embedder = _make_embedder(["slide a"], [v0], query)
        _, score = embedder.find_slide("query")
        assert 0.0 <= score <= 1.0

    def test_returns_real_propresenter_index(self):
        v0 = _unit(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        v1 = _unit(np.array([0.0, 1.0, 0.0], dtype=np.float32))
        v2 = _unit(np.array([0.0, 0.0, 1.0], dtype=np.float32))
        query = _unit(np.array([0.0, 1.0, 0.0], dtype=np.float32))
        mock_model = MagicMock()
        mock_model.encode.side_effect = [
            np.array([v0, v1, v2], dtype=np.float32),
            np.array([query], dtype=np.float32),
        ]
        embedder = SlideEmbedder()
        with patch("sentence_transformers.SentenceTransformer", return_value=mock_model):
            embedder.load()
        embedder._model = mock_model
        embedder.build(["a", "b", "c"], slide_indices=[0, 2, 3])
        idx, _ = embedder.find_slide("b")
        assert idx == 2


# ---------------------------------------------------------------------------
# find_slide_with_margin()
# ---------------------------------------------------------------------------

class TestFindSlideWithMargin:
    def test_returns_minus_one_before_build(self):
        embedder = SlideEmbedder()
        idx, score, margin = embedder.find_slide_with_margin("anything")
        assert idx == -1
        assert score == 0.0
        assert margin == 0.0

    def test_margin_is_score_for_single_slide(self):
        v = _unit(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        query = _unit(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        embedder = _make_embedder(["only slide"], [v], query)
        _, score, margin = embedder.find_slide_with_margin("query")
        assert margin == pytest.approx(score, abs=1e-6)

    def test_clear_winner_has_large_margin(self):
        v0 = _unit(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        v1 = _unit(np.array([0.0, 1.0, 0.0], dtype=np.float32))
        # query almost identical to v1
        query = _unit(np.array([0.01, 0.99, 0.0], dtype=np.float32))
        embedder = _make_embedder(["slide a", "slide b"], [v0, v1], query)
        _, _, margin = embedder.find_slide_with_margin("query")
        assert margin > 0.1

    def test_ambiguous_query_has_small_margin(self):
        # two slides with nearly identical direction → small margin
        v0 = _unit(np.array([1.0, 1.0, 0.0], dtype=np.float32))
        v1 = _unit(np.array([1.0, 1.01, 0.0], dtype=np.float32))
        query = _unit(np.array([1.0, 1.0, 0.0], dtype=np.float32))
        embedder = _make_embedder(["slide a", "slide b"], [v0, v1], query)
        _, _, margin = embedder.find_slide_with_margin("query")
        assert margin < 0.1
