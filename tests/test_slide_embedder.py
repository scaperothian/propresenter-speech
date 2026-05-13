"""
Unit tests for SlideEmbedder (sentence-transformers dense + optional BM25).

BM25-only tests (bm25_weight=1.0) need no mocks — rank_bm25 is pure Python.
Dense-only tests (bm25_weight=0.0) patch sentence_transformers.SentenceTransformer.
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


def _dense_embedder(slide_texts: list[str], slide_vecs: list[np.ndarray], query_vec: np.ndarray) -> SlideEmbedder:
    """Build a dense-only embedder (bm25_weight=0.0) with mocked SentenceTransformer."""
    mock_model = MagicMock()
    # encode() is called twice: once during build (slide matrix), once per find_slide (query row)
    mock_model.encode.side_effect = [
        np.array(slide_vecs, dtype=np.float32),
        np.array([query_vec], dtype=np.float32),
    ]
    embedder = SlideEmbedder(bm25_weight=0.0)
    with patch("sentence_transformers.SentenceTransformer", return_value=mock_model):
        embedder.load()
    embedder._model = mock_model
    embedder.build(slide_texts)
    return embedder


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_raises_on_bm25_weight_above_one(self):
        with pytest.raises(ValueError):
            SlideEmbedder(bm25_weight=1.1)

    def test_raises_on_negative_bm25_weight(self):
        with pytest.raises(ValueError):
            SlideEmbedder(bm25_weight=-0.1)


# ---------------------------------------------------------------------------
# build()
# ---------------------------------------------------------------------------

class TestBuild:
    def test_slide_count_matches_input(self):
        embedder = SlideEmbedder(bm25_weight=1.0)
        embedder.build(["slide one", "slide two", "slide three"])
        assert embedder.slide_count == 3

    def test_bm25_only_skips_dense_model(self):
        embedder = SlideEmbedder(bm25_weight=1.0)
        embedder.build(["hello world", "foo bar"])
        assert embedder._embeddings is None
        assert embedder._model is None

    def test_dense_embeddings_built_when_weight_below_one(self):
        vecs = np.array([np.ones(4, dtype=np.float32), np.zeros(4, dtype=np.float32)])
        mock_model = MagicMock()
        mock_model.encode.return_value = vecs
        embedder = SlideEmbedder(bm25_weight=0.0)
        with patch("sentence_transformers.SentenceTransformer", return_value=mock_model):
            embedder.load()
        embedder._model = mock_model
        embedder.build(["a", "b"])
        assert embedder._embeddings.shape == (2, 4)

    def test_raises_on_empty_texts(self):
        embedder = SlideEmbedder(bm25_weight=1.0)
        with pytest.raises(ValueError):
            embedder.build([])

    def test_raises_when_indices_length_mismatches_texts(self):
        embedder = SlideEmbedder(bm25_weight=1.0)
        with pytest.raises(ValueError):
            embedder.build(["only one text"], slide_indices=[0, 1])

    def test_custom_slide_indices_stored(self):
        embedder = SlideEmbedder(bm25_weight=1.0)
        embedder.build(["slide with text"], slide_indices=[2])
        assert embedder._slide_indices == [2]

    def test_default_indices_are_sequential(self):
        embedder = SlideEmbedder(bm25_weight=1.0)
        embedder.build(["a", "b", "c"])
        assert embedder._slide_indices == [0, 1, 2]


# ---------------------------------------------------------------------------
# find_slide() — BM25-only (no mocking needed)
# ---------------------------------------------------------------------------

class TestFindSlideBM25:
    def test_returns_minus_one_before_build(self):
        embedder = SlideEmbedder(bm25_weight=1.0)
        idx, score = embedder.find_slide("anything")
        assert idx == -1
        assert score == 0.0

    def test_returns_minus_one_for_empty_query(self):
        embedder = SlideEmbedder(bm25_weight=1.0)
        embedder.build(["hello world"])
        idx, _ = embedder.find_slide("   ")
        assert idx == -1

    def test_prefers_slide_with_unique_matching_word(self):
        # "went" is diagnostic: present in slide 1, absent in slide 0
        embedder = SlideEmbedder(bm25_weight=1.0)
        embedder.build(["mary mary mary", "everywhere that mary went"])
        idx, _ = embedder.find_slide("mary went")
        assert idx == 1

    def test_score_in_range_zero_to_one(self):
        embedder = SlideEmbedder(bm25_weight=1.0)
        embedder.build(["hello world", "foo bar baz"])
        _, score = embedder.find_slide("hello world")
        assert 0.0 <= score <= 1.0

    def test_returns_real_propresenter_index(self):
        # Slide 1 was empty and skipped; real indices are [0, 2, 3].
        # Three-doc corpus keeps IDF > 0 so BM25 can discriminate.
        embedder = SlideEmbedder(bm25_weight=1.0)
        embedder.build(["mary went", "john ran", "something else"], slide_indices=[0, 2, 3])
        idx, _ = embedder.find_slide("john ran")
        assert idx == 2

    def test_no_match_gives_zero_score(self):
        embedder = SlideEmbedder(bm25_weight=1.0)
        embedder.build(["hello world", "foo bar"])
        _, score = embedder.find_slide("xyz qrs")
        assert score == 0.0


# ---------------------------------------------------------------------------
# find_slide() — dense-only (bm25_weight=0.0, mocked fastembed)
# ---------------------------------------------------------------------------

class TestFindSlideDense:
    def test_picks_closest_by_cosine(self):
        v0 = _unit(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        v1 = _unit(np.array([0.0, 1.0, 0.0], dtype=np.float32))
        query = _unit(np.array([0.1, 0.9, 0.0], dtype=np.float32))
        embedder = _dense_embedder(["slide a", "slide b"], [v0, v1], query)
        idx, score = embedder.find_slide("query")
        assert idx == 1
        assert score > 0.8

    def test_identical_vector_scores_one(self):
        v = _unit(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        other = _unit(np.array([0.0, 1.0, 0.0], dtype=np.float32))
        embedder = _dense_embedder(["slide a", "slide b"], [v, other], v)
        idx, score = embedder.find_slide("query")
        assert idx == 0
        assert pytest.approx(score, abs=1e-5) == 1.0

    def test_score_clipped_to_zero_one(self):
        v0 = _unit(np.random.rand(32).astype(np.float32))
        query = _unit(np.random.rand(32).astype(np.float32))
        embedder = _dense_embedder(["slide a"], [v0], query)
        _, score = embedder.find_slide("query")
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# find_slide_with_margin()
# ---------------------------------------------------------------------------

class TestFindSlideWithMargin:
    def test_returns_minus_one_before_build(self):
        embedder = SlideEmbedder(bm25_weight=1.0)
        idx, score, margin = embedder.find_slide_with_margin("anything")
        assert idx == -1
        assert score == 0.0
        assert margin == 0.0

    def test_margin_is_zero_for_single_slide(self):
        embedder = SlideEmbedder(bm25_weight=1.0)
        embedder.build(["allegiance to the flag"])
        _, score, margin = embedder.find_slide_with_margin("allegiance flag")
        # single slide: margin equals best score
        assert margin == pytest.approx(score, abs=1e-6)

    def test_clear_winner_has_large_margin(self):
        # "everywhere" is unique to slide 1; other slides share no term
        embedder = SlideEmbedder(bm25_weight=1.0)
        embedder.build([
            "mary mary mary",
            "everywhere that mary went",
            "her lamb was sure to go",
        ])
        _, _, margin = embedder.find_slide_with_margin("everywhere that mary")
        assert margin > 0.1

    def test_ambiguous_query_has_small_margin(self):
        # query "mary" matches all slides equally → small margin
        embedder = SlideEmbedder(bm25_weight=1.0)
        embedder.build([
            "mary mary mary",
            "everywhere that mary went",
            "her lamb was sure to go",
        ])
        _, _, margin = embedder.find_slide_with_margin("mary")
        # IDF of "mary" across 3 docs where it appears in 2 is the same for both
        # slides that contain it; the margin should be small
        assert margin < 0.5  # not a clear winner
