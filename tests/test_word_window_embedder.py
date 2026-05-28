"""
Unit tests for WordWindowEmbedder.

All tests patch sentence_transformers.SentenceTransformer so no model
download or GPU is required.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from propresenter_speech.slide_embedder import WordWindowEmbedder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _make_embedder(
    slides: list[tuple[int, str]],
    context_words: int,
    window_vecs: list[np.ndarray],
    query_vec: np.ndarray,
    stride: int = 1,
) -> WordWindowEmbedder:
    """Build a WordWindowEmbedder with a mocked SentenceTransformer."""
    mock_model = MagicMock()
    mock_model.encode.side_effect = [
        np.array(window_vecs, dtype=np.float32),   # called during build()
        np.array([query_vec], dtype=np.float32),    # called during find_slide*()
    ]
    embedder = WordWindowEmbedder(stride=stride)
    with patch("sentence_transformers.SentenceTransformer", return_value=mock_model):
        embedder.load()
    embedder._model = mock_model
    embedder.build(slides, context_words=context_words)
    return embedder


def _slides_from(*texts: str) -> list[tuple[int, str]]:
    """Helper: assign sequential slide indices."""
    return [(i, t) for i, t in enumerate(texts)]


# ---------------------------------------------------------------------------
# build() — window count and label assignment
# ---------------------------------------------------------------------------

class TestBuild:
    def _mock_build(self, slides, context_words, stride=1) -> WordWindowEmbedder:
        mock_model = MagicMock()

        def _encode(texts, **_kwargs):
            return np.ones((len(texts), 4), dtype=np.float32)

        mock_model.encode.side_effect = _encode
        embedder = WordWindowEmbedder(stride=stride)
        with patch("sentence_transformers.SentenceTransformer", return_value=mock_model):
            embedder.load()
        embedder._model = mock_model
        embedder.build(slides, context_words=context_words)
        return embedder

    def test_window_count_stride1(self):
        # "mary had a little lamb" = 5 words, context=3 → 3 windows
        embedder = self._mock_build(_slides_from("mary had a little lamb"), context_words=3)
        assert len(embedder._labels) == 3

    def test_window_count_stride2(self):
        # 5 words, context=3, stride=2 → windows at [0,1,2] and [2,3,4] = 2
        embedder = self._mock_build(
            _slides_from("mary had a little lamb"), context_words=3, stride=2
        )
        assert len(embedder._labels) == 2

    def test_window_count_exact_fit(self):
        # 3 words, context=3 → exactly 1 window
        embedder = self._mock_build(_slides_from("one two three"), context_words=3)
        assert len(embedder._labels) == 1

    def test_all_labels_from_single_slide(self):
        embedder = self._mock_build(_slides_from("one two three four five"), context_words=3)
        assert all(label == 0 for label in embedder._labels)

    def test_cross_boundary_label_transitions(self):
        # slide 0: "one two three"  slide 1: "four five six"
        # words: [one(0) two(0) three(0) four(1) five(1) six(1)]
        # context=3 windows (stride=1), labelled by first word's slide:
        #   [one two three]   → label 0  ← first word is slide 0
        #   [two three four]  → label 0  ← first word is slide 0
        #   [three four five] → label 0  ← first word is slide 0
        #   [four five six]   → label 1  ← first word is slide 1
        embedder = self._mock_build(
            _slides_from("one two three", "four five six"), context_words=3
        )
        assert embedder._labels[0] == 0
        assert embedder._labels[1] == 0
        assert embedder._labels[2] == 0
        assert embedder._labels[3] == 1

    def test_repeated_slide_produces_two_label_groups(self):
        # chorus appears twice: slide 0 = verse, slide 1 = chorus first,
        # slide 2 = verse2, slide 3 = chorus second
        slides = [
            (0, "verse one text"),
            (1, "chorus words here"),
            (2, "verse two text"),
            (3, "chorus words here"),  # same text, different idx
        ]
        embedder = self._mock_build(slides, context_words=2)
        # Labels 1 and 3 should both appear — chorus occupies two separate
        # positions in the continuum with their own indices.
        assert 1 in embedder._labels
        assert 3 in embedder._labels

    def test_raises_on_empty_slides(self):
        embedder = WordWindowEmbedder()
        with pytest.raises(ValueError, match="must not be empty"):
            embedder.build([], context_words=3)

    def test_raises_when_context_words_exceeds_total(self):
        embedder = WordWindowEmbedder()
        mock_model = MagicMock()
        with patch("sentence_transformers.SentenceTransformer", return_value=mock_model):
            embedder.load()
        embedder._model = mock_model
        with pytest.raises(ValueError, match="context_words"):
            embedder.build(_slides_from("only two words"), context_words=5)


# ---------------------------------------------------------------------------
# find_slide_with_margin() — returns slide_idx, not window position
# ---------------------------------------------------------------------------

class TestFindSlideWithMargin:
    def test_returns_minus_one_before_build(self):
        embedder = WordWindowEmbedder()
        idx, score, margin = embedder.find_slide_with_margin("anything")
        assert idx == -1
        assert score == 0.0
        assert margin == 0.0

    def test_returns_minus_one_for_empty_query(self):
        slides = _slides_from("hello world something else")
        vecs = [_unit(np.array([1.0, 0.0, 0.0], dtype=np.float32))] * 2
        embedder = WordWindowEmbedder()
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array(vecs, dtype=np.float32)
        with patch("sentence_transformers.SentenceTransformer", return_value=mock_model):
            embedder.load()
        embedder._model = mock_model
        embedder.build(slides, context_words=3)
        idx, _, _ = embedder.find_slide_with_margin("   ")
        assert idx == -1

    def test_returns_label_of_best_window_not_its_position(self):
        # slide 0: "alpha beta" (2 words), slide 1: "gamma delta" (2 words)
        # context=2, stride=1 → windows (first-word labelling):
        #   [alpha beta]  → label 0  ← first word "alpha" is slide 0
        #   [beta gamma]  → label 0  ← first word "beta" is slide 0
        #   [gamma delta] → label 1  ← first word "gamma" is slide 1
        # Give the third window ([gamma delta]) the highest cosine → should return label 1
        slides = [(0, "alpha beta"), (1, "gamma delta")]
        v_window0 = _unit(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        v_window1 = _unit(np.array([0.5, 0.5, 0.0], dtype=np.float32))
        v_window2 = _unit(np.array([0.0, 1.0, 0.0], dtype=np.float32))
        query = _unit(np.array([0.0, 1.0, 0.0], dtype=np.float32))  # closest to window2
        embedder = _make_embedder(
            slides, context_words=2,
            window_vecs=[v_window0, v_window1, v_window2],
            query_vec=query,
        )
        idx, score, _ = embedder.find_slide_with_margin("query")
        assert idx == 1   # label of window2 — slide 1
        assert score > 0.8

    def test_custom_slide_indices_returned_not_position(self):
        # Non-sequential slide indices: slides at positions 0,2,5 in ProPresenter
        slides = [(0, "first slide"), (2, "second slide"), (5, "third slide")]
        # 3 slides × 2 words each = 6 words, context=2, stride=1 → 5 windows
        # Last window [5th,6th words] → label=5
        vecs = [
            _unit(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)),
            _unit(np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)),
            _unit(np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)),
            _unit(np.array([0.0, 0.0, 0.5, 0.5], dtype=np.float32)),
            _unit(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)),
        ]
        # Query points at last window → should return slide idx 5
        query = _unit(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
        embedder = _make_embedder(slides, context_words=2, window_vecs=vecs, query_vec=query)
        idx, _, _ = embedder.find_slide_with_margin("query")
        assert idx == 5

    def test_margin_positive_for_clear_winner(self):
        slides = _slides_from("one two three four five six")
        v0 = _unit(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        v1 = _unit(np.array([0.0, 1.0, 0.0], dtype=np.float32))
        v2 = _unit(np.array([0.0, 0.0, 1.0], dtype=np.float32))
        v3 = _unit(np.array([0.5, 0.5, 0.0], dtype=np.float32))
        query = _unit(np.array([0.0, 1.0, 0.0], dtype=np.float32))
        embedder = _make_embedder(slides, context_words=3, window_vecs=[v0, v1, v2, v3], query_vec=query)
        _, _, margin = embedder.find_slide_with_margin("query")
        assert margin > 0.0


# ---------------------------------------------------------------------------
# avg_words_per_slide
# ---------------------------------------------------------------------------

class TestAvgWordsPerSlide:
    def test_returns_default_before_build(self):
        embedder = WordWindowEmbedder()
        assert embedder.avg_words_per_slide == 3

    def test_computes_from_slide_texts(self):
        # slide 0: 4 words, slide 1: 2 words → avg = 3
        slides = _slides_from("one two three four", "five six")
        mock_model = MagicMock()
        mock_model.encode.return_value = np.ones((4, 4), dtype=np.float32)
        embedder = WordWindowEmbedder()
        with patch("sentence_transformers.SentenceTransformer", return_value=mock_model):
            embedder.load()
        embedder._model = mock_model
        embedder.build(slides, context_words=2)
        assert embedder.avg_words_per_slide == 3

    def test_repeated_slide_counts_each_occurrence(self):
        # chorus appears twice → 4 text entries, avg reflects all occurrences
        slides = [
            (0, "verse one text"),      # 3 words
            (1, "chorus words here"),   # 3 words
            (2, "verse two text"),      # 3 words
            (3, "chorus words here"),   # 3 words (repeated)
        ]
        mock_model = MagicMock()
        mock_model.encode.return_value = np.ones((10, 4), dtype=np.float32)
        embedder = WordWindowEmbedder()
        with patch("sentence_transformers.SentenceTransformer", return_value=mock_model):
            embedder.load()
        embedder._model = mock_model
        embedder.build(slides, context_words=2)
        assert embedder.avg_words_per_slide == 3
