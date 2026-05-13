"""
Hybrid slide matching: sentence-transformer dense embeddings + optional BM25.

Dense scoring uses cosine similarity over sentence-transformers embeddings
(PyTorch-based, trained for semantic similarity).  BM25 can be blended in via
bm25_weight > 0 if desired, but defaults to 0.0 (pure dense).

    score = bm25_weight × bm25_normalised + (1 − bm25_weight) × cosine_sim

Default model: all-MiniLM-L6-v2 (~80 MB, cached after first run via
sentence-transformers / HuggingFace Hub).
"""

import logging
import re
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_BM25_WEIGHT = 0.0


class SlideEmbedder:
    """
    Builds and queries a dense (+ optional BM25) index over slide texts.

    Usage::

        embedder = SlideEmbedder()
        embedder.build(["I pledge allegiance to the flag",
                        "Of the United States of America"])
        slide_idx, confidence = embedder.find_slide("allegiance flag")
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        bm25_weight: float = DEFAULT_BM25_WEIGHT,
    ):
        if not 0.0 <= bm25_weight <= 1.0:
            raise ValueError(f"bm25_weight must be in [0, 1], got {bm25_weight}")
        self._model_name = model_name
        self._bm25_weight = bm25_weight
        self._model = None
        self._embeddings: Optional[np.ndarray] = None  # shape (n_slides, dim)
        self._bm25 = None
        self._slide_indices: list[int] = []  # maps position → ProPresenter 0-based index
        self._slide_count: int = 0

    def load(self) -> None:
        """Download (first run) and initialise the sentence-transformers model."""
        from sentence_transformers import SentenceTransformer
        logger.info("Loading sentence-transformers model '%s'…", self._model_name)
        self._model = SentenceTransformer(self._model_name)
        logger.info("Embedding model ready.")

    def build(
        self,
        slide_texts: list[str],
        slide_indices: list[int] | None = None,
    ) -> None:
        """
        Create and cache dense embeddings (and optionally a BM25 index) for every slide.

        Args:
            slide_texts:   Plain text for each slide to index.
            slide_indices: Corresponding 0-based ProPresenter slide indices.
                           Defaults to [0, 1, 2, …] when all slides have text.
        """
        if not slide_texts:
            raise ValueError("slide_texts must not be empty")

        self._slide_indices = (
            slide_indices if slide_indices is not None else list(range(len(slide_texts)))
        )
        if len(self._slide_indices) != len(slide_texts):
            raise ValueError("slide_indices length must match slide_texts length")

        # BM25 index — only built when blending is requested
        if self._bm25_weight > 0.0:
            from rank_bm25 import BM25Okapi
            tokenized = [_tokenize(t) for t in slide_texts]
            self._bm25_index = BM25Okapi(tokenized)
            logger.info("BM25 index built for %d slides.", len(slide_texts))
        else:
            self._bm25_index = None

        # Dense embeddings
        if self._bm25_weight < 1.0:
            if self._model is None:
                self.load()
            logger.info("Building dense embeddings for %d slides…", len(slide_texts))
            self._embeddings = np.array(
                self._model.encode(slide_texts, show_progress_bar=False),
                dtype=np.float32,
            )
            logger.info("Slide embeddings ready (%d × %d).", *self._embeddings.shape)
        else:
            self._embeddings = None

        # Sentinel used by find_slide to detect un-built state
        self._bm25 = self._bm25_index if self._bm25_weight > 0.0 else True
        self._slide_count = len(slide_texts)

    def find_slide(self, text: str) -> tuple[int, float]:
        """
        Return (slide_index, hybrid_score) for the best-matching slide.

        slide_index is the 0-based ProPresenter index (not the position in the
        embedding array).  Returns (-1, 0.0) when no index is built or the
        query is empty.  hybrid_score is in [0, 1].
        """
        if self._bm25 is None or not text.strip():
            return -1, 0.0
        scores = self._hybrid_scores(_tokenize(text), text)
        best_pos = int(np.argmax(scores))
        return self._slide_indices[best_pos], float(scores[best_pos])

    def find_slide_with_margin(self, text: str) -> tuple[int, float, float]:
        """
        Like find_slide but also returns the margin (best − second-best score).

        A large margin means the winner is unambiguous even when the absolute
        score is modest.  Returns (-1, 0.0, 0.0) on empty query or missing index.
        """
        if self._bm25 is None or not text.strip():
            return -1, 0.0, 0.0
        scores = self._hybrid_scores(_tokenize(text), text)
        best_pos = int(np.argmax(scores))
        best_score = float(scores[best_pos])
        if len(scores) > 1:
            second = float(np.partition(scores, -2)[-2])
            margin = best_score - second
        else:
            margin = best_score
        return self._slide_indices[best_pos], best_score, margin

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _hybrid_scores(self, tokens: list[str], text: str) -> np.ndarray:
        """Compute normalised hybrid BM25 + dense scores for all slides."""
        # BM25 scores (unbounded ≥ 0) — normalise to [0, 1]
        if self._bm25_weight > 0.0 and self._bm25_index is not None:
            bm25_raw = np.array(self._bm25_index.get_scores(tokens), dtype=np.float64)
            bm25_max = bm25_raw.max()
            bm25_scores = bm25_raw / bm25_max if bm25_max > 0 else bm25_raw
        else:
            bm25_scores = np.zeros(self._slide_count)

        # Dense cosine scores ([−1, 1]) — clip to [0, 1]
        if self._bm25_weight < 1.0 and self._embeddings is not None:
            query = np.array(
                self._model.encode([text], show_progress_bar=False)[0],
                dtype=np.float32,
            )
            query_norm = float(np.linalg.norm(query))
            if query_norm > 0:
                slide_norms = np.linalg.norm(self._embeddings, axis=1)
                cosine = np.where(
                    slide_norms > 0,
                    (self._embeddings @ query) / (slide_norms * query_norm),
                    0.0,
                )
                dense_scores = np.clip(cosine, 0.0, 1.0)
            else:
                dense_scores = np.zeros(self._slide_count)
        else:
            dense_scores = np.zeros(self._slide_count)

        return (
            self._bm25_weight * bm25_scores
            + (1.0 - self._bm25_weight) * dense_scores
        )

    @property
    def slide_count(self) -> int:
        return self._slide_count


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation — used by BM25 build and query paths."""
    text = re.sub(r"\\[a-zA-Z]+\d*\s?", " ", text)   # RTF control words
    text = re.sub(r"<[^>]+>", " ", text)               # HTML tags
    text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return [w for w in text.split() if w]
