"""
Dense slide matching using sentence-transformer embeddings.

Scoring uses cosine similarity over all-MiniLM-L6-v2 embeddings
(cached after first run via HuggingFace Hub, ~80 MB).
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "all-MiniLM-L6-v2"


class SlideEmbedder:
    """
    Builds and queries a dense cosine-similarity index over slide texts.

    Usage::

        embedder = SlideEmbedder()
        embedder.build(["I pledge allegiance to the flag",
                        "Of the United States of America"])
        slide_idx, confidence = embedder.find_slide("allegiance flag")
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self._model_name = model_name
        self._model = None
        self._embeddings: Optional[np.ndarray] = None  # shape (n_slides, dim)
        self._slide_indices: list[int] = []
        self._slide_texts: list[str] = []
        self._slide_count: int = 0
        self._built = False

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
        Create and cache dense embeddings for every slide.

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
        self._slide_texts = slide_texts

        if self._model is None:
            self.load()
        logger.info("Building dense embeddings for %d slides…", len(slide_texts))
        self._embeddings = np.array(
            self._model.encode(slide_texts, show_progress_bar=False),
            dtype=np.float32,
        )
        logger.info("Slide embeddings ready (%d × %d).", *self._embeddings.shape)
        self._slide_count = len(slide_texts)
        self._built = True

    def find_slide(self, text: str) -> tuple[int, float]:
        """
        Return (slide_index, cosine_score) for the best-matching slide.

        slide_index is the 0-based ProPresenter index (not the position in the
        embedding array).  Returns (-1, 0.0) when no index is built or the
        query is empty.  cosine_score is clipped to [0, 1].
        """
        if not self._built or not text.strip():
            return -1, 0.0
        raw = self._cosine_scores(text)
        best_pos = int(np.argmax(raw))
        return self._slide_indices[best_pos], float(np.clip(raw[best_pos], 0.0, 1.0))

    def find_slide_with_margin(self, text: str) -> tuple[int, float, float]:
        """
        Like find_slide but also returns the margin (best − second-best raw cosine).

        Margin is computed on unclipped cosine values so that slides scoring
        slightly negative still contribute a meaningful spread.  The returned
        score is clipped to [0, 1]; the margin may be negative (ambiguous match).
        Returns (-1, 0.0, 0.0) on empty query or missing index.
        """
        if not self._built or not text.strip():
            return -1, 0.0, 0.0
        raw = self._cosine_scores(text)
        best_pos = int(np.argmax(raw))
        best_raw = float(raw[best_pos])
        if len(raw) > 1:
            second = float(np.partition(raw, -2)[-2])
            margin = best_raw - second
        else:
            margin = best_raw
        return self._slide_indices[best_pos], float(np.clip(best_raw, 0.0, 1.0)), margin

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _cosine_scores(self, text: str) -> np.ndarray:
        """Return raw (unclipped) cosine similarities in [-1, 1] for all slides."""
        query = np.array(
            self._model.encode([text], show_progress_bar=False)[0],
            dtype=np.float32,
        )
        query_norm = float(np.linalg.norm(query))
        if query_norm == 0:
            return np.zeros(self._slide_count)
        slide_norms = np.linalg.norm(self._embeddings, axis=1)
        return np.where(
            slide_norms > 0,
            (self._embeddings @ query) / (slide_norms * query_norm),
            0.0,
        )

    @property
    def slide_count(self) -> int:
        return self._slide_count

    @property
    def avg_words_per_slide(self) -> int:
        """Average word count per slide, rounded to nearest int (minimum 1)."""
        if not self._built or not self._slide_texts:
            return 3
        total = sum(len(t.split()) for t in self._slide_texts)
        return max(1, round(total / len(self._slide_texts)))
