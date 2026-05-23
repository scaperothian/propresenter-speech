import logging
from collections import deque
from typing import Optional

DEFAULT_CONTEXT_WORDS = 3
DEFAULT_SIMILARITY_THRESHOLD = 0.4
DEFAULT_MIN_MARGIN = 0.15

from propresenter_client.main import ProPresenterController

from ..slide_embedder import SlideEmbedder

logger = logging.getLogger(__name__)


class FollowEnhancedHandler:
    """
    Cues whichever slide best matches the most recent spoken words via
    sentence-transformer cosine similarity.  Can jump to any slide freely.
    Does not parse explicit voice commands.
    """

    def __init__(
        self,
        pro_controller: ProPresenterController,
        slide_embedder: SlideEmbedder,
        context_words: int = 3,
        similarity_threshold: float = 0.4,
        min_margin: float = 0.15,
        verbose: bool = False,
    ):
        self.pro_controller = pro_controller
        self.slide_embedder = slide_embedder
        self.context_words = context_words
        self.similarity_threshold = similarity_threshold
        self.min_margin = min_margin
        self.verbose = verbose
        self._current_slide_idx: Optional[int] = None

    def on_startup(self) -> None:
        pass

    def startup_description(self) -> str:
        return (
            f"Follow-enhanced mode active — semantic matching, "
            f"context={self.context_words} words, "
            f"threshold={self.similarity_threshold:.2f}, "
            f"min_margin={self.min_margin:.2f}"
        )

    def on_transcription(self, _text: str, word_buffer: deque) -> None:
        query_words = list(word_buffer)[-self.context_words :]
        if len(query_words) < 2:
            return

        query = " ".join(query_words)
        slide_idx, confidence, margin = self.slide_embedder.find_slide_with_margin(query)

        if self.verbose:
            print(
                f"  query: {query!r}  →  slide {slide_idx + 1}"
                f"  ({confidence:.3f}, margin {margin:.3f})"
            )

        if slide_idx < 0:
            return
        if confidence < self.similarity_threshold and margin < self.min_margin:
            return
        if slide_idx == self._current_slide_idx:
            return

        ok = self.pro_controller.go_to_slide(slide_idx + 1)
        if ok:
            self._current_slide_idx = slide_idx
            print(f"→ Slide {slide_idx + 1} (confidence: {confidence:.2f}, query: {query!r})")
        else:
            print(f"✗ Failed: go to slide {slide_idx + 1}")
