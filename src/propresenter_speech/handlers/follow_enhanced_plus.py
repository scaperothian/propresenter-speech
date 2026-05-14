import logging
import sys
import time
from collections import deque
from typing import Optional

from propresenter_client.main import ProPresenterController

from ..audio_pipeline import COMMAND_COOLDOWN
from ..command_parser import Command, CommandParser, CommandType
from ..slide_embedder import SlideEmbedder
from ..slide_follower import SlideFollower
from .follow_enhanced import (
    DEFAULT_CONTEXT_WORDS,
    DEFAULT_MIN_MARGIN,
    DEFAULT_SIMILARITY_THRESHOLD,
)

logger = logging.getLogger(__name__)


class FollowEnhancedPlusHandler:
    """
    Combines follow and follow-enhanced modes.

    Each audio window runs the dense embedding search against all slides.
    If the best-matching slide differs from the current one and meets the
    confidence/margin threshold, the presentation jumps there directly.
    When the embedding match is ambiguous the handler falls back to
    trigger-word matching for a sequential advance — reducing false jumps
    while still catching intentional navigation.

    Explicit voice commands ('next slide', 'previous slide', 'go to slide N')
    are always honoured and update trigger words without hitting the
    ProPresenter API for the new slide index.
    """

    def __init__(
        self,
        pro_controller: ProPresenterController,
        command_parser: CommandParser,
        slide_follower: SlideFollower,
        slide_embedder: SlideEmbedder,
        context_words: int = DEFAULT_CONTEXT_WORDS,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        min_margin: float = DEFAULT_MIN_MARGIN,
        verbose: bool = False,
    ):
        self.pro_controller = pro_controller
        self.command_parser = command_parser
        self.slide_follower = slide_follower
        self.slide_embedder = slide_embedder
        self.context_words = context_words
        self.similarity_threshold = similarity_threshold
        self.min_margin = min_margin
        self.verbose = verbose
        self._last_advance: float = 0.0
        self._current_slide_idx: Optional[int] = None

    def on_startup(self) -> None:
        ok, reason = self.slide_follower.validate()
        if not ok:
            print(f"Error: Cannot start follow-enhanced-plus mode — {reason}")
            sys.exit(1)
        self.slide_follower.refresh()
        self._current_slide_idx = self.slide_follower.current_slide_index

    def startup_description(self) -> str:
        return (
            f"Follow-enhanced-plus mode active — "
            f"embedding search ({self.slide_embedder.slide_count} slides) + trigger words.\n"
            f"Listening for trigger: {self.slide_follower.trigger_words}\n"
            "Explicit commands ('next slide', 'previous slide', 'go to slide N') also work."
        )

    def on_transcription(self, text: str, word_buffer: deque) -> None:
        command = self.command_parser.parse(text)
        if command.type != CommandType.UNKNOWN:
            if time.monotonic() - self._last_advance < COMMAND_COOLDOWN:
                return
            self._execute(command)
            self._last_advance = time.monotonic()
            self._current_slide_idx = self.slide_follower.current_slide_index
            if self.slide_follower.has_triggers and self.verbose:
                print(f"  trigger words: {self.slide_follower.trigger_words}")
            return

        if time.monotonic() - self._last_advance < COMMAND_COOLDOWN:
            return

        query_words = list(word_buffer)[-self.context_words:]
        if len(query_words) >= 2:
            query = " ".join(query_words)
            slide_idx, confidence, margin = self.slide_embedder.find_slide_with_margin(query)

            if self.verbose:
                print(
                    f"  query: {query!r}  →  slide {slide_idx + 1}"
                    f"  ({confidence:.3f}, margin {margin:.3f})"
                )

            if (
                slide_idx >= 0
                and slide_idx != self._current_slide_idx
                and (confidence >= self.similarity_threshold or margin >= self.min_margin)
            ):
                ok = self.pro_controller.go_to_slide(slide_idx + 1)
                if ok:
                    self._current_slide_idx = slide_idx
                    self.slide_follower.refresh_to_slide(slide_idx)
                    self._last_advance = time.monotonic()
                    print(
                        f"→ Slide {slide_idx + 1} "
                        f"(embedding: {confidence:.2f}, margin {margin:.2f}, query: {query!r})"
                    )
                else:
                    print(f"✗ Failed: go to slide {slide_idx + 1}")
                return

        self._check_trigger(text)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_trigger(self, text: str) -> None:
        if not self.slide_follower.has_triggers:
            self.slide_follower.refresh()

        while self.slide_follower.has_triggers and self.slide_follower.matches(text):
            triggered_on = self.slide_follower.trigger_words
            ok = self.pro_controller.next_slide()
            if not ok:
                print("✗ Failed: next slide (trigger)")
                break
            print(f"→ Next slide (trigger: {triggered_on})")
            self._last_advance = time.monotonic()
            if not self.slide_follower.refresh_after_advance():
                break
            self._current_slide_idx = self.slide_follower.current_slide_index
            if self.slide_follower.has_triggers and self.verbose:
                print(f"  trigger words: {self.slide_follower.trigger_words}")

    def _execute(self, command: Command) -> None:
        if command.type == CommandType.NEXT_SLIDE:
            ok = self.pro_controller.next_slide()
            print("→ Next slide" if ok else "✗ Failed: next slide")
            if ok:
                self.slide_follower.refresh_after_advance(delta=1)

        elif command.type == CommandType.PREVIOUS_SLIDE:
            ok = self.pro_controller.previous_slide()
            print("← Previous slide" if ok else "✗ Failed: previous slide")
            if ok:
                self.slide_follower.refresh_after_advance(delta=-1)

        elif command.type == CommandType.GO_TO_SLIDE:
            n = command.slide_number
            ok = self.pro_controller.go_to_slide(n)
            print(f"→ Slide {n}" if ok else f"✗ Failed: go to slide {n}")
            if ok:
                self.slide_follower.refresh_to_slide(n - 1)
