import logging
import sys
import time
from collections import deque

from propresenter_client.main import ProPresenterController

from ..audio_pipeline import COMMAND_COOLDOWN
from ..command_parser import Command, CommandParser, CommandType
from ..slide_follower import SlideFollower

logger = logging.getLogger(__name__)


class FollowHandler:
    """
    Auto-advances on trigger words and accepts all explicit commands.

    After each auto-advance a cooldown prevents the overlapping next
    poll window from immediately re-triggering the same advance.
    """

    def __init__(
        self,
        pro_controller: ProPresenterController,
        command_parser: CommandParser,
        slide_follower: SlideFollower,
        verbose: bool = False,
    ):
        self.pro_controller = pro_controller
        self.command_parser = command_parser
        self.slide_follower = slide_follower
        self.verbose = verbose
        self._last_advance: float = 0.0

    def on_startup(self) -> None:
        ok, reason = self.slide_follower.validate()
        if not ok:
            print(f"Error: Cannot start follow mode — {reason}")
            sys.exit(1)
        self.slide_follower.refresh()

    def startup_description(self) -> str:
        return (
            f"Follow mode active. Listening for: {self.slide_follower.trigger_words}\n"
            "Explicit commands ('next slide', 'previous slide', 'go to slide N') also work."
        )

    def on_transcription(self, text: str, word_buffer: deque) -> None:
        command = self.command_parser.parse(text)
        if command.type != CommandType.UNKNOWN:
            if time.monotonic() - self._last_advance < COMMAND_COOLDOWN:
                return
            self._execute(command)
            self._last_advance = time.monotonic()
            if self.slide_follower.has_triggers and self.verbose:
                print(f"  trigger words: {self.slide_follower.trigger_words}")
            return

        if time.monotonic() - self._last_advance < COMMAND_COOLDOWN:
            return

        if not self.slide_follower.has_triggers:
            self.slide_follower.refresh()

        while self.slide_follower.has_triggers and self.slide_follower.matches(text):
            triggered_on = self.slide_follower.trigger_words
            ok = self.pro_controller.next_slide()
            if not ok:
                print("✗ Failed: next slide (follow)")
                break
            print(f"→ Next slide (follow: {triggered_on})")
            self._last_advance = time.monotonic()
            if not self.slide_follower.refresh_after_advance():
                break
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
