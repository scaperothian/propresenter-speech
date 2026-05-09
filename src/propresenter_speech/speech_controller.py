"""
Orchestrates the full voice → slide control pipeline.

Flow:
  AudioCapture (mic thread) ──► segment queue ──► SpeechController.run()
       │                                                    │
       │                                         Transcriber (Whisper)
       │                                                    │
       └───────────────────────────────────────► CommandParser
                                                            │
                                      (mode-dependent dispatch)
                                         ┌──────────────────┴──────────────────┐
                                presentation mode                          follow mode
                               explicit commands only           trigger words + explicit commands
                                                                            │
                                                                      SlideFollower
                                                            │
                                               ProPresenterController
"""

import logging
import queue
from typing import Optional

import numpy as np

from propresenter_slides.main import ProPresenterController

from .audio_capture import AudioCapture
from .command_parser import Command, CommandParser, CommandType
from .modes import Mode
from .slide_follower import SlideFollower
from .transcriber import Transcriber

logger = logging.getLogger(__name__)

_SEGMENT_QUEUE_SIZE = 2


class SpeechController:
    """
    Wires together audio capture, transcription, command parsing, and slide control.

    In ``presentation`` mode (default) only explicit voice commands are acted on.
    In ``follow`` mode the controller also watches for the last word(s) of the
    active slide and auto-advances when they are heard, while still accepting all
    explicit commands.

    Typical usage::

        controller = SpeechController(
            transcriber=Transcriber("base"),
            command_parser=CommandParser(),
            pro_controller=ProPresenterController(),
            audio_capture=AudioCapture(),
            mode=Mode.FOLLOW,
            slide_follower=SlideFollower(pro_controller),
        )
        controller.run()  # blocks until Ctrl-C
    """

    def __init__(
        self,
        transcriber: Transcriber,
        command_parser: CommandParser,
        pro_controller: ProPresenterController,
        audio_capture: AudioCapture,
        mode: Mode = Mode.PRESENTATION,
        slide_follower: Optional[SlideFollower] = None,
        verbose: bool = False,
    ):
        self.transcriber = transcriber
        self.command_parser = command_parser
        self.pro_controller = pro_controller
        self.audio_capture = audio_capture
        self.mode = mode
        self.slide_follower = slide_follower
        self.verbose = verbose
        self._segment_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=_SEGMENT_QUEUE_SIZE)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Load Whisper, start the microphone, and process commands until Ctrl-C."""
        print("Loading Whisper model — this may take a moment on first run…")
        self.transcriber.load()
        print("Whisper ready.")

        if self.mode == Mode.FOLLOW:
            self._init_follow_mode()

        self.audio_capture.start(self._enqueue_segment)
        self._print_listening_banner()
        print("Press Ctrl-C to stop.\n")

        try:
            while True:
                try:
                    segment = self._segment_queue.get(timeout=0.5)
                    self._handle_segment(segment)
                except queue.Empty:
                    continue
        except KeyboardInterrupt:
            print("\nStopping…")
        finally:
            self.audio_capture.stop()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _init_follow_mode(self) -> None:
        if self.slide_follower is None:
            logger.warning("Follow mode requested but no SlideFollower provided; falling back to presentation mode.")
            self.mode = Mode.PRESENTATION
            return
        ok = self.slide_follower.refresh()
        if ok:
            print(f"Follow mode active. Listening for: {self.slide_follower.trigger_words}")
        else:
            print("Follow mode active. (Could not read slide text — will retry on each transcription.)")

    def _print_listening_banner(self) -> None:
        if self.mode == Mode.FOLLOW:
            print("Listening in follow mode. Slide advances automatically on trigger words.")
            print("Explicit commands ('next slide', 'previous slide', 'go to slide N') also work.")
        else:
            print("Listening for voice commands. Say 'next slide', 'previous slide', or 'go to slide N'.")  # noqa: E501

    def _enqueue_segment(self, audio: np.ndarray) -> None:
        try:
            self._segment_queue.put_nowait(audio)
        except queue.Full:
            logger.debug("Segment queue full — dropping oldest segment.")
            try:
                self._segment_queue.get_nowait()
            except queue.Empty:
                pass
            self._segment_queue.put_nowait(audio)

    def _handle_segment(self, audio: np.ndarray) -> None:
        text = self.transcriber.transcribe(audio)
        if not text.strip():
            return

        if self.verbose:
            print(f"  heard: {text!r}")

        command = self.command_parser.parse(text)

        if command.type != CommandType.UNKNOWN:
            self._execute(command, text)
            if self.mode == Mode.FOLLOW and self.slide_follower:
                self.slide_follower.refresh()
                if self.slide_follower.has_triggers and self.verbose:
                    print(f"  trigger words: {self.slide_follower.trigger_words}")
        elif self.mode == Mode.FOLLOW:
            self._handle_follow(text)

    def _handle_follow(self, text: str) -> None:
        if self.slide_follower is None:
            return

        if not self.slide_follower.has_triggers:
            self.slide_follower.refresh()

        if self.slide_follower.matches(text):
            ok = self.pro_controller.next_slide()
            if ok:
                print(f"→ Next slide (follow: {self.slide_follower.trigger_words})")
                self.slide_follower.refresh()
                if self.slide_follower.has_triggers and self.verbose:
                    print(f"  trigger words: {self.slide_follower.trigger_words}")
            else:
                print("✗ Failed: next slide (follow)")

    def _execute(self, command: Command, raw_text: str) -> None:
        if command.type == CommandType.NEXT_SLIDE:
            ok = self.pro_controller.next_slide()
            print("→ Next slide" if ok else "✗ Failed: next slide")

        elif command.type == CommandType.PREVIOUS_SLIDE:
            ok = self.pro_controller.previous_slide()
            print("← Previous slide" if ok else "✗ Failed: previous slide")

        elif command.type == CommandType.GO_TO_SLIDE:
            n = command.slide_number
            ok = self.pro_controller.go_to_slide(n)
            print(f"→ Slide {n}" if ok else f"✗ Failed: go to slide {n}")

        else:
            logger.debug("No command matched for: %r", raw_text)
