"""
Orchestrates the full voice → slide control pipeline.

Flow:
  AudioCapture (mic thread) ──► segment queue ──► SpeechController.run()
       │                                                    │
       │                                         Transcriber (Whisper)
       │                                                    │
       └───────────────────────────────────────► CommandParser
                                                            │
                                               ProPresenterController
"""

import logging
import queue

import numpy as np

from propresenter_slides.main import ProPresenterController

from .audio_capture import AudioCapture
from .command_parser import Command, CommandParser, CommandType
from .transcriber import Transcriber

logger = logging.getLogger(__name__)

# Drop incoming segments if the processing queue is already full to maintain
# real-time responsiveness (we prefer the latest utterance over a stale one).
_SEGMENT_QUEUE_SIZE = 2


class SpeechController:
    """
    Wires together audio capture, transcription, command parsing, and slide control.

    Typical usage::

        controller = SpeechController(
            transcriber=Transcriber("base"),
            command_parser=CommandParser(),
            pro_controller=ProPresenterController(),
            audio_capture=AudioCapture(),
        )
        controller.run()  # blocks until Ctrl-C
    """

    def __init__(
        self,
        transcriber: Transcriber,
        command_parser: CommandParser,
        pro_controller: ProPresenterController,
        audio_capture: AudioCapture,
        verbose: bool = False,
    ):
        self.transcriber = transcriber
        self.command_parser = command_parser
        self.pro_controller = pro_controller
        self.audio_capture = audio_capture
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

        self.audio_capture.start(self._enqueue_segment)
        print("Listening for voice commands. Say 'next slide', 'previous slide', or 'go to slide N'.")  # noqa: E501
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
        self._execute(command, text)

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
