"""
Follow-enhanced mode controller.

Differences from follow mode:
  - Audio is captured in a continuous ring buffer rather than waiting for
    silence; Whisper is invoked on a rolling window every POLL_INTERVAL seconds.
  - Slide matching is semantic (cosine similarity over fastembed embeddings)
    rather than exact trigger-word matching.
  - The matched slide is triggered directly via go_to_slide(), so the
    presentation can jump forward or backward as the speaker moves around.

Pipeline::

    sounddevice stream → ring buffer (last WINDOW_SECONDS of PCM)
         │
         └── timer thread (every POLL_INTERVAL s, if Whisper is free)
                  │   snapshot of ring buffer
                  ▼
              Transcriber (Whisper)
                  │   text
                  ▼
              word deque (rolling last N words across segments)
                  │   n-gram string
                  ▼
              SlideEmbedder.find_slide()
                  │   (slide_index, confidence)
                  ▼
              ProPresenterController.go_to_slide()   ← only on new match
"""

import collections
import logging
import threading
import time
from typing import Optional

import numpy as np
import sounddevice as sd

from propresenter_slides.main import ProPresenterController

from .audio_capture import SAMPLE_RATE
from .slide_embedder import SlideEmbedder
from .slide_follower import extract_words
from .transcriber import Transcriber

logger = logging.getLogger(__name__)

DEFAULT_WINDOW_SECONDS = 2.0
DEFAULT_POLL_INTERVAL = 0.2
DEFAULT_CONTEXT_WORDS = 3
DEFAULT_SIMILARITY_THRESHOLD = 0.4
DEFAULT_MIN_MARGIN = 0.15


class FollowEnhancedController:
    """
    Continuously matches spoken audio against slide embeddings and cues the
    best-matching slide in ProPresenter.

    Args:
        transcriber:          Loaded Whisper transcriber.
        pro_controller:       ProPresenter HTTP client.
        slide_embedder:       Built SlideEmbedder (embeddings already computed).
        device:               sounddevice input device index (None = default).
        window_seconds:       Length of the rolling audio window fed to Whisper.
        poll_interval:        Seconds between Whisper inference calls.
        context_words:        Number of recent words to form the query n-gram.
        similarity_threshold: Minimum hybrid score to trigger a cue.
        min_margin:           Minimum gap between best and second-best score to
                              trigger a cue even when below similarity_threshold.
        verbose:              Print transcriptions and match scores.
    """

    def __init__(
        self,
        transcriber: Transcriber,
        pro_controller: ProPresenterController,
        slide_embedder: SlideEmbedder,
        device: Optional[int] = None,
        window_seconds: float = DEFAULT_WINDOW_SECONDS,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        context_words: int = DEFAULT_CONTEXT_WORDS,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        min_margin: float = DEFAULT_MIN_MARGIN,
        verbose: bool = False,
    ):
        self.transcriber = transcriber
        self.pro_controller = pro_controller
        self.slide_embedder = slide_embedder
        self.device = device
        self.poll_interval = poll_interval
        self.context_words = context_words
        self.similarity_threshold = similarity_threshold
        self.min_margin = min_margin
        self.verbose = verbose

        self._window_frames = int(window_seconds * SAMPLE_RATE)
        self._ring: collections.deque[float] = collections.deque(
            maxlen=self._window_frames
        )
        self._word_buffer: collections.deque[str] = collections.deque(maxlen=200)
        self._whisper_busy = False
        self._current_slide_idx: Optional[int] = None
        self._running = False
        self._stream: Optional[sd.InputStream] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Open the microphone and run until Ctrl-C or 'q' + Enter."""
        self._running = True

        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=self.device,
            blocksize=int(SAMPLE_RATE * self.poll_interval),
            callback=self._sd_callback,
        )
        self._stream.start()

        poll_thread = threading.Thread(
            target=self._poll_loop, daemon=True, name="fe-poll"
        )
        poll_thread.start()

        print(
            f"Follow-enhanced mode active — semantic matching, "
            f"window={self._window_frames / SAMPLE_RATE:.1f}s, "
            f"context={self.context_words} words, "
            f"threshold={self.similarity_threshold:.2f}, "
            f"min_margin={self.min_margin:.2f}"
        )
        print("Press 'q' + Enter or Ctrl-C to stop.\n")

        stop = threading.Event()
        threading.Thread(
            target=self._keyboard_listener, args=(stop,), daemon=True
        ).start()

        try:
            while not stop.is_set():
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            print("\nStopping…")
            self._running = False
            self._stream.stop()
            self._stream.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _sd_callback(
        self,
        indata: np.ndarray,
        _frames: int,
        _time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        if status:
            logger.debug("sounddevice status: %s", status)
        self._ring.extend(indata[:, 0].tolist())

    def _poll_loop(self) -> None:
        while self._running:
            time.sleep(self.poll_interval)
            if self._whisper_busy or len(self._ring) < SAMPLE_RATE * 0.5:
                continue
            audio = np.array(list(self._ring), dtype=np.float32)
            threading.Thread(
                target=self._transcribe_and_match,
                args=(audio,),
                daemon=True,
                name="fe-whisper",
            ).start()

    def _transcribe_and_match(self, audio: np.ndarray) -> None:
        self._whisper_busy = True
        try:
            text = self.transcriber.transcribe(audio)
            if not text.strip():
                return

            if self.verbose:
                print(f"  heard: {text!r}")

            words = extract_words(text)
            self._word_buffer.extend(words)

            query_words = list(self._word_buffer)[-self.context_words:]
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
        finally:
            self._whisper_busy = False

    @staticmethod
    def _keyboard_listener(stop: threading.Event) -> None:
        while not stop.is_set():
            try:
                if input().strip().lower() == "q":
                    stop.set()
            except EOFError:
                break
