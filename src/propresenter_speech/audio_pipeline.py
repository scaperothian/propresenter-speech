"""
Shared audio pipeline for all speech modes.

Mic mode:
  sounddevice InputStream → ring buffer (last window_seconds of PCM)
       │
       └── timer thread (every poll_interval s, when Whisper is free)
                │   snapshot
                ▼
            Transcriber → ModeHandler.on_transcription(text, word_buffer)

File mode:
  audio file → sliding window (window_seconds wide, advances by poll_interval)
                ▼
            Transcriber → ModeHandler.on_transcription(text, word_buffer)
"""

import collections
import logging
import threading
import time
from typing import TYPE_CHECKING, Optional

import numpy as np
import sounddevice as sd

from .slide_follower import extract_words
from .transcriber import Transcriber

if TYPE_CHECKING:
    from .handlers.base import ModeHandler

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
DEFAULT_WINDOW_SECONDS = 2.0
DEFAULT_POLL_INTERVAL = 0.2
COMMAND_COOLDOWN = 1.8


class AudioPipeline:
    """
    Drives the Whisper transcription loop and dispatches results to a ModeHandler.

    All modes share this pipeline; mode-specific logic lives exclusively in the
    handler.  Pass ``audio_file`` to process a WAV/FLAC/OGG file instead of the
    microphone.
    """

    def __init__(
        self,
        transcriber: Transcriber,
        handler: "ModeHandler",
        device: Optional[int] = None,
        audio_file: Optional[str] = None,
        window_seconds: float = DEFAULT_WINDOW_SECONDS,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        verbose: bool = False,
        playback: bool = False,
        output_device: Optional[int] = None,
    ):
        self.transcriber = transcriber
        self.handler = handler
        self.device = device
        self.audio_file = audio_file
        self.poll_interval = poll_interval
        self.verbose = verbose
        self.playback = playback
        self.output_device = output_device

        self._window_frames = int(window_seconds * SAMPLE_RATE)
        self._ring: collections.deque = collections.deque(maxlen=self._window_frames)
        self._word_buffer: collections.deque[str] = collections.deque(maxlen=200)
        self._whisper_busy = False
        self._running = False

    def run(self) -> None:
        """Call handler startup, start audio, process until Ctrl-C or 'q' + Enter."""
        self.handler.on_startup()
        print(self.handler.startup_description())
        self._running = True

        if self.audio_file:
            self._run_file()
        else:
            self._run_mic()

    # ------------------------------------------------------------------
    # Mic mode
    # ------------------------------------------------------------------

    def _run_mic(self) -> None:
        device_info = sd.query_devices(self.device if self.device is not None else sd.default.device[0])
        print(f"Audio input: [{device_info['index']}] {device_info['name']}")

        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=self.device,
            blocksize=int(SAMPLE_RATE * self.poll_interval),
            callback=self._sd_callback,
        )
        stream.start()

        threading.Thread(
            target=self._poll_loop, daemon=True, name="pipeline-poll"
        ).start()

        print("Press 'q' + Enter or Ctrl-C to stop.\n")
        self._wait_for_stop()

        stream.stop()
        stream.close()

    def _sd_callback(self, indata, _frames, _time_info, status) -> None:
        if status:
            logger.debug("sounddevice status: %s", status)
        self._ring.extend(indata[:, 0].tolist())

    def _poll_loop(self) -> None:
        while self._running:
            time.sleep(self.poll_interval)
            if self._whisper_busy or len(self._ring) < SAMPLE_RATE * 0.5:
                continue
            audio = np.array(list(self._ring), dtype=np.float32)
            # Mic mode has no file position, so audio_time is 0.0.
            # Handlers that need positional information (AccuracyHandler) only
            # run against files, where _run_file() passes a frame-derived T_snap.
            threading.Thread(
                target=self._process,
                args=(audio, 0.0),
                daemon=True,
                name="pipeline-whisper",
            ).start()

    # ------------------------------------------------------------------
    # File mode
    # ------------------------------------------------------------------

    def _run_file(self) -> None:
        import soundfile as sf
        from tqdm import tqdm

        try:
            audio_orig, sample_rate = sf.read(self.audio_file, dtype="float32", always_2d=False)
        except Exception as exc:
            logger.error("Failed to load audio file '%s': %s", self.audio_file, exc)
            return

        if audio_orig.ndim > 1:
            audio_orig = audio_orig.mean(axis=1)

        duration = len(audio_orig) / sample_rate
        print(f"Processing file: {self.audio_file}  ({duration:.1f}s)")

        if self.playback:
            output_info = sd.query_devices(
                self.output_device if self.output_device is not None else sd.default.device[1]
            )
            print(f"Audio output: [{output_info['index']}] {output_info['name']}")
            sd.play(audio_orig, samplerate=sample_rate, device=self.output_device)

        audio = audio_orig
        if sample_rate != SAMPLE_RATE:
            logger.info("Resampling from %d Hz to %d Hz", sample_rate, SAMPLE_RATE)
            audio = _resample(audio_orig, sample_rate, SAMPLE_RATE)

        poll_frames = int(self.poll_interval * SAMPLE_RATE)
        min_frames = int(SAMPLE_RATE * 0.5)
        total_frames = len(audio)

        # Sliding window: advance by poll_frames each step, always transcribe
        # the trailing window_frames.  Mirrors mic mode's ring buffer exactly —
        # poll_interval controls step size, window_seconds controls context depth.
        with tqdm(total=int(duration), unit="s", desc="Processing", ncols=70) as progress:
            end = poll_frames
            while self._running:
                end = min(end, total_frames)
                chunk = audio[max(0, end - self._window_frames) : end]
                if len(chunk) >= min_frames:
                    self._process(chunk, end / SAMPLE_RATE)
                progress.n = int(end / SAMPLE_RATE)
                progress.refresh()
                if end >= total_frames:
                    break
                end += poll_frames

        if self.playback:
            sd.stop()

    # ------------------------------------------------------------------
    # Shared transcription
    # ------------------------------------------------------------------

    def _process(self, audio: np.ndarray, audio_time: float = 0.0) -> None:
        # audio_time is T_snap for file mode (end of transcribed window, frames-derived),
        # or 0.0 for mic mode.  Forwarded verbatim to on_transcription() so handlers
        # can use it for ground-truth lookup without any latency correction.
        self._whisper_busy = True
        try:
            text = self.transcriber.transcribe(audio)
            if not text.strip():
                return
            if self.verbose:
                print(f"  heard: {text!r}")
            self._word_buffer.extend(extract_words(text))
            self.handler.on_transcription(text, self._word_buffer, audio_time)
        finally:
            self._whisper_busy = False

    # ------------------------------------------------------------------
    # Stop / keyboard
    # ------------------------------------------------------------------

    def _wait_for_stop(self) -> None:
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

    @staticmethod
    def _keyboard_listener(stop: threading.Event) -> None:
        while not stop.is_set():
            try:
                if input().strip().lower() == "q":
                    stop.set()
            except EOFError:
                break


# ------------------------------------------------------------------
# Utilities (previously in audio_capture.py)
# ------------------------------------------------------------------

def list_input_devices() -> list[dict]:
    """Return a list of available input audio devices."""
    devices = sd.query_devices()
    return [
        {"index": i, "name": d["name"], "channels": d["max_input_channels"]}
        for i, d in enumerate(devices)
        if d["max_input_channels"] > 0
    ]


def list_output_devices() -> list[dict]:
    """Return a list of available output audio devices."""
    devices = sd.query_devices()
    return [
        {"index": i, "name": d["name"], "channels": d["max_output_channels"]}
        for i, d in enumerate(devices)
        if d["max_output_channels"] > 0
    ]


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Linear-interpolation resample — adequate quality for 16 kHz voice."""
    n_samples = int(len(audio) * target_sr / orig_sr)
    return np.interp(
        np.linspace(0, len(audio) - 1, n_samples),
        np.arange(len(audio)),
        audio,
    ).astype(np.float32)
