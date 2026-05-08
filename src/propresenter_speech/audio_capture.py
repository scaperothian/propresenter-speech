"""
Real-time audio capture from the Mac microphone using sounddevice.

Architecture:
  - sounddevice streams raw PCM chunks into a thread-safe queue via a callback.
  - A separate processing thread reads those chunks, applies energy-based VAD
    (Voice Activity Detection), assembles speech segments, and fires the caller's
    ``on_segment`` callback with a complete numpy array for each utterance.

Tuning guide:
  silence_threshold  — RMS energy (0–1) below which a frame is considered
                        silence.  Increase if background noise triggers false
                        positives; decrease if quiet speech is missed.
  silence_duration   — seconds of continuous silence required to close a
                        speech segment.  Shorter = more responsive;
                        longer = fewer split utterances.
  max_speech_duration — hard cap on segment length before a forced flush,
                        preventing runaway buffering during continuous speech.
"""

import logging
import queue
import threading
from typing import Callable, Optional

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000          # Hz — matches Whisper's expected input rate
CHUNK_SECONDS = 0.1           # seconds per sounddevice callback block
CHUNK_FRAMES = int(SAMPLE_RATE * CHUNK_SECONDS)

DEFAULT_SILENCE_THRESHOLD = 0.01   # RMS in float32 scale
DEFAULT_SILENCE_DURATION = 0.8     # seconds of silence to close a segment
DEFAULT_MIN_SPEECH_DURATION = 0.3  # seconds — discard shorter segments
DEFAULT_MAX_SPEECH_DURATION = 10.0 # seconds — force-flush if speech goes longer


class AudioCapture:
    """Captures microphone audio and emits speech segments via a callback."""

    def __init__(
        self,
        device: Optional[int] = None,
        sample_rate: int = SAMPLE_RATE,
        silence_threshold: float = DEFAULT_SILENCE_THRESHOLD,
        silence_duration: float = DEFAULT_SILENCE_DURATION,
        min_speech_duration: float = DEFAULT_MIN_SPEECH_DURATION,
        max_speech_duration: float = DEFAULT_MAX_SPEECH_DURATION,
    ):
        self.device = device
        self.sample_rate = sample_rate
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.min_speech_duration = min_speech_duration
        self.max_speech_duration = max_speech_duration

        self._chunk_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._running = False
        self._stream: Optional[sd.InputStream] = None
        self._processor_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self, on_segment: Callable[[np.ndarray], None]) -> None:
        """
        Open the microphone stream and start the VAD processing thread.

        Args:
            on_segment: Called with a 1-D float32 numpy array for each
                        detected speech segment.  Called from the processing
                        thread — keep it non-blocking or hand off to a queue.
        """
        self._running = True
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            device=self.device,
            blocksize=CHUNK_FRAMES,
            callback=self._sd_callback,
        )
        self._processor_thread = threading.Thread(
            target=self._process_loop,
            args=(on_segment,),
            daemon=True,
            name="audio-vad",
        )
        self._stream.start()
        self._processor_thread.start()
        logger.info(
            "Audio capture started (device=%s, rate=%d Hz, threshold=%.4f)",
            self.device or "default",
            self.sample_rate,
            self.silence_threshold,
        )

    def stop(self) -> None:
        """Stop the microphone stream and VAD thread."""
        self._running = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if self._processor_thread is not None:
            self._processor_thread.join(timeout=2.0)
        logger.info("Audio capture stopped.")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _sd_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        if status:
            logger.debug("sounddevice status: %s", status)
        self._chunk_queue.put(indata[:, 0].copy())  # mono slice

    def _process_loop(self, on_segment: Callable[[np.ndarray], None]) -> None:
        speech_chunks: list[np.ndarray] = []
        silence_frame_count = 0
        in_speech = False

        silence_frames_needed = int(self.silence_duration / CHUNK_SECONDS)
        max_speech_frames = int(self.max_speech_duration / CHUNK_SECONDS)
        min_speech_frames = int(self.min_speech_duration / CHUNK_SECONDS)

        while self._running:
            try:
                chunk = self._chunk_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            rms = float(np.sqrt(np.mean(chunk ** 2)))
            is_voice = rms >= self.silence_threshold

            if is_voice:
                if not in_speech:
                    in_speech = True
                    logger.debug("Speech start detected (rms=%.4f)", rms)
                silence_frame_count = 0
                speech_chunks.append(chunk)
            elif in_speech:
                silence_frame_count += 1
                speech_chunks.append(chunk)  # include trailing silence so Whisper has context

                end_of_utterance = silence_frame_count >= silence_frames_needed
                too_long = len(speech_chunks) >= max_speech_frames

                if end_of_utterance or too_long:
                    if len(speech_chunks) >= min_speech_frames:
                        segment = np.concatenate(speech_chunks)
                        logger.debug(
                            "Emitting speech segment (%.2f s)", len(segment) / self.sample_rate
                        )
                        on_segment(segment)
                    speech_chunks = []
                    silence_frame_count = 0
                    in_speech = False


# ------------------------------------------------------------------
# Utility
# ------------------------------------------------------------------

def list_input_devices() -> list[dict]:
    """Return a list of available input audio devices."""
    devices = sd.query_devices()
    return [
        {"index": i, "name": d["name"], "channels": d["max_input_channels"]}
        for i, d in enumerate(devices)
        if d["max_input_channels"] > 0
    ]
