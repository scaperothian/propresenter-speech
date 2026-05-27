"""
File-mode audio pipeline for offline evaluation.

Audio file
    │  sliding window (window_seconds wide, advances by poll_interval)
    ▼
FilePipeline       (synchronous — no threading)
    │  audio chunk
    ▼
Predictor.predict(chunk) → result
    │
    ▼
ModeHandler.on_prediction(result, audio_time)

audio_time is T_snap — the file position (seconds) at the END of the
transcribed window, derived from frame counts with no wall-clock jitter.

For live microphone capture see audio_pipeline.AudioPipeline.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import numpy as np
import sounddevice as sd

from .audio_pipeline import (
    _BasePipeline,
    DEFAULT_POLL_INTERVAL,
    DEFAULT_WINDOW_SECONDS,
    SAMPLE_RATE,
)

if TYPE_CHECKING:
    from .handlers.base import ModeHandler
    from .predictor import Predictor

logger = logging.getLogger(__name__)


class FilePipeline(_BasePipeline):
    """File-mode pipeline: sliding-window audio file → Predictor → ModeHandler."""

    def __init__(
        self,
        predictor: "Predictor",
        handler: "ModeHandler",
        audio_file: str,
        window_seconds: float = DEFAULT_WINDOW_SECONDS,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        playback: bool = False,
        output_device: Optional[int] = None,
    ):
        super().__init__(predictor, handler, window_seconds, poll_interval)
        self.audio_file = audio_file
        self.playback = playback
        self.output_device = output_device

    def run(self) -> None:
        self.handler.on_startup()
        print(self.handler.startup_description())
        self._running = True
        self._run_file()

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


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Linear-interpolation resample — adequate quality for 16 kHz voice."""
    n_samples = int(len(audio) * target_sr / orig_sr)
    return np.interp(
        np.linspace(0, len(audio) - 1, n_samples),
        np.arange(len(audio)),
        audio,
    ).astype(np.float32)
