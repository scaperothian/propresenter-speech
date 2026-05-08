"""
Whisper ASR wrapper using faster-whisper (CTranslate2 backend).

faster-whisper is a reimplementation of OpenAI Whisper that runs 4x faster on
CPU and uses half the memory.  Models are downloaded automatically from
HuggingFace on first use and cached in ~/.cache/huggingface/hub/.

Available model sizes (speed vs. accuracy tradeoff on CPU):
  tiny   ~39M params  — fastest, lowest accuracy
  base   ~74M params  — good balance (default)
  small  ~244M params — better accuracy, noticeable slowdown on CPU
  medium ~769M params — high accuracy, slow on CPU
  large-v3  ~1.5B params — best accuracy, very slow without a GPU
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

WHISPER_SAMPLE_RATE = 16_000  # Hz — Whisper always expects 16 kHz mono audio


class Transcriber:
    """Lazy-loading wrapper around faster-whisper's WhisperModel."""

    def __init__(
        self,
        model_name: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
    ):
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self._model = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Download (if necessary) and load the Whisper model into memory."""
        if self._model is not None:
            return
        from faster_whisper import WhisperModel  # imported here so tests can patch

        logger.info(
            "Loading Whisper model '%s' (device=%s, compute_type=%s)…",
            self.model_name, self.device, self.compute_type,
        )
        self._model = WhisperModel(
            self.model_name,
            device=self.device,
            compute_type=self.compute_type,
        )
        logger.info("Whisper model '%s' ready.", self.model_name)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    # ------------------------------------------------------------------
    # Transcription
    # ------------------------------------------------------------------

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = WHISPER_SAMPLE_RATE,
        language: Optional[str] = "en",
    ) -> str:
        """
        Transcribe a numpy audio array to text.

        Args:
            audio:       1-D float32 array at ``sample_rate`` Hz.  int16 input
                         is normalised to [-1, 1] automatically.
            sample_rate: Sample rate of the incoming audio.  Resampled to
                         16 kHz when it differs.
            language:    BCP-47 language hint.  Pass None to auto-detect.

        Returns:
            Transcribed text, or an empty string if nothing was recognised.
        """
        if not self.is_loaded:
            self.load()

        audio = _prepare_audio(audio, sample_rate)

        opts: dict = {"beam_size": 5}
        if language:
            opts["language"] = language

        segments, _ = self._model.transcribe(audio, **opts)
        text = " ".join(seg.text.strip() for seg in segments).strip()
        logger.debug("Transcribed: %r", text)
        return text


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _prepare_audio(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Return a 16 kHz mono float32 array ready for Whisper."""
    audio = np.asarray(audio)

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    if sample_rate != WHISPER_SAMPLE_RATE:
        audio = _resample(audio, sample_rate, WHISPER_SAMPLE_RATE)

    return audio


def _resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Nearest-neighbour integer resampling (adequate for speech)."""
    if src_rate == dst_rate:
        return audio
    new_len = int(len(audio) * dst_rate / src_rate)
    indices = np.round(np.linspace(0, len(audio) - 1, new_len)).astype(int)
    return audio[indices]
