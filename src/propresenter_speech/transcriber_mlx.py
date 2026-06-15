"""
Whisper ASR wrapper using mlx-whisper (Apple GPU via MLX).

A drop-in alternative to ``Transcriber`` exposing the same ``transcribe()``
interface, so ``WhisperPredictor`` can use either.  faster-whisper (CTranslate2)
has no Metal backend and is CPU-only on Mac; mlx-whisper runs the same Whisper
models on the Apple GPU in fp16, ~3–4x faster — the only way to run small/medium
in real time on Apple Silicon.

Requires the whisper-mlx extra: poetry install --extras whisper-mlx (Apple Silicon).
"""

from __future__ import annotations

import importlib
import logging
from typing import Optional

import numpy as np

from .transcriber import WHISPER_SAMPLE_RATE, _prepare_audio

logger = logging.getLogger(__name__)

# --model size → mlx-community HF repo.  The "-mlx" suffix is consistent across
# sizes; "large" maps to large-v3.  A value containing "/" is used verbatim so
# callers can point at any repo (e.g. mlx-community/whisper-large-v3-turbo).
_MODEL_REPOS = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large": "mlx-community/whisper-large-v3-mlx",
}


def _resolve_repo(model_name: str) -> str:
    if "/" in model_name:
        return model_name
    try:
        return _MODEL_REPOS[model_name]
    except KeyError:
        raise ValueError(
            f"Unknown Whisper model '{model_name}'. Choose one of "
            f"{', '.join(_MODEL_REPOS)} or pass a full HF repo id."
        )


class MLXTranscriber:
    """Lazy-loading mlx-whisper wrapper with the same API as ``Transcriber``."""

    def __init__(self, model_name: str = "base"):
        self.model_name = model_name
        self.device = "mlx"
        self._repo = _resolve_repo(model_name)
        self._loaded = False

    def load(self) -> None:
        """Download (if necessary) and load the model so the first window isn't delayed."""
        if self._loaded:
            return
        try:
            import mlx.core as mx
        except ImportError:
            raise ImportError("whisper-mlx extras required: poetry install --extras whisper-mlx")
        # ModelHolder is the fp16 cache mlx_whisper.transcribe() reads from;
        # priming it here moves the load cost to startup instead of first poll.
        tmod = importlib.import_module("mlx_whisper.transcribe")
        logger.info("Loading mlx-whisper model '%s' (Apple GPU)…", self._repo)
        tmod.ModelHolder.get_model(self._repo, mx.float16)
        self._loaded = True
        logger.info("mlx-whisper model '%s' ready.", self._repo)

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = WHISPER_SAMPLE_RATE,
        language: Optional[str] = "en",
    ) -> str:
        """Transcribe a numpy audio array to text (16 kHz mono; resampled if needed)."""
        if not self._loaded:
            self.load()
        import mlx_whisper

        audio = _prepare_audio(audio, sample_rate)
        opts: dict = {"path_or_hf_repo": self._repo, "fp16": True, "temperature": 0.0}
        if language:
            opts["language"] = language

        result = mlx_whisper.transcribe(audio, **opts)
        text = (result.get("text") or "").strip()
        logger.debug("Transcribed: %r", text)
        return text
