"""
MERTPredictor: wraps m-a-p/MERT-v1-95M for audio-embedding-based slide matching.

MERT produces frame-level representations from 24 kHz audio.  This predictor
resamples the pipeline's 16 kHz audio to 24 kHz internally, runs inference,
and mean-pools the last hidden state to produce a single embedding vector per
window.

Requires the torch extra: poetry install --extras torch
"""

from __future__ import annotations

import logging

import numpy as np

from .base import AudioEmbeddingResult

logger = logging.getLogger(__name__)

_MERT_MODEL = "m-a-p/MERT-v1-95M"
_MERT_SAMPLE_RATE = 24_000
_PIPELINE_SAMPLE_RATE = 16_000


def _resample_to_mert(audio: np.ndarray) -> np.ndarray:
    """Linear-interpolation upsample from 16 kHz to 24 kHz."""
    n_out = int(len(audio) * _MERT_SAMPLE_RATE / _PIPELINE_SAMPLE_RATE)
    return np.interp(
        np.linspace(0, len(audio) - 1, n_out),
        np.arange(len(audio)),
        audio,
    ).astype(np.float32)


class MERTPredictor:
    """
    Produces a mean-pooled MERT embedding for each audio window.

    The pipeline captures at 16 kHz; audio is upsampled to 24 kHz before
    passing to MERT.  Returns AudioEmbeddingResult with a 1-D float32 array.
    """

    def __init__(self, model_name: str = _MERT_MODEL, verbose: bool = False):
        self._model_name = model_name
        self._verbose = verbose
        self._model = None
        self._processor = None

    def load(self) -> None:
        try:
            from transformers import AutoModel, Wav2Vec2FeatureExtractor
        except ImportError as exc:
            raise ImportError("torch extras required: poetry install --extras torch") from exc

        logging.getLogger("transformers").setLevel(logging.WARNING)
        logger.info("Loading MERT model: %s", self._model_name)
        self._processor = Wav2Vec2FeatureExtractor.from_pretrained(
            self._model_name, trust_remote_code=True
        )
        self._model = AutoModel.from_pretrained(
            self._model_name, trust_remote_code=True
        )
        self._model.eval()
        logger.info("MERT model ready.")

    def embed_24k(self, audio_24k: np.ndarray) -> np.ndarray:
        """Run MERT on pre-resampled 24 kHz audio; return mean-pooled embedding [D]."""
        if self._model is None:
            self.load()
        import torch

        inputs = self._processor(
            audio_24k,
            sampling_rate=_MERT_SAMPLE_RATE,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = self._model(**inputs)
        return outputs.last_hidden_state[0].mean(dim=0).numpy()

    def predict(self, audio: np.ndarray) -> AudioEmbeddingResult:
        """Resample 16 kHz input to 24 kHz and return AudioEmbeddingResult."""
        audio_24k = _resample_to_mert(audio)
        embedding = self.embed_24k(audio_24k)

        if self._verbose:
            logger.debug("MERT embedding norm: %.4f", float(np.linalg.norm(embedding)))

        return AudioEmbeddingResult(embedding=embedding)
