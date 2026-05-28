"""
Wav2VecPredictor: wraps Wav2Vec2ForCTC for ASR transcription.

Requires the torch extra: poetry install --extras torch
"""

from __future__ import annotations

import collections
import logging

import numpy as np

from .base import TranscriptionResult
from ..slide_follower import extract_words

logger = logging.getLogger(__name__)

DEFAULT_WAV2VEC_MODEL = "facebook/wav2vec2-large-960h-lv60-self"
_SAMPLE_RATE = 16_000


class Wav2VecPredictor:
    """Transcribes audio via HuggingFace Wav2Vec2ForCTC and maintains a cumulative word buffer."""

    def __init__(self, model_name: str = DEFAULT_WAV2VEC_MODEL, verbose: bool = False):
        self._model_name = model_name
        self._verbose = verbose
        self._processor = None
        self._model = None
        self._word_buffer: collections.deque[str] = collections.deque(maxlen=200)

    def load(self) -> None:
        try:
            import torch  # noqa: F401
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        except ImportError:
            raise ImportError("torch extras required: poetry install --extras torch")
        logging.getLogger("transformers").setLevel(logging.WARNING)
        logger.info("Loading Wav2Vec2 processor and model '%s'…", self._model_name)
        self._processor = Wav2Vec2Processor.from_pretrained(self._model_name)
        self._model = Wav2Vec2ForCTC.from_pretrained(self._model_name)
        self._model.eval()
        logger.info("Wav2Vec2 model ready.")

    def predict(self, audio: np.ndarray) -> TranscriptionResult:
        if self._model is None:
            self.load()
        import torch
        inputs = self._processor(
            audio, sampling_rate=_SAMPLE_RATE, return_tensors="pt", padding=False
        )
        with torch.no_grad():
            logits = self._model(inputs.input_values).logits
        ids = torch.argmax(logits, dim=-1)
        text = self._processor.decode(ids[0]).strip()
        if text:
            if self._verbose:
                print(f"  heard: {text!r}")
            self._word_buffer.extend(extract_words(text))
        return TranscriptionResult(text=text, word_buffer=self._word_buffer)
