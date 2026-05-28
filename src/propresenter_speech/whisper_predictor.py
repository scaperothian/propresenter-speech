"""
WhisperPredictor: wraps Transcriber and produces TranscriptionResults.

Owns the rolling word buffer so that the pipeline itself has no knowledge of
text or word tokenisation.
"""

from __future__ import annotations

import collections

import numpy as np

from .predictor import TranscriptionResult
from .slide_follower import extract_words
from .transcriber import Transcriber


class WhisperPredictor:
    """Transcribes audio via Whisper and maintains a cumulative word buffer."""

    def __init__(self, transcriber: Transcriber, verbose: bool = False):
        self._transcriber = transcriber
        self._verbose = verbose
        self._word_buffer: collections.deque[str] = collections.deque(maxlen=200)

    def predict(self, audio: np.ndarray) -> TranscriptionResult:
        text = self._transcriber.transcribe(audio)
        if text.strip():
            if self._verbose:
                print(f"  heard: {text!r}")
            self._word_buffer.extend(extract_words(text))
        return TranscriptionResult(text=text, word_buffer=self._word_buffer)
