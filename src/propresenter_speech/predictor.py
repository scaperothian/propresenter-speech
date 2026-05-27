"""
Predictor protocol and shared result types.

A Predictor converts a raw audio chunk into a prediction result that a
ModeHandler can act on.  The pipeline is agnostic to the prediction strategy —
Whisper transcription, MERT embeddings, HMMs, etc. can all conform to this
protocol.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


class Predictor(Protocol):
    """Convert an audio chunk into a result, or None to skip this chunk."""

    def predict(self, audio: np.ndarray) -> Any | None: ...


@dataclass
class TranscriptionResult:
    """Result produced by a text-based predictor (e.g. WhisperPredictor)."""

    text: str
    word_buffer: deque  # cumulative rolling word history for the session
