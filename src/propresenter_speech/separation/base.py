"""
SourceSeparator protocol.

A SourceSeparator transforms a raw audio chunk into an isolated-vocals chunk
before it reaches the Predictor.  The pipeline is agnostic to the separation
strategy — Demucs, Spleeter, Open-Unmix, etc. can all conform to this protocol.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class SourceSeparator(Protocol):
    """Transform a 16 kHz float32 mono chunk into an isolated-vocals chunk of equal length."""

    def separate(self, audio: np.ndarray) -> np.ndarray: ...
