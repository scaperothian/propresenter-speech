"""
Unit tests for AudioPipeline utilities.
The resample function is tested directly; the pipeline run loop is not tested
here since it requires real sounddevice hardware.
"""

import numpy as np
import pytest

from propresenter_speech.audio_pipeline import _resample, SAMPLE_RATE


class TestResample:
    def test_same_rate_returns_same_length(self):
        audio = np.ones(16000, dtype=np.float32)
        result = _resample(audio, 16000, 16000)
        assert len(result) == 16000

    def test_upsample_length(self):
        audio = np.ones(8000, dtype=np.float32)
        result = _resample(audio, 8000, 16000)
        assert len(result) == 16000

    def test_downsample_length(self):
        audio = np.ones(48000, dtype=np.float32)
        result = _resample(audio, 48000, 16000)
        assert len(result) == 16000

    def test_output_dtype_is_float32(self):
        audio = np.ones(8000, dtype=np.float64)
        result = _resample(audio, 8000, 16000)
        assert result.dtype == np.float32
