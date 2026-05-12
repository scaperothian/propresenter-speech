"""
Unit tests for AudioFileCapture.
No real audio files needed — soundfile.read is patched with synthetic numpy arrays.
"""

import threading
import numpy as np
import pytest
from unittest.mock import patch

from propresenter_speech.audio_capture import (
    AudioFileCapture,
    SAMPLE_RATE,
    _resample,
)


def _sine(duration_s: float, freq: float = 440.0, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Generate a mono float32 sine wave."""
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    return (np.sin(2 * np.pi * freq * t) * 0.5).astype(np.float32)


def _silence(duration_s: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    return np.zeros(int(sr * duration_s), dtype=np.float32)


# ---------------------------------------------------------------------------
# _resample
# ---------------------------------------------------------------------------

class TestResample:
    def test_same_rate_returns_unchanged(self):
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


# ---------------------------------------------------------------------------
# AudioFileCapture — VAD behaviour
# ---------------------------------------------------------------------------

class TestAudioFileCapture:
    def _run_capture(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE, **kwargs) -> list[np.ndarray]:
        """Run AudioFileCapture against synthetic audio; collect emitted segments."""
        segments: list[np.ndarray] = []
        done = threading.Event()

        capture = AudioFileCapture(file_path="fake.wav", **kwargs)

        def on_segment(seg: np.ndarray) -> None:
            segments.append(seg)

        with patch("soundfile.read", return_value=(audio, sample_rate)):
            capture.start(on_segment)
            capture._thread.join(timeout=10)

        return segments

    def test_emits_segment_for_loud_audio(self):
        audio = np.concatenate([_silence(0.5), _sine(1.5), _silence(1.0)])
        segments = self._run_capture(audio)
        assert len(segments) == 1

    def test_emits_multiple_segments_for_separate_utterances(self):
        audio = np.concatenate([
            _sine(1.0),
            _silence(1.5),
            _sine(1.0),
            _silence(1.0),
        ])
        segments = self._run_capture(audio)
        assert len(segments) == 2

    def test_silent_file_emits_no_segments(self):
        audio = _silence(2.0)
        segments = self._run_capture(audio)
        assert len(segments) == 0

    def test_resamples_non_16k_audio(self):
        audio_44k = _sine(1.0, sr=44100)
        segments = self._run_capture(audio_44k, sample_rate=44100)
        assert len(segments) == 1

    def test_mixes_stereo_to_mono(self):
        mono = _sine(1.5)
        stereo = np.stack([mono, mono], axis=1)
        segments = self._run_capture(stereo)
        assert len(segments) == 1

    def test_stop_interrupts_processing(self):
        audio = _sine(5.0)
        segments: list[np.ndarray] = []
        capture = AudioFileCapture(file_path="fake.wav")

        with patch("soundfile.read", return_value=(audio, SAMPLE_RATE)):
            capture.start(lambda seg: segments.append(seg))
            capture.stop()
