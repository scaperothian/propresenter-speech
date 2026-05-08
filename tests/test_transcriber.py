"""
Unit tests for Transcriber.

faster-whisper is never actually imported — we patch it at module level so
these tests run without CTranslate2 or any model weights.
"""

# pylint: disable=redefined-outer-name,unused-argument
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from propresenter_speech.transcriber import (
    Transcriber,
    WHISPER_SAMPLE_RATE,
    _prepare_audio,
    _resample,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def make_audio(seconds: float = 1.0, sample_rate: int = WHISPER_SAMPLE_RATE) -> np.ndarray:
    """Synthetic sine-wave audio at 440 Hz."""
    t = np.linspace(0, seconds, int(seconds * sample_rate), dtype=np.float32)
    return 0.5 * np.sin(2 * np.pi * 440 * t)


def _make_segment(text: str) -> MagicMock:
    seg = MagicMock()
    seg.text = text
    return seg


@pytest.fixture
def mock_faster_whisper():
    """Patch the faster_whisper module imported inside transcriber.py."""
    with patch("faster_whisper.WhisperModel") as mock_cls:
        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter([_make_segment("next slide")]), MagicMock())
        mock_cls.return_value = mock_model
        yield mock_cls, mock_model


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

class TestTranscriberLifecycle:
    def test_is_not_loaded_on_init(self):
        t = Transcriber()
        assert not t.is_loaded

    def test_load_instantiates_whisper_model(self, mock_faster_whisper):
        mock_cls, _ = mock_faster_whisper
        t = Transcriber(model_name="tiny")
        t.load()
        mock_cls.assert_called_once_with("tiny", device="cpu", compute_type="int8")

    def test_is_loaded_after_load(self, mock_faster_whisper):
        t = Transcriber()
        t.load()
        assert t.is_loaded

    def test_load_is_idempotent(self, mock_faster_whisper):
        mock_cls, _ = mock_faster_whisper
        t = Transcriber()
        t.load()
        t.load()
        mock_cls.assert_called_once()

    def test_default_model_name_is_base(self):
        assert Transcriber().model_name == "base"

    def test_custom_model_name_stored(self):
        assert Transcriber(model_name="small").model_name == "small"

    def test_default_device_is_cpu(self):
        assert Transcriber().device == "cpu"

    def test_default_compute_type_is_int8(self):
        assert Transcriber().compute_type == "int8"


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

class TestTranscription:
    def test_transcribe_returns_joined_segment_text(self, mock_faster_whisper):
        _, mock_model = mock_faster_whisper
        mock_model.transcribe.return_value = (
            iter([_make_segment(" hello"), _make_segment(" world")]),
            MagicMock(),
        )
        t = Transcriber()
        t.load()
        assert t.transcribe(make_audio()) == "hello world"

    def test_transcribe_strips_outer_whitespace(self, mock_faster_whisper):
        _, mock_model = mock_faster_whisper
        mock_model.transcribe.return_value = (
            iter([_make_segment("  next slide  ")]),
            MagicMock(),
        )
        t = Transcriber()
        t.load()
        assert t.transcribe(make_audio()) == "next slide"

    def test_transcribe_returns_empty_on_no_segments(self, mock_faster_whisper):
        _, mock_model = mock_faster_whisper
        mock_model.transcribe.return_value = (iter([]), MagicMock())
        t = Transcriber()
        t.load()
        assert t.transcribe(make_audio()) == ""

    def test_transcribe_triggers_load_if_not_loaded(self, mock_faster_whisper):
        mock_cls, mock_model = mock_faster_whisper
        mock_model.transcribe.return_value = (iter([_make_segment("slide one")]), MagicMock())
        t = Transcriber()
        assert not t.is_loaded
        t.transcribe(make_audio())
        mock_cls.assert_called_once()

    def test_transcribe_passes_language_hint(self, mock_faster_whisper):
        _, mock_model = mock_faster_whisper
        mock_model.transcribe.return_value = (iter([_make_segment("next slide")]), MagicMock())
        t = Transcriber()
        t.load()
        t.transcribe(make_audio(), language="en")
        _, kwargs = mock_model.transcribe.call_args
        assert kwargs.get("language") == "en"

    def test_transcribe_omits_language_when_none(self, mock_faster_whisper):
        _, mock_model = mock_faster_whisper
        mock_model.transcribe.return_value = (iter([_make_segment("next slide")]), MagicMock())
        t = Transcriber()
        t.load()
        t.transcribe(make_audio(), language=None)
        _, kwargs = mock_model.transcribe.call_args
        assert "language" not in kwargs

    def test_transcribe_passes_beam_size(self, mock_faster_whisper):
        _, mock_model = mock_faster_whisper
        mock_model.transcribe.return_value = (iter([_make_segment("next slide")]), MagicMock())
        t = Transcriber()
        t.load()
        t.transcribe(make_audio())
        _, kwargs = mock_model.transcribe.call_args
        assert kwargs.get("beam_size") == 5


# ---------------------------------------------------------------------------
# _prepare_audio helper
# ---------------------------------------------------------------------------

class TestPrepareAudio:
    def test_float32_passthrough(self):
        audio = make_audio()
        result = _prepare_audio(audio, WHISPER_SAMPLE_RATE)
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, audio)

    def test_int16_normalised_to_float32(self):
        audio_int16 = (make_audio() * 32767).astype(np.int16)
        result = _prepare_audio(audio_int16, WHISPER_SAMPLE_RATE)
        assert result.dtype == np.float32
        assert result.max() <= 1.0
        assert result.min() >= -1.0

    def test_stereo_flattened_to_mono(self):
        stereo = np.stack([make_audio(), make_audio()], axis=1)
        result = _prepare_audio(stereo, WHISPER_SAMPLE_RATE)
        assert result.ndim == 1

    def test_resampling_changes_length_to_16k(self):
        audio_44k = make_audio(seconds=1.0, sample_rate=44100)
        result = _prepare_audio(audio_44k, 44100)
        assert abs(len(result) - WHISPER_SAMPLE_RATE) <= 2

    def test_correct_sample_rate_no_resample(self):
        audio = make_audio()
        result = _prepare_audio(audio, WHISPER_SAMPLE_RATE)
        assert len(result) == len(audio)


# ---------------------------------------------------------------------------
# _resample helper
# ---------------------------------------------------------------------------

class TestResample:
    def test_same_rate_returns_original(self):
        audio = make_audio()
        result = _resample(audio, WHISPER_SAMPLE_RATE, WHISPER_SAMPLE_RATE)
        np.testing.assert_array_equal(result, audio)

    def test_downsample_reduces_length(self):
        audio = make_audio(seconds=1.0, sample_rate=44100)
        result = _resample(audio, 44100, 16000)
        assert len(result) < len(audio)

    def test_upsample_increases_length(self):
        audio = make_audio(seconds=1.0, sample_rate=8000)
        result = _resample(audio, 8000, 16000)
        assert len(result) > len(audio)
