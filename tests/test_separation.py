"""
Unit tests for the source-separation stage.

DemucsSeparator is tested with demucs mocked out (no model download, no GPU);
the pipeline-ordering tests use a stub separator and need no torch at all.
"""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from propresenter_speech.audio_pipeline import _BasePipeline
from propresenter_speech.separation import DemucsSeparator
from propresenter_speech.separation.demucs import _resolve_device


class TestLoadImportError:
    def test_load_without_demucs_raises_with_install_hint(self):
        # Pre-import torch so patch.dict's sys.modules snapshot includes its
        # submodules; otherwise restoring the snapshot evicts them and the next
        # `import torch` re-runs its init, which torch does not survive.
        try:
            import torch  # noqa: F401
        except ImportError:
            pass
        separator = DemucsSeparator()
        with patch.dict(
            sys.modules,
            {"demucs": None, "demucs.apply": None, "demucs.pretrained": None},
        ):
            with pytest.raises(ImportError, match="poetry install --extras separation"):
                separator.load()


class TestResolveDevice:
    def test_explicit_device_passes_through(self):
        assert _resolve_device("cpu") == "cpu"
        assert _resolve_device("mps") == "mps"
        assert _resolve_device("cuda") == "cuda"

    def test_auto_prefers_cuda(self):
        torch = pytest.importorskip("torch")
        with patch.object(torch.cuda, "is_available", return_value=True):
            assert _resolve_device("auto") == "cuda"

    def test_auto_falls_back_to_cpu(self):
        torch = pytest.importorskip("torch")
        with patch.object(torch.cuda, "is_available", return_value=False), patch.object(
            torch.backends.mps, "is_available", return_value=False
        ):
            assert _resolve_device("auto") == "cpu"


class TestSeparate:
    @pytest.fixture
    def separator(self):
        torch = pytest.importorskip("torch")
        pytest.importorskip("demucs")

        fake_model = MagicMock()
        fake_model.sources = ["drums", "bass", "other", "vocals"]
        fake_model.segment = 7.8
        fake_model.to.return_value = fake_model
        fake_model.eval.return_value = fake_model

        def fake_apply_model(model, wav, **kwargs):
            batch, channels, n_frames = wav.shape
            stems = torch.ones(batch, len(fake_model.sources), channels, n_frames)
            stems[:, fake_model.sources.index("vocals")] = 0.0
            return stems

        with patch("demucs.pretrained.get_model", return_value=fake_model), patch(
            "demucs.apply.apply_model", side_effect=fake_apply_model
        ):
            separator = DemucsSeparator(device="cpu")
            separator.load()
            yield separator

    def test_output_shape_and_dtype(self, separator):
        audio = np.sin(np.linspace(0, 200 * np.pi, 32000)).astype(np.float32)
        result = separator.separate(audio)
        assert result.dtype == np.float32
        assert result.ndim == 1
        assert len(result) == len(audio)

    def test_vocals_stem_drives_output(self, separator):
        # Fake apply_model returns zeros only in the vocals stem; a zero-mean
        # input therefore yields a near-zero output iff the vocals index was
        # selected (other stems would denormalise to ~ref_std).
        audio = np.sin(np.linspace(0, 200 * np.pi, 32000)).astype(np.float32)
        result = separator.separate(audio)
        assert np.abs(result).max() < 0.05

    def test_model_receives_stereo_44k_batch(self, separator):
        torch = pytest.importorskip("torch")
        captured = {}

        def capture_apply_model(model, wav, **kwargs):
            captured["shape"] = tuple(wav.shape)
            return torch.zeros(wav.shape[0], 4, wav.shape[1], wav.shape[2])

        with patch("demucs.apply.apply_model", side_effect=capture_apply_model):
            separator.separate(np.zeros(16000, dtype=np.float32))
        assert captured["shape"] == (1, 2, 44100)


class _StubPredictor:
    def __init__(self):
        self.received = None

    def predict(self, audio):
        self.received = audio
        return "result"


class _StubHandler:
    def __init__(self):
        self.predictions = []

    def on_startup(self):
        pass

    def startup_description(self):
        return ""

    def on_prediction(self, result, audio_time):
        self.predictions.append((result, audio_time))


class _StubSeparator:
    def separate(self, audio):
        return audio * 0.5


class TestPipelineOrdering:
    def test_separator_applied_before_predict(self):
        predictor = _StubPredictor()
        handler = _StubHandler()
        pipeline = _BasePipeline(
            predictor, handler, window_seconds=2.0, poll_interval=0.2,
            separator=_StubSeparator(),
        )
        audio = np.ones(8000, dtype=np.float32)
        pipeline._process(audio)
        np.testing.assert_array_equal(predictor.received, audio * 0.5)
        assert handler.predictions == [("result", 0.0)]

    def test_no_separator_leaves_audio_untouched(self):
        predictor = _StubPredictor()
        pipeline = _BasePipeline(
            predictor, _StubHandler(), window_seconds=2.0, poll_interval=0.2,
        )
        audio = np.ones(8000, dtype=np.float32)
        pipeline._process(audio)
        np.testing.assert_array_equal(predictor.received, audio)
