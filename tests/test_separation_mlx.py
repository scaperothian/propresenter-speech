"""
Unit tests for the MLX (demucs-mlx) source-separation backend.

mlx and demucs_mlx are mocked out entirely — no Apple GPU, no model download,
no native MLX install required — so these run anywhere alongside the rest of the
suite.  They pin the SourceSeparator contract: 16 kHz float32 mono in/out, equal
length, lazy load, and the documented ImportError when the extra is absent.
"""

import importlib.machinery
import sys
import types
from unittest.mock import patch

import numpy as np
import pytest

from propresenter_speech.separation import MLXDemucsSeparator, build_separator


def _install_fake_mlx(monkeypatch, separate_fn):
    """Inject fake ``mlx.core`` and ``demucs_mlx`` modules; return the FakeSeparator class."""
    mlx = types.ModuleType("mlx")
    mlx_core = types.ModuleType("mlx.core")
    mlx_core.array = lambda x: np.asarray(x, dtype=np.float32)
    mlx_core.eval = lambda *a, **k: None
    mlx.core = mlx_core

    demucs_mlx = types.ModuleType("demucs_mlx")

    class FakeSeparator:
        last_instance = None

        def __init__(self, model="htdemucs", shifts=1, overlap=0.25):
            self.model = types.SimpleNamespace(
                segment=7.8,
                samplerate=44_100,
                audio_channels=2,
                sources=["drums", "bass", "other", "vocals"],
            )
            self.split = None
            FakeSeparator.last_instance = self

        def update_parameter(self, *, split=None, **kwargs):
            if split is not None:
                self.split = split

        def separate_tensor(self, wav, return_mx=False):
            return separate_fn(np.asarray(wav))

    demucs_mlx.Separator = FakeSeparator
    # find_spec() (used by build_separator's auto path) raises if __spec__ is None.
    for mod in (mlx, mlx_core, demucs_mlx):
        mod.__spec__ = importlib.machinery.ModuleSpec(mod.__name__, None)
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", mlx_core)
    monkeypatch.setitem(sys.modules, "demucs_mlx", demucs_mlx)
    return FakeSeparator


def _vocals_zero_fn(wav):
    """vocals stem is silence; the other stems mirror the input."""
    stems = {name: wav.copy() for name in ("drums", "bass", "other")}
    stems["vocals"] = np.zeros_like(wav)
    return wav, stems


class TestLoadImportError:
    def test_load_without_demucs_mlx_raises_with_install_hint(self):
        separator = MLXDemucsSeparator()
        with patch.dict(sys.modules, {"demucs_mlx": None}):
            with pytest.raises(ImportError, match="separation-mlx"):
                separator.load()


class TestSeparate:
    @pytest.fixture
    def separator(self, monkeypatch):
        _install_fake_mlx(monkeypatch, _vocals_zero_fn)
        sep = MLXDemucsSeparator()
        sep.load()
        return sep

    def test_output_shape_and_dtype(self, separator):
        audio = np.sin(np.linspace(0, 200 * np.pi, 32000)).astype(np.float32)
        result = separator.separate(audio)
        assert result.dtype == np.float32
        assert result.ndim == 1
        assert len(result) == len(audio)

    def test_device_is_mlx(self, separator):
        assert separator.device == "mlx"

    def test_vocals_stem_drives_output(self, separator):
        # vocals stem is zeros; a zero-mean input denormalises back to ~0.
        audio = np.sin(np.linspace(0, 200 * np.pi, 32000)).astype(np.float32)
        result = separator.separate(audio)
        assert np.abs(result).max() < 0.05

    def test_model_receives_stereo_44k(self, separator):
        captured = {}

        def capture_fn(wav):
            captured["shape"] = tuple(wav.shape)
            return _vocals_zero_fn(wav)

        # Re-patch separate_tensor on the live fake instance.
        separator._sep.separate_tensor = lambda wav, return_mx=False: capture_fn(
            np.asarray(wav)
        )
        separator.separate(np.zeros(16000, dtype=np.float32))
        assert captured["shape"] == (2, 44100)

    def test_short_window_disables_split(self, separator):
        # 2 s < htdemucs 7.8 s segment → split=False for ~2x speedup.
        separator.separate(np.zeros(32000, dtype=np.float32))
        assert separator._sep.split is False

    def test_long_window_enables_split(self, separator):
        # 10 s > 7.8 s segment → must chunk (split=True).
        separator.separate(np.zeros(160000, dtype=np.float32))
        assert separator._sep.split is True


class TestLazyLoad:
    def test_separate_loads_on_first_call(self, monkeypatch):
        _install_fake_mlx(monkeypatch, _vocals_zero_fn)
        sep = MLXDemucsSeparator()
        assert sep._sep is None
        sep.separate(np.zeros(16000, dtype=np.float32))
        assert sep._sep is not None


class TestFactory:
    def test_auto_prefers_mlx_when_available(self, monkeypatch):
        _install_fake_mlx(monkeypatch, _vocals_zero_fn)
        sep = build_separator("auto", "htdemucs", log=lambda *a, **k: None)
        assert sep.device == "mlx"

    def test_explicit_mlx_backend(self, monkeypatch):
        _install_fake_mlx(monkeypatch, _vocals_zero_fn)
        sep = build_separator("demucs-mlx", "htdemucs", log=lambda *a, **k: None)
        assert isinstance(sep, MLXDemucsSeparator)
