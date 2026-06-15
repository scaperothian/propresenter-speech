"""
Unit tests for MLXTranscriber.

mlx and mlx_whisper are mocked at the sys.modules level — no Apple GPU, no model
download — so these run anywhere.  They pin the Transcriber-compatible interface
(load/transcribe), repo resolution, and the documented ImportError.
"""

import importlib.machinery
import sys
import types

import numpy as np
import pytest

from propresenter_speech.transcriber_mlx import MLXTranscriber, _resolve_repo


def make_audio(seconds: float = 1.0, sample_rate: int = 16_000) -> np.ndarray:
    t = np.linspace(0, seconds, int(seconds * sample_rate), dtype=np.float32)
    return 0.5 * np.sin(2 * np.pi * 440 * t)


def _install_fake_mlx(monkeypatch, text=" next slide "):
    """Inject fake mlx.core, mlx_whisper, and mlx_whisper.transcribe modules."""
    mlx = types.ModuleType("mlx")
    mlx_core = types.ModuleType("mlx.core")
    mlx_core.float16 = "float16"
    mlx.core = mlx_core

    transcribe_calls = []

    mlx_whisper = types.ModuleType("mlx_whisper")

    def fake_transcribe(audio, **kwargs):
        transcribe_calls.append((audio, kwargs))
        return {"text": text}

    mlx_whisper.transcribe = fake_transcribe

    # Submodule mlx_whisper.transcribe (shadowed by the function attribute, but
    # importlib.import_module reaches it by sys.modules key) holds ModelHolder.
    tmod = types.ModuleType("mlx_whisper.transcribe")
    load_calls = []

    class ModelHolder:
        @staticmethod
        def get_model(repo, dtype):
            load_calls.append((repo, dtype))

    tmod.ModelHolder = ModelHolder

    for mod in (mlx, mlx_core, mlx_whisper, tmod):
        mod.__spec__ = importlib.machinery.ModuleSpec(mod.__name__, None)
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", mlx_core)
    monkeypatch.setitem(sys.modules, "mlx_whisper", mlx_whisper)
    monkeypatch.setitem(sys.modules, "mlx_whisper.transcribe", tmod)
    return load_calls, transcribe_calls


class TestResolveRepo:
    def test_known_sizes_map_to_mlx_community(self):
        assert _resolve_repo("base") == "mlx-community/whisper-base-mlx"
        assert _resolve_repo("large") == "mlx-community/whisper-large-v3-mlx"

    def test_full_repo_passthrough(self):
        assert _resolve_repo("mlx-community/whisper-large-v3-turbo") == (
            "mlx-community/whisper-large-v3-turbo"
        )

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown Whisper model"):
            _resolve_repo("gigantic")


class TestLifecycle:
    def test_device_is_mlx(self):
        assert MLXTranscriber().device == "mlx"

    def test_default_model_is_base(self):
        assert MLXTranscriber().model_name == "base"

    def test_not_loaded_on_init(self):
        assert not MLXTranscriber().is_loaded

    def test_load_primes_model_holder(self, monkeypatch):
        load_calls, _ = _install_fake_mlx(monkeypatch)
        t = MLXTranscriber(model_name="small")
        t.load()
        assert t.is_loaded
        assert load_calls == [("mlx-community/whisper-small-mlx", "float16")]

    def test_load_is_idempotent(self, monkeypatch):
        load_calls, _ = _install_fake_mlx(monkeypatch)
        t = MLXTranscriber()
        t.load()
        t.load()
        assert len(load_calls) == 1

    def test_load_without_mlx_raises(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "mlx", None)
        with pytest.raises(ImportError, match="whisper-mlx"):
            MLXTranscriber().load()


class TestTranscription:
    def test_returns_stripped_text(self, monkeypatch):
        _install_fake_mlx(monkeypatch, text="  hello world  ")
        t = MLXTranscriber()
        t.load()
        assert t.transcribe(make_audio()) == "hello world"

    def test_triggers_load_if_not_loaded(self, monkeypatch):
        load_calls, _ = _install_fake_mlx(monkeypatch)
        t = MLXTranscriber()
        assert not t.is_loaded
        t.transcribe(make_audio())
        assert len(load_calls) == 1

    def test_passes_repo_and_fp16_and_language(self, monkeypatch):
        _, calls = _install_fake_mlx(monkeypatch)
        t = MLXTranscriber(model_name="base")
        t.load()
        t.transcribe(make_audio(), language="en")
        _audio, kwargs = calls[-1]
        assert kwargs["path_or_hf_repo"] == "mlx-community/whisper-base-mlx"
        assert kwargs["fp16"] is True
        assert kwargs["temperature"] == 0.0
        assert kwargs["language"] == "en"

    def test_omits_language_when_none(self, monkeypatch):
        _, calls = _install_fake_mlx(monkeypatch)
        t = MLXTranscriber()
        t.load()
        t.transcribe(make_audio(), language=None)
        _audio, kwargs = calls[-1]
        assert "language" not in kwargs

    def test_empty_text_returns_empty(self, monkeypatch):
        _install_fake_mlx(monkeypatch, text="")
        t = MLXTranscriber()
        t.load()
        assert t.transcribe(make_audio()) == ""
