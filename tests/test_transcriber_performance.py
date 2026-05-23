"""
Real-time factor (RTF) benchmark for Whisper transcription.

RTF = transcription_time / audio_duration
  RTF < 1.0  →  faster than real time; pipeline can keep up
  RTF > 1.0  →  slower than real time; pipeline lags behind audio

Run with:
  poetry run pytest tests/test_transcriber_performance.py -v -s

The test always passes as long as the model is functional — the RTF
result is printed for human inspection.  Only an absurd RTF (>20x) is
treated as a hard failure so CI catches completely broken environments.
"""

import time

import numpy as np
import pytest

from propresenter_speech.audio_pipeline import SAMPLE_RATE
from propresenter_speech.transcriber import Transcriber

CHUNK_SECONDS = 2.0
WARMUP_RUNS = 1
TIMED_RUNS = 5


def _silent_chunk(seconds: float = CHUNK_SECONDS) -> np.ndarray:
    """Return a silent float32 PCM array at 16 kHz — consistent worst-case input."""
    return np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)


@pytest.fixture(scope="module")
def tiny_model() -> Transcriber:
    t = Transcriber("tiny")
    t.load()
    return t


@pytest.fixture(scope="module")
def base_model() -> Transcriber:
    t = Transcriber("base")
    t.load()
    return t


def _measure_rtf(transcriber: Transcriber, chunk: np.ndarray) -> tuple[float, float]:
    """
    Return (avg_ms, rtf) for TIMED_RUNS transcriptions of chunk.
    Runs WARMUP_RUNS first to prime any JIT / cache effects.
    """
    for _ in range(WARMUP_RUNS):
        transcriber.transcribe(chunk)

    times = []
    for _ in range(TIMED_RUNS):
        t0 = time.perf_counter()
        transcriber.transcribe(chunk)
        times.append(time.perf_counter() - t0)

    avg_ms = sum(times) / len(times) * 1000
    rtf = (avg_ms / 1000) / (len(chunk) / SAMPLE_RATE)
    return avg_ms, rtf


def _report(model_name: str, avg_ms: float, rtf: float) -> None:
    status = (
        "faster than real time"
        if rtf < 1.0
        else f"slower than real time — lags ~{(rtf - 1) * CHUNK_SECONDS * 1000:.0f} ms/chunk"
    )
    print(
        f"\n  model={model_name}  chunk={CHUNK_SECONDS:.1f}s  "
        f"avg={avg_ms:.0f}ms  RTF={rtf:.3f}x  →  {status}"
    )


class TestRealTimeFactor:
    def test_tiny_model_rtf(self, tiny_model, capsys):
        chunk = _silent_chunk()
        avg_ms, rtf = _measure_rtf(tiny_model, chunk)
        with capsys.disabled():
            _report("tiny", avg_ms, rtf)
        assert rtf < 20.0, f"tiny RTF {rtf:.2f}x exceeds sanity limit"

    def test_base_model_rtf(self, base_model, capsys):
        chunk = _silent_chunk()
        avg_ms, rtf = _measure_rtf(base_model, chunk)
        with capsys.disabled():
            _report("base", avg_ms, rtf)
        assert rtf < 20.0, f"base RTF {rtf:.2f}x exceeds sanity limit"

    def test_pipeline_headroom(self, tiny_model, capsys):
        """
        Estimates whether the pipeline can keep up in real time.

        Total per-chunk cost = sleep(window) + transcription.
        If transcription alone exceeds window_seconds, the pipeline
        falls behind by that excess on every chunk.
        """
        chunk = _silent_chunk()
        avg_ms, rtf = _measure_rtf(tiny_model, chunk)
        lag_per_chunk_ms = max(0.0, avg_ms - CHUNK_SECONDS * 1000)
        with capsys.disabled():
            if lag_per_chunk_ms == 0:
                print(f"\n  pipeline headroom: {CHUNK_SECONDS * 1000 - avg_ms:.0f} ms/chunk (no lag)")
            else:
                print(f"\n  pipeline lag: ~{lag_per_chunk_ms:.0f} ms/chunk — consider --model tiny or --window-seconds 3.0")
