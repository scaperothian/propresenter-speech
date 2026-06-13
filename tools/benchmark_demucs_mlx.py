#!/usr/bin/env python3
"""
Standalone demucs-mlx source-separation latency benchmark (Apple Silicon).

Measures per-call latency and real-time factor (RTF) for the full vocal-isolation
path used by the live pipeline: 16 kHz mono input → upsample to 44.1 kHz →
fake stereo → MLX separate_tensor → vocals stem → downmix → downsample to 16 kHz.

No propresenter-speech dependencies required; demucs-mlx must be installed
(poetry install --extras separation-mlx).  Runs on the Apple GPU via MLX, where
PyTorch Demucs cannot (the MPS 2^16 conv limit forces the torch path to CPU).

Usage:
    python tools/benchmark_demucs_mlx.py
    python tools/benchmark_demucs_mlx.py --duration 3.0 --runs 10 --warmup 2
    python tools/benchmark_demucs_mlx.py --model htdemucs_ft
"""

from __future__ import annotations

import argparse
import sys
import time

PIPELINE_SAMPLE_RATE = 16_000
DEMUCS_SAMPLE_RATE = 44_100
DEFAULT_MODEL = "htdemucs"


def _synthetic_audio(duration: float) -> "np.ndarray":
    """White noise at low amplitude — produces real separation work without silence skip."""
    import numpy as np
    rng = np.random.default_rng(42)
    return (rng.standard_normal(int(duration * PIPELINE_SAMPLE_RATE)) * 0.1).astype("float32")


def _resample(audio: "np.ndarray", orig_sr: int, target_sr: int) -> "np.ndarray":
    import numpy as np
    n_samples = int(len(audio) * target_sr / orig_sr)
    return np.interp(
        np.linspace(0, len(audio) - 1, n_samples),
        np.arange(len(audio)),
        audio,
    ).astype(np.float32)


def _run_model(
    model_name: str,
    audio: "np.ndarray",
    duration: float,
    runs: int,
    warmup: int,
) -> dict:
    import mlx.core as mx
    import numpy as np
    from demucs_mlx import Separator

    print(f"  loading …", end=" ", flush=True)
    t0 = time.perf_counter()
    sep = Separator(model=model_name, shifts=1, overlap=0.25)
    segment_sec = float(getattr(sep.model, "segment", 0.0) or 0.0)
    split = segment_sec <= 0 or duration > segment_sec
    sep.update_parameter(split=split)
    load_sec = time.perf_counter() - t0
    print(f"ready in {load_sec:.1f}s")

    def _separate():
        upsampled = _resample(audio, PIPELINE_SAMPLE_RATE, DEMUCS_SAMPLE_RATE)
        stereo = np.stack([upsampled, upsampled], axis=0)
        ref_mean = float(stereo.mean())
        ref_std = float(stereo.std()) + 1e-8
        stereo = (stereo - ref_mean) / ref_std
        _wav, stems = sep.separate_tensor(mx.array(stereo), return_mx=True)
        vocals = stems["vocals"]
        mx.eval(vocals)
        vocals_np = np.asarray(vocals, dtype=np.float32) * ref_std + ref_mean
        if vocals_np.ndim == 2:
            ch_axis = 0 if vocals_np.shape[0] < vocals_np.shape[1] else 1
            vocals_np = vocals_np.mean(axis=ch_axis)
        return _resample(vocals_np.astype(np.float32), DEMUCS_SAMPLE_RATE, PIPELINE_SAMPLE_RATE)

    for _ in range(warmup):
        _separate()

    latencies: list[float] = []
    for i in range(runs):
        t0 = time.perf_counter()
        _separate()
        ms = (time.perf_counter() - t0) * 1000
        latencies.append(ms)
        print(f"    run {i + 1}/{runs}: {ms:.0f} ms", flush=True)

    avg_ms = sum(latencies) / len(latencies)
    return {
        "model": model_name,
        "avg_ms": avg_ms,
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "rtf": avg_ms / 1000 / duration,
        "load_sec": load_sec,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark demucs-mlx vocal-isolation latency (Apple GPU).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        choices=["htdemucs", "htdemucs_ft"],
        help="Demucs model variant (htdemucs_ft: 4-model bag, slower)",
    )
    parser.add_argument(
        "--duration", type=float, default=2.0, metavar="SECS",
        help="Synthetic audio duration fed to Demucs each call (at 16 kHz)",
    )
    parser.add_argument(
        "--runs", type=int, default=5, metavar="N",
        help="Timed separation calls",
    )
    parser.add_argument(
        "--warmup", type=int, default=1, metavar="N",
        help="Untimed warm-up calls before measurement",
    )
    args = parser.parse_args()

    for pkg, hint in [
        ("numpy", "pip install numpy"),
        ("mlx", "poetry install --extras separation-mlx"),
        ("demucs_mlx", "poetry install --extras separation-mlx"),
    ]:
        try:
            __import__(pkg)
        except ImportError:
            print(f"Error: {pkg} not found.  {hint}")
            sys.exit(1)

    audio = _synthetic_audio(args.duration)

    print(f"\ndemucs-mlx Source-Separation Benchmark")
    print(f"======================================")
    print(
        f"Model: {args.model}\n"
        f"Audio: {args.duration:.1f}s synthetic @ {PIPELINE_SAMPLE_RATE} Hz  |  "
        f"Runs: {args.runs} timed + {args.warmup} warm-up  |  Device: mlx (Apple GPU)\n"
        f"Note: full pipeline path measured (16 kHz → 44.1 kHz → separate → 16 kHz).\n"
        f"(Weights convert from torch-hub to ~/.cache/demucs-mlx on first run)\n"
    )

    print(f"[{args.model}]")
    try:
        result = _run_model(args.model, audio, args.duration, args.runs, args.warmup)
    except Exception as exc:
        print(f"  FAILED: {exc}")
        sys.exit(1)

    print()
    col = 9
    print(f"{'Avg':>{col}}  {'Min':>{col}}  {'Max':>{col}}  {'RTF':>7}  {'Load':>7}  Note")
    print(f"{'─'*col}  {'─'*col}  {'─'*col}  {'─'*7}  {'─'*7}  {'─'*22}")
    r = result
    rtf = r["rtf"]
    note = "real-time ok" if rtf < 1.0 else "SLOWER than real-time"
    print(
        f"{r['avg_ms']:>{col-2}.0f} ms  {r['min_ms']:>{col-2}.0f} ms  "
        f"{r['max_ms']:>{col-2}.0f} ms  {rtf:>6.3f}x  {r['load_sec']:>5.1f}s  {note}"
    )
    print(
        f"\nRTF = avg_latency / audio_duration.  "
        f"RTF < 1.0 means demucs-mlx keeps up with live audio at "
        f"poll_interval ≥ {args.duration:.1f}s.\n"
    )


if __name__ == "__main__":
    main()
