#!/usr/bin/env python3
"""
Standalone MERT (m-a-p/MERT-v1-95M) embedding latency benchmark.

Measures per-call latency and real-time factor (RTF) for MERT audio embedding
inference using synthetic audio.  Audio is fed at 24 kHz (MERT's native rate).
In the live pipeline, 16 kHz mic audio is upsampled to 24 kHz before each call.

No propresenter-speech dependencies required; torch and transformers must be
installed (poetry install --extras torch).

Usage:
    python tools/benchmark_mert.py
    python tools/benchmark_mert.py --duration 3.0 --runs 10 --warmup 2
    python tools/benchmark_mert.py --model m-a-p/MERT-v1-330M
"""

from __future__ import annotations

import argparse
import sys
import time

SAMPLE_RATE = 24_000  # MERT native sample rate
DEFAULT_MODEL = "m-a-p/MERT-v1-95M"


def _synthetic_audio(duration: float) -> "np.ndarray":
    """White noise at low amplitude — produces real encoder work without silence skip."""
    import numpy as np
    rng = np.random.default_rng(42)
    return (rng.standard_normal(int(duration * SAMPLE_RATE)) * 0.1).astype("float32")


def _run_model(
    model_name: str,
    audio: "np.ndarray",
    duration: float,
    runs: int,
    warmup: int,
) -> dict:
    import torch
    from transformers import AutoModel, Wav2Vec2FeatureExtractor

    print(f"  loading …", end=" ", flush=True)
    t0 = time.perf_counter()
    processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model.eval()
    load_sec = time.perf_counter() - t0
    print(f"ready in {load_sec:.1f}s")

    inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")

    def _infer():
        with torch.no_grad():
            outputs = model(**inputs)
        outputs.last_hidden_state[0].mean(dim=0)

    for _ in range(warmup):
        _infer()

    latencies: list[float] = []
    for i in range(runs):
        t0 = time.perf_counter()
        _infer()
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
        description="Benchmark MERT audio embedding inference latency.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--duration", type=float, default=2.0, metavar="SECS",
        help="Synthetic audio duration fed to MERT each call (at 24 kHz)",
    )
    parser.add_argument(
        "--runs", type=int, default=5, metavar="N",
        help="Timed inference calls",
    )
    parser.add_argument(
        "--warmup", type=int, default=1, metavar="N",
        help="Untimed warm-up calls before measurement",
    )
    args = parser.parse_args()

    for pkg, hint in [
        ("numpy", "pip install numpy"),
        ("torch", "poetry install --extras torch"),
        ("transformers", "poetry install --extras torch"),
    ]:
        try:
            __import__(pkg)
        except ImportError:
            print(f"Error: {pkg} not found.  {hint}")
            sys.exit(1)

    audio = _synthetic_audio(args.duration)

    print(f"\nMERT Embedding Benchmark")
    print(f"========================")
    print(
        f"Model: {args.model}\n"
        f"Audio: {args.duration:.1f}s synthetic @ {SAMPLE_RATE} Hz (24 kHz native)  |  "
        f"Runs: {args.runs} timed + {args.warmup} warm-up  |  Device: CPU\n"
        f"Note: live pipeline upsamples 16 kHz mic audio to 24 kHz before each call.\n"
        f"(Model downloads automatically from HuggingFace on first run)\n"
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
        f"RTF < 1.0 means MERT keeps up with live audio at "
        f"poll_interval ≥ {args.duration:.1f}s.\n"
    )


if __name__ == "__main__":
    main()
