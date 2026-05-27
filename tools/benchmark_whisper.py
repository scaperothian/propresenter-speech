#!/usr/bin/env python3
"""
Standalone faster-whisper inference latency benchmark.

Measures per-call latency and real-time factor (RTF) for each Whisper model
variant using synthetic audio.  No propresenter-speech dependencies required.

Dependencies: faster-whisper, numpy (both installable via pip)

Usage:
    python tools/benchmark_whisper.py
    python tools/benchmark_whisper.py --models tiny base small
    python tools/benchmark_whisper.py --duration 5.0 --runs 10 --warmup 2
"""

from __future__ import annotations

import argparse
import sys
import time

ALL_MODELS = ["tiny", "base", "small", "medium", "large"]
SAMPLE_RATE = 16_000


def _synthetic_audio(duration: float) -> "np.ndarray":
    """White noise at low amplitude — triggers real encoder work without silence skip."""
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
    from faster_whisper import WhisperModel

    print(f"  loading …", end=" ", flush=True)
    t0 = time.perf_counter()
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    load_sec = time.perf_counter() - t0
    print(f"ready in {load_sec:.1f}s")

    for _ in range(warmup):
        segs, _ = model.transcribe(audio, language="en")
        list(segs)

    latencies: list[float] = []
    for i in range(runs):
        t0 = time.perf_counter()
        segs, _ = model.transcribe(audio, language="en")
        list(segs)
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
        description="Benchmark faster-whisper inference latency across model variants.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS,
        metavar="MODEL", help="Models to benchmark",
    )
    parser.add_argument(
        "--duration", type=float, default=2.0, metavar="SECS",
        help="Synthetic audio duration fed to Whisper each call",
    )
    parser.add_argument(
        "--runs", type=int, default=5, metavar="N",
        help="Timed inference calls per model",
    )
    parser.add_argument(
        "--warmup", type=int, default=1, metavar="N",
        help="Untimed warm-up calls before measurement",
    )
    args = parser.parse_args()

    try:
        import numpy as np  # noqa: F401
    except ImportError:
        print("Error: numpy not found.  pip install numpy")
        sys.exit(1)
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        print("Error: faster-whisper not found.  pip install faster-whisper")
        sys.exit(1)

    audio = _synthetic_audio(args.duration)

    print(f"\nWhisper Inference Benchmark")
    print(f"===========================")
    print(
        f"Audio: {args.duration:.1f}s synthetic  |  "
        f"Runs: {args.runs} timed + {args.warmup} warm-up  |  "
        f"Device: CPU / int8\n"
        f"(Models download automatically from HuggingFace on first run)\n"
    )

    results: list[dict] = []
    for model_name in args.models:
        print(f"[{model_name}]")
        try:
            results.append(_run_model(model_name, audio, args.duration, args.runs, args.warmup))
        except Exception as exc:
            print(f"  FAILED: {exc}")
        print()

    if not results:
        return

    col = 9
    print(f"\n{'Model':<8}  {'Avg':>{col}}  {'Min':>{col}}  {'Max':>{col}}  {'RTF':>7}  {'Load':>7}  Note")
    print(f"{'─'*8}  {'─'*col}  {'─'*col}  {'─'*col}  {'─'*7}  {'─'*7}  {'─'*22}")
    for r in results:
        rtf = r["rtf"]
        note = "real-time ok" if rtf < 1.0 else "SLOWER than real-time"
        print(
            f"{r['model']:<8}  {r['avg_ms']:>{col-2}.0f} ms  "
            f"{r['min_ms']:>{col-2}.0f} ms  {r['max_ms']:>{col-2}.0f} ms  "
            f"{rtf:>6.3f}x  {r['load_sec']:>5.1f}s  {note}"
        )

    best_realtime = [r for r in results if r["rtf"] < 1.0]
    if best_realtime:
        rec = max(best_realtime, key=lambda r: ALL_MODELS.index(r["model"]))
        print(f"\nRecommended for live use: {rec['model']}  (RTF {rec['rtf']:.3f}x, avg {rec['avg_ms']:.0f} ms/call)")
    else:
        print("\nNo tested model runs faster than real-time on this hardware.")

    print(
        f"\nRTF = avg_latency / audio_duration.  "
        f"RTF < 1.0 means Whisper keeps up with live audio at poll_interval ≥ {args.duration:.1f}s.\n"
    )


if __name__ == "__main__":
    main()
