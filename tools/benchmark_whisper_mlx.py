#!/usr/bin/env python3
"""
Standalone mlx-whisper inference latency benchmark (Apple GPU).

Measures per-call latency and real-time factor (RTF) for Whisper running on the
Apple GPU via MLX, using the same synthetic-audio harness as
tools/benchmark_whisper.py (faster-whisper / CPU) so the two are directly
comparable.  faster-whisper (CTranslate2) has no Metal backend and is CPU-only
on Mac; mlx-whisper runs the same models on the GPU in fp16.

Dependencies: mlx-whisper, numpy  (pip install mlx-whisper)

Usage:
    python tools/benchmark_whisper_mlx.py
    python tools/benchmark_whisper_mlx.py --models tiny base small medium
    python tools/benchmark_whisper_mlx.py --models large-v3-turbo --duration 5.0
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time

# Short name → mlx-community HF repo.  The "-mlx" suffix is consistent across
# sizes; large-v3-turbo is the exception (no suffix).
REPOS = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
}
ALL_MODELS = list(REPOS)
DEFAULT_MODELS = ["tiny", "base", "small", "medium"]
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
    import mlx.core as mx
    import mlx_whisper

    repo = REPOS[model_name]
    # ModelHolder is the fp16 cache transcribe() uses; preloading it here lets us
    # time the load separately and keeps the timed calls pure inference.
    tmod = importlib.import_module("mlx_whisper.transcribe")

    print(f"  loading {repo} …", end=" ", flush=True)
    t0 = time.perf_counter()
    tmod.ModelHolder.get_model(repo, mx.float16)
    load_sec = time.perf_counter() - t0
    print(f"ready in {load_sec:.1f}s")

    def _infer():
        # temperature=0.0 → single decode pass (no fallback schedule), so the
        # number reflects steady-state inference, not noise-driven retries.
        mlx_whisper.transcribe(
            audio, path_or_hf_repo=repo, language="en", temperature=0.0, fp16=True
        )

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
        description="Benchmark mlx-whisper (Apple GPU) inference latency across model variants.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--models", nargs="+", default=DEFAULT_MODELS, choices=ALL_MODELS,
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
        import mlx_whisper  # noqa: F401
    except ImportError:
        print("Error: mlx-whisper not found.  pip install mlx-whisper")
        sys.exit(1)

    audio = _synthetic_audio(args.duration)

    print(f"\nmlx-whisper Inference Benchmark (Apple GPU)")
    print(f"===========================================")
    print(
        f"Audio: {args.duration:.1f}s synthetic  |  "
        f"Runs: {args.runs} timed + {args.warmup} warm-up  |  "
        f"Device: GPU (MLX) / fp16\n"
        f"Compare against: python tools/benchmark_whisper.py  (CPU / int8)\n"
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
    print(f"\n{'Model':<16}  {'Avg':>{col}}  {'Min':>{col}}  {'Max':>{col}}  {'RTF':>7}  {'Load':>7}  Note")
    print(f"{'─'*16}  {'─'*col}  {'─'*col}  {'─'*col}  {'─'*7}  {'─'*7}  {'─'*22}")
    for r in results:
        rtf = r["rtf"]
        note = "real-time ok" if rtf < 1.0 else "SLOWER than real-time"
        print(
            f"{r['model']:<16}  {r['avg_ms']:>{col-2}.0f} ms  "
            f"{r['min_ms']:>{col-2}.0f} ms  {r['max_ms']:>{col-2}.0f} ms  "
            f"{rtf:>6.3f}x  {r['load_sec']:>5.1f}s  {note}"
        )

    best_realtime = [r for r in results if r["rtf"] < 1.0]
    if best_realtime:
        rec = max(best_realtime, key=lambda r: ALL_MODELS.index(r["model"]))
        print(f"\nLargest real-time-capable model: {rec['model']}  "
              f"(RTF {rec['rtf']:.3f}x, avg {rec['avg_ms']:.0f} ms/call)")
    else:
        print("\nNo tested model runs faster than real-time on this hardware.")

    print(
        f"\nRTF = avg_latency / audio_duration.  "
        f"RTF < 1.0 means Whisper keeps up with live audio at poll_interval ≥ {args.duration:.1f}s.\n"
    )


if __name__ == "__main__":
    main()
