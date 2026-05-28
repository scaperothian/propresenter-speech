#!/usr/bin/env python3
"""
Standalone Wav2Vec2-ALT (SpeechBrain lyric-tuned) inference latency benchmark.

Loads the fine-tuned wav2vec2 encoder (wav2vec2.ckpt) and CTC head (model.ckpt)
from the SpeechBrain CKPT+... directory and measures per-call latency using
synthetic audio.  Mirrors the architecture used by Wav2VecAltPredictor.

No propresenter-speech dependencies required; torch and transformers must be
installed (poetry install --extras torch).

Usage:
    python tools/benchmark_wav2vec2_alt.py --ckpt-dir path/to/CKPT+...
    python tools/benchmark_wav2vec2_alt.py --ckpt-dir path/to/CKPT+... --duration 3.0 --runs 10
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

SAMPLE_RATE = 16_000
BASE_MODEL = "facebook/wav2vec2-large-960h-lv60-self"
HIDDEN_DIM = 1024
DEFAULT_CKPT_DIR = Path(
    "/Users/das/wav2vec-alt-experiment/model/save/downloaded/model/save"
    "/CKPT+2022-05-13+09-25-17+00"
)


def _synthetic_audio(duration: float) -> "np.ndarray":
    """White noise at low amplitude — produces real encoder work without silence skip."""
    import numpy as np
    rng = np.random.default_rng(42)
    return (rng.standard_normal(int(duration * SAMPLE_RATE)) * 0.1).astype("float32")


def _load_model(ckpt_dir: Path) -> tuple:
    """Load encoder + CTC head from checkpoint.  Returns (model, enc, ctc_head, norm, relu)."""
    import torch
    import torch.nn as nn
    from transformers import Wav2Vec2Model

    wav2vec_ckpt = ckpt_dir / "wav2vec2.ckpt"
    model_ckpt = ckpt_dir / "model.ckpt"
    for p in (wav2vec_ckpt, model_ckpt):
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")

    print(f"  loading base encoder ({BASE_MODEL}) …", end=" ", flush=True)
    model = Wav2Vec2Model.from_pretrained(BASE_MODEL)
    raw = torch.load(wav2vec_ckpt, map_location="cpu", weights_only=True)
    sd = {(k[len("model."):] if k.startswith("model.") else k): v for k, v in raw.items()}
    model.load_state_dict(sd, strict=False)
    model.eval()

    head_sd = torch.load(model_ckpt, map_location="cpu", weights_only=True)
    enc = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
    enc.weight = nn.Parameter(head_sd["0.linear.w.weight"])
    enc.bias = nn.Parameter(head_sd["0.linear.w.bias"])
    enc.eval()

    n_classes = head_sd["3.w.weight"].shape[0]
    ctc_head = nn.Linear(HIDDEN_DIM, n_classes)
    ctc_head.weight = nn.Parameter(head_sd["3.w.weight"])
    ctc_head.bias = nn.Parameter(head_sd["3.w.bias"])
    ctc_head.eval()

    norm = nn.LayerNorm(HIDDEN_DIM)
    norm.eval()
    relu = nn.LeakyReLU()

    return model, enc, ctc_head, norm, relu


def _run(
    model,
    enc,
    ctc_head,
    norm,
    relu,
    audio: "np.ndarray",
    duration: float,
    runs: int,
    warmup: int,
) -> dict:
    import torch

    audio_t = torch.tensor(audio, dtype=torch.float32)
    audio_t = (audio_t - audio_t.mean()) / (audio_t.std() + 1e-7)
    audio_t = audio_t.unsqueeze(0)

    def _infer():
        with torch.no_grad():
            out = model(audio_t, output_hidden_states=True)
            features = norm(out.hidden_states[-1])
            features = relu(enc(features))
            ctc_head(features).argmax(dim=-1)

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
        "avg_ms": avg_ms,
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "rtf": avg_ms / 1000 / duration,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark Wav2Vec2-ALT (SpeechBrain lyric-tuned) inference latency.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ckpt-dir", default=str(DEFAULT_CKPT_DIR), metavar="PATH",
        help="Path to SpeechBrain CKPT+... directory containing wav2vec2.ckpt and model.ckpt",
    )
    parser.add_argument(
        "--duration", type=float, default=2.0, metavar="SECS",
        help="Synthetic audio duration fed to the model each call",
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

    ckpt_dir = Path(args.ckpt_dir)
    audio = _synthetic_audio(args.duration)

    print(f"\nWav2Vec2-ALT Inference Benchmark")
    print(f"================================")
    print(
        f"Checkpoint: {ckpt_dir}\n"
        f"Audio: {args.duration:.1f}s synthetic @ {SAMPLE_RATE} Hz  |  "
        f"Runs: {args.runs} timed + {args.warmup} warm-up  |  Device: CPU\n"
    )

    print("[wav2vec2-alt]")
    t0 = time.perf_counter()
    try:
        model, enc, ctc_head, norm, relu = _load_model(ckpt_dir)
    except Exception as exc:
        print(f"  FAILED to load: {exc}")
        sys.exit(1)
    load_sec = time.perf_counter() - t0
    print(f"ready in {load_sec:.1f}s")

    try:
        result = _run(model, enc, ctc_head, norm, relu, audio, args.duration, args.runs, args.warmup)
    except Exception as exc:
        print(f"  FAILED: {exc}")
        sys.exit(1)

    result["load_sec"] = load_sec
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
        f"RTF < 1.0 means the model keeps up with live audio at "
        f"poll_interval ≥ {args.duration:.1f}s.\n"
    )


if __name__ == "__main__":
    main()
