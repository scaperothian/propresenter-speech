#!/usr/bin/env python3
"""
Unified inference-latency benchmark across all supported model backends.

Runs synthetic-audio timing loops for:
  - Whisper (all five sizes via faster-whisper)
  - Wav2Vec2ForCTC (facebook/wav2vec2-large-960h-lv60-self)
  - Wav2Vec2-ALT (SpeechBrain lyric-tuned checkpoint)
  - MERT (m-a-p/MERT-v1-95M)

Each section can be skipped with --skip-* flags.  Results are printed in a
single comparison table at the end.

Usage:
    python tools/benchmark_all.py
    python tools/benchmark_all.py --duration 3.0 --runs 10 --warmup 2
    python tools/benchmark_all.py --skip-wav2vec --skip-wav2vec-alt
    python tools/benchmark_all.py --wav2vec-alt-ckpt /path/to/CKPT+...
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_TOOLS = Path(__file__).resolve().parent
_WHISPER_SR   = 16_000
_WAV2VEC_SR   = 16_000
_MERT_SR      = 24_000
_WHISPER_MODELS = ["tiny", "base", "small", "medium", "large"]
_WAV2VEC_MODEL  = "facebook/wav2vec2-large-960h-lv60-self"
_MERT_MODEL     = "m-a-p/MERT-v1-95M"
_DEFAULT_CKPT   = Path(
    "/Users/das/wav2vec-alt-experiment/model/save/downloaded/model/save"
    "/CKPT+2022-05-13+09-25-17+00"
)


# ─── Audio helpers ────────────────────────────────────────────────────────────

def _synth(duration: float, sample_rate: int) -> "np.ndarray":
    import numpy as np
    rng = np.random.default_rng(42)
    return (rng.standard_normal(int(duration * sample_rate)) * 0.1).astype("float32")


# ─── Per-model runners ────────────────────────────────────────────────────────

def bench_whisper(model_name: str, audio: "np.ndarray", duration: float,
                  runs: int, warmup: int) -> dict:
    from faster_whisper import WhisperModel
    t0 = time.perf_counter()
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    load_sec = time.perf_counter() - t0

    def _infer():
        segs, _ = model.transcribe(audio, language="en", beam_size=1, vad_filter=False)
        list(segs)

    for _ in range(warmup):
        _infer()

    latencies = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _infer()
        latencies.append((time.perf_counter() - t0) * 1000)

    avg_ms = sum(latencies) / len(latencies)
    return {"label": f"Whisper {model_name}", "avg_ms": avg_ms,
            "min_ms": min(latencies), "max_ms": max(latencies),
            "rtf": avg_ms / 1000 / duration, "load_sec": load_sec}


def bench_wav2vec(audio: "np.ndarray", duration: float, runs: int, warmup: int) -> dict:
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    t0 = time.perf_counter()
    processor = Wav2Vec2Processor.from_pretrained(_WAV2VEC_MODEL)
    model = Wav2Vec2ForCTC.from_pretrained(_WAV2VEC_MODEL)
    model.eval()
    load_sec = time.perf_counter() - t0

    inputs = processor(audio, sampling_rate=_WAV2VEC_SR, return_tensors="pt", padding=False)

    def _infer():
        with torch.no_grad():
            logits = model(inputs.input_values).logits
        torch.argmax(logits, dim=-1)

    for _ in range(warmup):
        _infer()

    latencies = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _infer()
        latencies.append((time.perf_counter() - t0) * 1000)

    avg_ms = sum(latencies) / len(latencies)
    return {"label": "Wav2Vec2ForCTC", "avg_ms": avg_ms,
            "min_ms": min(latencies), "max_ms": max(latencies),
            "rtf": avg_ms / 1000 / duration, "load_sec": load_sec}


def bench_wav2vec_alt(ckpt_dir: Path, audio: "np.ndarray", duration: float,
                      runs: int, warmup: int) -> dict:
    import torch
    import torch.nn as nn
    from transformers import Wav2Vec2Model

    _HIDDEN = 1024

    t0 = time.perf_counter()
    wav2vec_ckpt = ckpt_dir / "wav2vec2.ckpt"
    model_ckpt   = ckpt_dir / "model.ckpt"
    for p in (wav2vec_ckpt, model_ckpt):
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")

    model = Wav2Vec2Model.from_pretrained(_WAV2VEC_MODEL)
    raw = torch.load(wav2vec_ckpt, map_location="cpu", weights_only=True)
    sd  = {(k[len("model."):] if k.startswith("model.") else k): v for k, v in raw.items()}
    model.load_state_dict(sd, strict=False)
    model.eval()

    head_sd  = torch.load(model_ckpt, map_location="cpu", weights_only=True)
    enc      = nn.Linear(_HIDDEN, _HIDDEN)
    enc.weight = nn.Parameter(head_sd["0.linear.w.weight"])
    enc.bias   = nn.Parameter(head_sd["0.linear.w.bias"])
    enc.eval()

    n_classes = head_sd["3.w.weight"].shape[0]
    ctc_head  = nn.Linear(_HIDDEN, n_classes)
    ctc_head.weight = nn.Parameter(head_sd["3.w.weight"])
    ctc_head.bias   = nn.Parameter(head_sd["3.w.bias"])
    ctc_head.eval()

    norm = nn.LayerNorm(_HIDDEN)
    norm.eval()
    relu = nn.LeakyReLU()
    load_sec = time.perf_counter() - t0

    audio_t = torch.tensor(audio, dtype=torch.float32)
    audio_t = (audio_t - audio_t.mean()) / (audio_t.std() + 1e-7)
    audio_t = audio_t.unsqueeze(0)

    def _infer():
        with torch.no_grad():
            out      = model(audio_t, output_hidden_states=True)
            features = norm(out.hidden_states[-1])
            features = relu(enc(features))
            ctc_head(features).argmax(dim=-1)

    for _ in range(warmup):
        _infer()

    latencies = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _infer()
        latencies.append((time.perf_counter() - t0) * 1000)

    avg_ms = sum(latencies) / len(latencies)
    return {"label": "Wav2Vec2-ALT", "avg_ms": avg_ms,
            "min_ms": min(latencies), "max_ms": max(latencies),
            "rtf": avg_ms / 1000 / duration, "load_sec": load_sec}


def bench_mert(audio_24k: "np.ndarray", duration: float, runs: int, warmup: int) -> dict:
    import torch
    from transformers import AutoModel, Wav2Vec2FeatureExtractor

    t0 = time.perf_counter()
    processor = Wav2Vec2FeatureExtractor.from_pretrained(_MERT_MODEL, trust_remote_code=True)
    model = AutoModel.from_pretrained(_MERT_MODEL, trust_remote_code=True)
    model.eval()
    load_sec = time.perf_counter() - t0

    inputs = processor(audio_24k, sampling_rate=_MERT_SR, return_tensors="pt")

    def _infer():
        with torch.no_grad():
            outputs = model(**inputs)
        outputs.last_hidden_state[0].mean(dim=0)

    for _ in range(warmup):
        _infer()

    latencies = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _infer()
        latencies.append((time.perf_counter() - t0) * 1000)

    avg_ms = sum(latencies) / len(latencies)
    return {"label": f"MERT {_MERT_MODEL.split('/')[-1]}", "avg_ms": avg_ms,
            "min_ms": min(latencies), "max_ms": max(latencies),
            "rtf": avg_ms / 1000 / duration, "load_sec": load_sec}


# ─── Table printer ────────────────────────────────────────────────────────────

def _print_table(results: list[dict], duration: float) -> None:
    col = 9
    lbl = 28
    sep = "─"
    print()
    print("=" * 72)
    print("  All-Models Benchmark Summary")
    print(f"  Audio: {duration:.1f}s synthetic  |  Device: CPU")
    print("=" * 72)
    print(f"  {'Model':<{lbl}}  {'Avg':>{col}}  {'Min':>{col}}  "
          f"{'Max':>{col}}  {'RTF':>7}  {'Load':>7}  Note")
    print(f"  {sep*lbl}  {sep*col}  {sep*col}  {sep*col}  {sep*7}  {sep*7}  {sep*22}")
    for r in results:
        note = "real-time ok" if r["rtf"] < 1.0 else "SLOWER than real-time"
        print(
            f"  {r['label']:<{lbl}}"
            f"  {r['avg_ms']:>{col-2}.0f} ms"
            f"  {r['min_ms']:>{col-2}.0f} ms"
            f"  {r['max_ms']:>{col-2}.0f} ms"
            f"  {r['rtf']:>6.3f}x"
            f"  {r['load_sec']:>5.1f}s"
            f"  {note}"
        )
    print()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark all model backends and print a unified latency table.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--duration", type=float, default=2.0, metavar="SECS",
                        help="Synthetic audio duration per call")
    parser.add_argument("--runs", type=int, default=5, metavar="N",
                        help="Timed inference calls per model")
    parser.add_argument("--warmup", type=int, default=1, metavar="N",
                        help="Untimed warm-up calls before measurement")
    parser.add_argument("--whisper-models", nargs="+",
                        default=["tiny", "base", "small"],
                        choices=_WHISPER_MODELS, metavar="MODEL",
                        help="Whisper model sizes to benchmark")
    parser.add_argument("--skip-whisper", action="store_true")
    parser.add_argument("--skip-wav2vec", action="store_true")
    parser.add_argument("--skip-wav2vec-alt", action="store_true")
    parser.add_argument("--skip-mert", action="store_true")
    parser.add_argument("--wav2vec-alt-ckpt", type=Path, default=_DEFAULT_CKPT,
                        metavar="PATH",
                        help="Path to SpeechBrain CKPT+... directory")
    args = parser.parse_args()

    # Dependency checks
    missing = []
    if not (args.skip_whisper):
        try:
            import faster_whisper  # noqa: F401
        except ImportError:
            missing.append("faster-whisper  (pip install faster-whisper)")
    for pkg, needed in [("torch", not (args.skip_wav2vec and args.skip_wav2vec_alt and args.skip_mert)),
                        ("transformers", not (args.skip_wav2vec and args.skip_wav2vec_alt and args.skip_mert))]:
        if needed:
            try:
                __import__(pkg)
            except ImportError:
                missing.append(f"{pkg}  (poetry install --extras torch)")
    if missing:
        for m in missing:
            print(f"Error: missing {m}")
        sys.exit(1)

    d, runs, warmup = args.duration, args.runs, args.warmup
    results: list[dict] = []
    errors:  list[str]  = []

    # Whisper
    if not args.skip_whisper:
        audio_16k = _synth(d, _WHISPER_SR)
        for model_name in args.whisper_models:
            print(f"[Whisper {model_name}] loading ...", end=" ", flush=True)
            try:
                r = bench_whisper(model_name, audio_16k, d, runs, warmup)
                print(f"done  avg={r['avg_ms']:.0f}ms  RTF={r['rtf']:.3f}x")
                results.append(r)
            except Exception as exc:
                print(f"FAILED: {exc}")
                errors.append(f"Whisper {model_name}: {exc}")

    # Wav2Vec2ForCTC
    if not args.skip_wav2vec:
        audio_16k = _synth(d, _WAV2VEC_SR)
        print("[Wav2Vec2ForCTC] loading ...", end=" ", flush=True)
        try:
            r = bench_wav2vec(audio_16k, d, runs, warmup)
            print(f"done  avg={r['avg_ms']:.0f}ms  RTF={r['rtf']:.3f}x")
            results.append(r)
        except Exception as exc:
            print(f"FAILED: {exc}")
            errors.append(f"Wav2Vec2ForCTC: {exc}")

    # Wav2Vec2-ALT
    if not args.skip_wav2vec_alt:
        audio_16k = _synth(d, _WAV2VEC_SR)
        print("[Wav2Vec2-ALT] loading ...", end=" ", flush=True)
        try:
            r = bench_wav2vec_alt(args.wav2vec_alt_ckpt, audio_16k, d, runs, warmup)
            print(f"done  avg={r['avg_ms']:.0f}ms  RTF={r['rtf']:.3f}x")
            results.append(r)
        except Exception as exc:
            print(f"FAILED: {exc}")
            errors.append(f"Wav2Vec2-ALT: {exc}")

    # MERT
    if not args.skip_mert:
        audio_24k = _synth(d, _MERT_SR)
        print("[MERT] loading ...", end=" ", flush=True)
        try:
            r = bench_mert(audio_24k, d, runs, warmup)
            print(f"done  avg={r['avg_ms']:.0f}ms  RTF={r['rtf']:.3f}x")
            results.append(r)
        except Exception as exc:
            print(f"FAILED: {exc}")
            errors.append(f"MERT: {exc}")

    if results:
        _print_table(results, d)

    if errors:
        print("Errors:")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
