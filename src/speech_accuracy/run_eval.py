#!/usr/bin/env python3
"""
Multi-song accuracy evaluation runner.

Evaluates slide-detection accuracy for one or more ground-truth JSON files
across all configured models (Whisper, MERT, wav2vec-alt).  For each JSON the
audio path is read from the file's "audio" field.

Usage:
    speech-accuracy-run-eval \\
        --ground-truth path/to/Song.json \\
        --results-dir logs/song_results

    speech-accuracy-run-eval \\
        --ground-truth spoken.json studio.json \\
        --results-dir logs/incubus_drive \\
        --whisper-models tiny base \\
        --skip-mert

Ground-truth JSON must be in propresenter-train format (groups/slides with
"trigger time" or "start time"/"stop time").  Output files are named
{model}_{tag}.log / {model}_{tag}.png where tag = the JSON filename stem.

NOTE: Whisper accuracy is text-based (ASR → sentence-embedding match).
MERT / wav2vec-alt accuracy is audio-embedding argmax vs ground-truth
section label.  These are different metrics — not directly comparable.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

SPEECH_ACCURACY     = REPO_ROOT / ".venv" / "bin" / "speech-accuracy"
SPEECH_ACCURACY_PLT = REPO_ROOT / ".venv" / "bin" / "speech-accuracy-plot"
WHISPER_PAIRWISE    = Path(__file__).resolve().parent / "whisper_pairwise.py"
MERT_PY    = Path("/Users/das/mert-experiment/.venv/bin/python")
MERT_TOOL  = Path("/Users/das/mert-experiment/tools/mert_accuracy.py")
WAV2VEC_PY = Path("/Users/das/wav2vec-alt-experiment/.venv/bin/python")
WAV2VEC_TOOL = Path("/Users/das/wav2vec-alt-experiment/tools/wav2vec_accuracy.py")

# Module-level — overwritten by main() after arg parsing.
RESULTS_DIR: Path = REPO_ROOT / "results" / "eval"


# ─── Helpers ─────────────────────────────────────────────────────────────────

def run(
    cmd: list[str | Path],
    cwd: Path,
    env: dict | None = None,
    label: str = "",
) -> int:
    print(f"\n{'─' * 60}")
    if label:
        print(f"  {label}")
    print(f"  cwd: {cwd}")
    print(f"  cmd: {' '.join(str(c) for c in cmd)}")
    print(f"{'─' * 60}")
    merged_env = {**os.environ, **(env or {})}
    result = subprocess.run([str(c) for c in cmd], cwd=cwd, env=merged_env, check=False)
    if result.returncode != 0:
        print(f"  WARNING: exited {result.returncode}", file=sys.stderr)
    return result.returncode


def read_summary(log_path: Path) -> dict | None:
    if not log_path.is_file():
        return None
    with open(log_path) as f:
        for line in f:
            obj = json.loads(line.strip())
            if obj.get("record_type") == "summary":
                return obj
    return None


# ─── Step runners ────────────────────────────────────────────────────────────

def _whisper_suffix(embedding_mode: str, separation: str) -> str:
    suffix = "_ww" if embedding_mode == "word-window" else ""
    if separation == "on":
        suffix += "_sep"
    return suffix


def run_whisper(
    gt_json: Path,
    tag: str,
    model: str,
    embedding_mode: str = "slide",
    separation: str = "off",
    separation_model: str = "htdemucs",
) -> Path:
    suffix = _whisper_suffix(embedding_mode, separation)
    log_path = RESULTS_DIR / f"whisper_{model}{suffix}_{tag}.log"
    png_path = RESULTS_DIR / f"whisper_{model}{suffix}_{tag}.png"
    cmd = [SPEECH_ACCURACY,
           "--ground-truth", str(gt_json),
           "--model", model,
           "--log-file", str(log_path)]
    if embedding_mode != "slide":
        cmd += ["--embedding-mode", embedding_mode]
    if separation == "on":
        cmd += ["--source-separation", "on", "--separation-model", separation_model]
    run(
        cmd,
        cwd=REPO_ROOT,
        label=f"Whisper {model} ({embedding_mode}{', sep' if separation == 'on' else ''}) — {tag}",
    )
    if log_path.is_file():
        run(
            [SPEECH_ACCURACY_PLT, "--log", str(log_path), "--output", str(png_path)],
            cwd=REPO_ROOT,
            label=f"Plot Whisper {model} — {tag}",
        )
    return log_path


def run_whisper_pairwise(gt_json: Path, tag: str) -> None:
    out = RESULTS_DIR / f"whisper_pairwise_{tag}.png"
    run(
        [sys.executable, str(WHISPER_PAIRWISE),
         "--ground-truth", str(gt_json),
         "--output", str(out)],
        cwd=REPO_ROOT,
        label=f"Whisper pairwise — {tag}",
    )


def run_mert(gt_json: Path, tag: str) -> Path:
    log_path      = RESULTS_DIR / f"mert_{tag}.log"
    png_path      = RESULTS_DIR / f"mert_{tag}.png"
    pairwise_path = RESULTS_DIR / f"mert_{tag}_pairwise.png"
    run(
        [MERT_PY, str(MERT_TOOL),
         "--ground-truth", str(gt_json),
         "--log-file", str(log_path),
         "--pairwise-output", str(pairwise_path),
         "--similarity-threshold", "0.20",
         "--min-margin", "0.05"],
        cwd=MERT_PY.parent.parent,
        env={"MPLBACKEND": "Agg", "TRANSFORMERS_OFFLINE": "1"},
        label=f"MERT — {tag}",
    )
    if log_path.is_file():
        run(
            [SPEECH_ACCURACY_PLT, "--log", str(log_path), "--output", str(png_path)],
            cwd=REPO_ROOT,
            label=f"Plot MERT — {tag}",
        )
    return log_path


def run_wav2vec(gt_json: Path, tag: str) -> Path:
    log_path      = RESULTS_DIR / f"wav2vec_{tag}.log"
    png_path      = RESULTS_DIR / f"wav2vec_{tag}.png"
    pairwise_path = RESULTS_DIR / f"wav2vec_{tag}_pairwise.png"
    run(
        [WAV2VEC_PY, str(WAV2VEC_TOOL),
         "--ground-truth", str(gt_json),
         "--log-file", str(log_path),
         "--pairwise-output", str(pairwise_path),
         "--similarity-threshold", "0.30",
         "--min-margin", "0.08"],
        cwd=WAV2VEC_PY.parent.parent,
        env={"MPLBACKEND": "Agg"},
        label=f"wav2vec-alt — {tag}",
    )
    if log_path.is_file():
        run(
            [SPEECH_ACCURACY_PLT, "--log", str(log_path), "--output", str(png_path)],
            cwd=REPO_ROOT,
            label=f"Plot wav2vec-alt — {tag}",
        )
    return log_path


# ─── Summary ─────────────────────────────────────────────────────────────────

def print_summary(
    tags: list[str],
    whisper_models: list[str],
    skip_mert: bool,
    skip_wav2vec: bool,
    embedding_mode: str = "slide",
    separation: str = "off",
) -> None:
    sep  = "=" * 70
    thin = "─" * 70
    col  = "─"
    print(f"\n{sep}")
    print("  Accuracy Summary")
    print(sep)
    print("  Whisper: text-based (ASR → sentence-embedding match).")
    print("  MERT / wav2vec-alt: audio-embedding argmax vs ground-truth section.")
    print(thin)
    print(f"  {'Model':<28}  {'Tag':<20}  {'Accuracy':>10}  {'Steps':>7}  {'Correct':>7}")
    print(f"  {col*28}  {col*20}  {col*10}  {col*7}  {col*7}")

    entries: list[tuple[str, str, Path]] = []
    for model in whisper_models:
        for tag in tags:
            suffix = _whisper_suffix(embedding_mode, separation)
            label = f"Whisper {model} ({embedding_mode}{', sep' if separation == 'on' else ''})"
            entries.append((label, tag, RESULTS_DIR / f"whisper_{model}{suffix}_{tag}.log"))
    if not skip_mert:
        for tag in tags:
            entries.append(("MERT-v1-95M", tag, RESULTS_DIR / f"mert_{tag}.log"))
    if not skip_wav2vec:
        for tag in tags:
            entries.append(("wav2vec2-large ALT", tag, RESULTS_DIR / f"wav2vec_{tag}.log"))

    for model_name, tag, log_path in entries:
        summary = read_summary(log_path)
        if summary:
            acc     = f"{summary['inference_accuracy'] * 100:.1f}%"
            total   = str(summary["total_steps"])
            correct = str(summary["correct_steps"])
        else:
            acc, total, correct = "n/a", "-", "-"
        print(f"  {model_name:<28}  {tag:<20}  {acc:>10}  {total:>7}  {correct:>7}")

    print()
    print("  Plots saved to:", RESULTS_DIR)
    for p in sorted(RESULTS_DIR.glob("*.png")):
        print(f"    {p.name}")
    print()


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run slide-detection accuracy evaluations across all models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ground-truth", required=True, nargs="+", type=Path, metavar="JSON",
        help="One or more ground-truth JSON files (propresenter-train format). "
             "Each filename stem is used as the tag in output filenames.",
    )
    parser.add_argument(
        "--results-dir", type=Path, default=None,
        help="Output directory (default: results/eval_YYYYMMDD_HHMMSS)",
    )
    parser.add_argument(
        "--whisper-models", nargs="+", default=["tiny"],
        metavar="MODEL",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model(s) to evaluate",
    )
    parser.add_argument(
        "--embedding-mode", default="slide", choices=["slide", "word-window"],
        help="Slide embedding mode for Whisper evaluation (default: slide)",
    )
    parser.add_argument(
        "--source-separation", default="off", choices=["on", "off"],
        help="Isolate vocals with Demucs before Whisper transcription",
    )
    parser.add_argument(
        "--separation-model", default="htdemucs", choices=["htdemucs", "htdemucs_ft"],
        help="Demucs model for --source-separation on",
    )
    parser.add_argument("--skip-whisper", action="store_true",
                        help="Skip all Whisper evaluation")
    parser.add_argument("--skip-mert", action="store_true",
                        help="Skip MERT evaluation")
    parser.add_argument("--skip-wav2vec", action="store_true",
                        help="Skip wav2vec-alt evaluation")
    parser.add_argument("--pairwise", action="store_true",
                        help="Generate pairwise slide-embedding plots (requires sentence_transformers)")
    args = parser.parse_args()

    global RESULTS_DIR
    if args.results_dir:
        RESULTS_DIR = args.results_dir.resolve()
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        RESULTS_DIR = REPO_ROOT / "results" / f"eval_{ts}"

    for gt in args.ground_truth:
        if not gt.exists():
            print(f"Error: ground-truth file not found: {gt}")
            sys.exit(1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Results directory: {RESULTS_DIR}")

    tags = [gt.stem for gt in args.ground_truth]

    for gt_json, tag in zip(args.ground_truth, tags):
        if not args.skip_whisper:
            for model in args.whisper_models:
                run_whisper(
                    gt_json, tag, model, args.embedding_mode,
                    separation=args.source_separation,
                    separation_model=args.separation_model,
                )
        if args.pairwise:
            run_whisper_pairwise(gt_json, tag)
        if not args.skip_mert:
            run_mert(gt_json, tag)
        if not args.skip_wav2vec:
            run_wav2vec(gt_json, tag)

    whisper_models = [] if args.skip_whisper else args.whisper_models
    print_summary(
        tags, whisper_models, args.skip_mert, args.skip_wav2vec,
        args.embedding_mode, args.source_separation,
    )


if __name__ == "__main__":
    main()
