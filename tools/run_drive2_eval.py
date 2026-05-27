#!/usr/bin/env python3
"""
Drive-2 accuracy evaluation runner.

Generates accuracy plots and a numeric summary for all models on the
Incubus Drive-2 song (spoken and studio audio versions).

Models evaluated:
  - Whisper base   (existing logs reused — plots only)
  - Whisper tiny   (runs fresh evaluation)
  - MERT-v1-95M    (audio-embedding section detection)
  - wav2vec2-large ALT/DALI  (audio-embedding section detection)

NOTE: Whisper accuracy is measured as % of inference steps where the
predicted slide index matches the ground-truth slide at that timestamp.
MERT / wav2vec-alt accuracy is argmax(section-prototype similarity) vs
ground-truth section label. These are different metrics.

Plots saved to: results/drive2/

Usage (from propresenter-speech directory):
    .venv/bin/python tools/run_drive2_eval.py
    .venv/bin/python tools/run_drive2_eval.py --skip-whisper-tiny
    .venv/bin/python tools/run_drive2_eval.py --skip-mert --skip-wav2vec
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────────────

REPO_ROOT   = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results" / "drive2"

DATASET     = Path("/Users/das/propresenter-dataset-gen/dataset/incubus/drive")
STUDIO_JSON = DATASET / "Drive-2.json"
SPOKEN_JSON = DATASET / "Drive-2_spoken.json"
STUDIO_WAV  = DATASET / "studio_drive.wav"
SPOKEN_WAV  = DATASET / "Drive-2_spoken.wav"

SPEECH_ACCURACY     = REPO_ROOT / ".venv" / "bin" / "speech-accuracy"
SPEECH_ACCURACY_PLT = REPO_ROOT / ".venv" / "bin" / "speech-accuracy-plot"
WHISPER_PAIRWISE    = REPO_ROOT / "tools" / "whisper_pairwise.py"
MERT_PY    = Path("/Users/das/mert-experiment/.venv/bin/python")
WAV2VEC_PY = Path("/Users/das/wav2vec-alt-experiment/.venv/bin/python")

# Existing Whisper-base logs (reuse rather than re-run)
BASE_SPOKEN_LOG = REPO_ROOT / "speech_accuracy_Drive-2_20260525_180142.log"
BASE_STUDIO_LOG = REPO_ROOT / "speech_accuracy_Drive-2_20260525_203343.log"


# ─── Helpers ─────────────────────────────────────────────────────────────────

def run(
    cmd: list[str | Path],
    cwd: Path,
    env: dict | None = None,
    label: str = "",
    capture_stdout: Path | None = None,
) -> int:
    """Run a subprocess, optionally tee-ing stdout to a file for later parsing."""
    print(f"\n{'─' * 60}")
    if label:
        print(f"  {label}")
    print(f"  cwd: {cwd}")
    print(f"  cmd: {' '.join(str(c) for c in cmd)}")
    print(f"{'─' * 60}")
    merged_env = {**os.environ, **(env or {})}

    if capture_stdout:
        lines: list[str] = []
        with subprocess.Popen(
            [str(c) for c in cmd],
            cwd=cwd,
            env=merged_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        ) as proc:
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                lines.append(line)
        rc = proc.returncode
        capture_stdout.write_text("".join(lines))
    else:
        result = subprocess.run(
            [str(c) for c in cmd], cwd=cwd, env=merged_env, check=False
        )
        rc = result.returncode

    if rc != 0:
        print(f"  WARNING: exited {rc}", file=sys.stderr)
    return rc


def read_whisper_summary(log_path: Path) -> dict | None:
    """Return the summary record from a speech-accuracy JSONL log."""
    if not log_path.is_file():
        return None
    with open(log_path) as f:
        for line in f:
            obj = json.loads(line.strip())
            if obj.get("record_type") == "summary":
                return obj
    return None




# ─── Step runners ────────────────────────────────────────────────────────────

def plot_existing_base_logs() -> None:
    """Generate PNGs from the already-complete Whisper-base log files."""
    for log, out_name, label in [
        (BASE_SPOKEN_LOG, "whisper_base_spoken.png", "Whisper base — spoken"),
        (BASE_STUDIO_LOG, "whisper_base_studio.png", "Whisper base — studio"),
    ]:
        if not log.is_file():
            print(f"  SKIP (log not found): {log}")
            continue
        run(
            [SPEECH_ACCURACY_PLT, "--log", str(log),
             "--output", str(RESULTS_DIR / out_name)],
            cwd=REPO_ROOT,
            label=f"Plot {label}",
        )


def run_whisper_pairwise() -> None:
    """Generate text-embedding pairwise plots for both audio versions."""
    for json_path, tag in [
        (SPOKEN_JSON, "spoken"),
        (STUDIO_JSON, "studio"),
    ]:
        out = RESULTS_DIR / f"whisper_pairwise_{tag}.png"
        run(
            [sys.executable, str(WHISPER_PAIRWISE),
             "--ground-truth", str(json_path),
             "--output", str(out)],
            cwd=REPO_ROOT,
            label=f"Whisper pairwise — {tag}",
        )


def run_whisper_tiny(tag: str) -> Path:
    """Run speech-accuracy with Whisper tiny for spoken or studio audio."""
    gt_json  = SPOKEN_JSON if tag == "spoken" else STUDIO_JSON
    log_path = RESULTS_DIR / f"whisper_tiny_{tag}.log"
    png_path = RESULTS_DIR / f"whisper_tiny_{tag}.png"

    run(
        [SPEECH_ACCURACY,
         "--ground-truth", str(gt_json),
         "--model", "tiny",
         "--log-file", str(log_path)],
        cwd=REPO_ROOT,
        label=f"Whisper tiny — {tag}",
    )

    if log_path.is_file():
        run(
            [SPEECH_ACCURACY_PLT,
             "--log", str(log_path),
             "--output", str(png_path)],
            cwd=REPO_ROOT,
            label=f"Plot Whisper tiny — {tag}",
        )
    return log_path


def run_mert(json_path: Path, tag: str) -> Path:
    """Run mert_accuracy evaluator and then render the plot with speech-accuracy-plot."""
    log_path      = RESULTS_DIR / f"mert_{tag}.log"
    png_path      = RESULTS_DIR / f"mert_{tag}.png"
    pairwise_path = RESULTS_DIR / f"mert_{tag}_pairwise.png"
    run(
        [MERT_PY, "tools/mert_accuracy.py",
         "--ground-truth", str(json_path),
         "--log-file", str(log_path),
         "--pairwise-output", str(pairwise_path),
         "--similarity-threshold", "0.20",
         "--min-margin", "0.05"],
        cwd=Path("/Users/das/mert-experiment"),
        env={"MPLBACKEND": "Agg"},
        label=f"MERT — {tag}",
    )
    if log_path.is_file():
        run(
            [SPEECH_ACCURACY_PLT, "--log", str(log_path), "--output", str(png_path)],
            cwd=REPO_ROOT,
            label=f"Plot MERT — {tag}",
        )
    return log_path


def run_wav2vec(json_path: Path, tag: str) -> Path:
    """Run wav2vec_accuracy evaluator and then render the plot with speech-accuracy-plot."""
    log_path      = RESULTS_DIR / f"wav2vec_{tag}.log"
    png_path      = RESULTS_DIR / f"wav2vec_{tag}.png"
    pairwise_path = RESULTS_DIR / f"wav2vec_{tag}_pairwise.png"
    run(
        [WAV2VEC_PY, "tools/wav2vec_accuracy.py",
         "--ground-truth", str(json_path),
         "--log-file", str(log_path),
         "--pairwise-output", str(pairwise_path),
         "--similarity-threshold", "0.30",
         "--min-margin", "0.08"],
        cwd=Path("/Users/das/wav2vec-alt-experiment"),
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

def print_summary(skip_whisper_tiny: bool, skip_mert: bool, skip_wav2vec: bool) -> None:
    """Print a final accuracy table across all evaluated models."""
    sep  = "=" * 70
    thin = "─" * 70
    col  = "─"
    print(f"\n{sep}")
    print("  Drive-2 Accuracy Summary")
    print(sep)
    print("  Whisper: text-based slide match  (ASR → sentence embedding).")
    print("  MERT / wav2vec-alt: audio-embedding section detection (no text).")
    print(thin)

    hdr = f"  {'Model':<28}  {'Audio':<8}  {'Accuracy':>10}  {'Steps':>7}  {'Correct':>7}"
    print(hdr)
    print(f"  {col*28}  {col*8}  {col*10}  {col*7}  {col*7}")

    whisper_entries: list[tuple[str, str, Path]] = [
        ("Whisper base", "spoken", BASE_SPOKEN_LOG),
        ("Whisper base", "studio", BASE_STUDIO_LOG),
    ]
    if not skip_whisper_tiny:
        whisper_entries += [
            ("Whisper tiny", "spoken", RESULTS_DIR / "whisper_tiny_spoken.log"),
            ("Whisper tiny", "studio", RESULTS_DIR / "whisper_tiny_studio.log"),
        ]

    for model_name, audio_tag, log_path in whisper_entries:
        summary = read_whisper_summary(log_path)
        if summary:
            acc     = f"{summary['inference_accuracy'] * 100:.1f}%"
            total   = str(summary["total_steps"])
            correct = str(summary["correct_steps"])
        else:
            acc, total, correct = "n/a", "-", "-"
        print(
            f"  {model_name:<28}  {audio_tag:<8}  {acc:>10}  {total:>7}  {correct:>7}"
        )

    audio_embed_entries: list[tuple[str, str, Path]] = []
    if not skip_mert:
        audio_embed_entries += [
            ("MERT-v1-95M", "spoken", RESULTS_DIR / "mert_spoken.log"),
            ("MERT-v1-95M", "studio", RESULTS_DIR / "mert_studio.log"),
        ]
    if not skip_wav2vec:
        audio_embed_entries += [
            ("wav2vec2-large ALT", "spoken", RESULTS_DIR / "wav2vec_spoken.log"),
            ("wav2vec2-large ALT", "studio", RESULTS_DIR / "wav2vec_studio.log"),
        ]

    if audio_embed_entries:
        print(f"  {col*28}  {col*8}  {col*10}  {col*7}  {col*7}")
        print("  (audio-embedding: argmax prototype similarity vs ground-truth section)")
        for display_name, audio_tag, log_path in audio_embed_entries:
            summary = read_whisper_summary(log_path)
            if summary:
                acc     = f"{summary['inference_accuracy'] * 100:.1f}%"
                total   = str(summary["total_steps"])
                correct = str(summary["correct_steps"])
            else:
                acc, total, correct = "n/a", "-", "-"
            print(
                f"  {display_name:<28}  {audio_tag:<8}  {acc:>10}  {total:>7}  {correct:>7}"
            )

    print()
    print("  Plots saved to:", RESULTS_DIR)
    for p in sorted(RESULTS_DIR.glob("*.png")):
        print(f"    {p.name}")
    print()


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    """Parse args, validate inputs, then run each evaluation in sequence."""
    parser = argparse.ArgumentParser(
        description="Run Drive-2 accuracy evaluations across all models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--skip-whisper-tiny", action="store_true",
                        help="Skip Whisper tiny evaluation (use existing base logs only)")
    parser.add_argument("--skip-mert", action="store_true",
                        help="Skip MERT evaluation")
    parser.add_argument("--skip-wav2vec", action="store_true",
                        help="Skip wav2vec-alt evaluation")
    args = parser.parse_args()

    for path in [STUDIO_JSON, SPOKEN_JSON, STUDIO_WAV, SPOKEN_WAV]:
        if not path.exists():
            print(f"Error: missing input file: {path}")
            sys.exit(1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Results directory: {RESULTS_DIR}")

    plot_existing_base_logs()
    run_whisper_pairwise()

    if not args.skip_whisper_tiny:
        run_whisper_tiny("spoken")
        run_whisper_tiny("studio")

    if not args.skip_mert:
        run_mert(STUDIO_JSON, "studio")
        run_mert(SPOKEN_JSON, "spoken")

    if not args.skip_wav2vec:
        run_wav2vec(STUDIO_JSON, "studio")
        run_wav2vec(SPOKEN_JSON, "spoken")

    print_summary(args.skip_whisper_tiny, args.skip_mert, args.skip_wav2vec)


if __name__ == "__main__":
    main()
