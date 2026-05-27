"""
Offline waveform + accuracy visualiser for speech-accuracy log files.

Usage:
  speech-accuracy-plot --log <path.log>
  speech-accuracy-plot --log <path.log> --output plot.png
  speech-accuracy-plot --log <path.log> --output plot.png --downsample 200
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_log(log_path: Path) -> tuple[list[dict], dict | None]:
    events: list[dict] = []
    summary: dict | None = None
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("record_type") == "summary":
                summary = obj
            else:
                events.append(obj)
    return events, summary


def plot_main() -> None:
    parser = argparse.ArgumentParser(
        prog="speech-accuracy-plot",
        description="Visualise waveform + inference accuracy from a speech-accuracy log file.",
    )
    parser.add_argument("--log", required=True, metavar="PATH", help="JSONL log from speech-accuracy")
    parser.add_argument(
        "--output", default=None, metavar="PATH",
        help="Save to file (PNG/PDF/SVG). Omit to display interactively.",
    )
    parser.add_argument(
        "--downsample", type=int, default=100, metavar="N",
        help="Waveform downsample factor (default: 100 → one sample per ~6 ms at 16 kHz)",
    )
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.is_file():
        print(f"Error: log file not found: {log_path}")
        sys.exit(1)

    events, summary = _load_log(log_path)
    if not summary:
        print("Error: no summary record found. Re-run speech-accuracy to generate a fresh log.")
        sys.exit(1)

    audio_file = summary.get("audio_file", "")
    gt_file = summary.get("ground_truth_file", "")

    if not Path(audio_file).is_file():
        print(f"Error: audio file not found: {audio_file}")
        sys.exit(1)
    if not Path(gt_file).is_file():
        print(f"Error: ground-truth file not found: {gt_file}")
        sys.exit(1)

    if args.output:
        import matplotlib
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib.lines import Line2D
    import numpy as np
    import soundfile as sf
    from speech_accuracy.evaluator import load_ground_truth

    audio, sr = sf.read(audio_file, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    ds = max(1, args.downsample)
    times_ds = np.arange(0, len(audio), ds) / sr
    audio_ds = audio[::ds]

    gt_slides, _, _ = load_ground_truth(gt_file)

    correct_events = [e for e in events if e["is_correct"]]
    wrong_events = [e for e in events if not e["is_correct"]]

    sim_threshold = summary.get("similarity_threshold", 0.4)
    min_margin = summary.get("min_margin", 0.15)
    name = summary.get("presentation", log_path.stem)
    accuracy_pct = summary.get("inference_accuracy", 0.0) * 100
    correct_steps = summary.get("correct_steps", len(correct_events))
    total_steps = summary.get("total_steps", len(events))

    fig, (ax_wave, ax_conf, ax_margin, ax_pred) = plt.subplots(
        4, 1, figsize=(22, 12), sharex=True,
        gridspec_kw={"height_ratios": [1.5, 1, 1, 1]},
    )
    fig.suptitle(
        f"{name}  ·  {correct_steps}/{total_steps} correct  ({accuracy_pct:.1f}%)"
        f"  ·  threshold={sim_threshold}  min_margin={min_margin}",
        fontsize=11,
    )

    # --- Waveform panel ---
    ax_wave.plot(times_ds, audio_ds, color="#cccccc", linewidth=0.3, zorder=1)
    ax_wave.set_ylim(-1.15, 1.15)
    ax_wave.set_ylabel("Amplitude")

    # Draw wrong first so correct overlays on top
    for e in wrong_events:
        ax_wave.axvline(e["audio_time"], color="red", linewidth=0.5, alpha=0.2, zorder=2)
    for e in correct_events:
        ax_wave.axvline(e["audio_time"], color="green", linewidth=0.5, alpha=0.2, zorder=3)

    # Ground-truth slide boundaries
    for slide in gt_slides:
        ax_wave.axvline(slide.start_sec, color="black", linestyle="--", linewidth=1.0, alpha=0.75, zorder=4)
        label = slide.text.replace("\n", " ")[:28]
        ax_wave.text(
            slide.start_sec + 0.5, 1.05, label,
            fontsize=5, color="black", va="top", rotation=90, alpha=0.8, zorder=5,
            clip_on=True,
        )

    ax_wave.legend(
        handles=[
            Line2D([0], [0], color="black", linestyle="--", label="Ground-truth boundary"),
            Line2D([0], [0], color="green", alpha=0.6, label="Correct prediction"),
            Line2D([0], [0], color="red", alpha=0.6, label="Wrong prediction"),
        ],
        loc="upper right", fontsize=7,
    )

    # --- Confidence panel ---
    if events:
        t_conf = [e["audio_time"] for e in events]
        conf = [e["confidence"] for e in events]
        cols = ["green" if e["is_correct"] else "red" for e in events]
        ax_conf.scatter(t_conf, conf, c=cols, s=4, alpha=0.5, linewidths=0, zorder=2)

    ax_conf.axhline(
        sim_threshold, color="#666666", linestyle="--", linewidth=0.8,
        label=f"similarity_threshold={sim_threshold}", zorder=1,
    )
    for slide in gt_slides:
        ax_conf.axvline(slide.start_sec, color="black", linestyle="--", linewidth=1.0, alpha=0.7, zorder=3)

    ax_conf.set_ylim(0, 1.05)
    ax_conf.set_ylabel("Confidence")
    ax_conf.legend(loc="upper right", fontsize=7)
    ax_conf.yaxis.grid(True, color="#e0e0e0", linewidth=0.5, zorder=0)

    # --- Margin panel ---
    if events:
        t_margin = [e["audio_time"] for e in events]
        margins = [e["margin"] for e in events]
        cols_margin = ["green" if e["is_correct"] else "red" for e in events]
        ax_margin.scatter(t_margin, margins, c=cols_margin, s=4, alpha=0.5, linewidths=0, zorder=2)

    ax_margin.axhline(
        min_margin, color="#666666", linestyle="--", linewidth=0.8,
        label=f"min_margin={min_margin}", zorder=1,
    )
    for slide in gt_slides:
        ax_margin.axvline(slide.start_sec, color="black", linestyle="--", linewidth=1.0, alpha=0.7, zorder=3)

    ax_margin.set_ylim(0, 1.05)
    ax_margin.set_ylabel("Margin")
    ax_margin.legend(loc="upper right", fontsize=7)
    ax_margin.yaxis.grid(True, color="#e0e0e0", linewidth=0.5, zorder=0)

    # --- Predicted slide panel ---
    if events:
        t_pred = [e["audio_time"] for e in events]
        pred_idx = [e["pred_slide_idx"] for e in events]
        gt_idx = [e["gt_slide_idx"] for e in events]
        cols_pred = ["green" if e["is_correct"] else "red" for e in events]
        # Ground-truth slide as a grey step line
        ax_pred.step(t_pred, gt_idx, color="black", linewidth=0.8,
                     where="post", zorder=1, label="GT slide")
        ax_pred.scatter(t_pred, pred_idx, c=cols_pred, s=4, alpha=0.6,
                        linewidths=0, zorder=2, label="Pred slide")

    for slide in gt_slides:
        ax_pred.axvline(slide.start_sec, color="black", linestyle="--",
                        linewidth=1.0, alpha=0.7, zorder=3)

    ax_pred.set_ylabel("Slide idx")
    ax_pred.set_xlabel("Time (s)")
    ax_pred.legend(loc="upper right", fontsize=7)
    ax_pred.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax_pred.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax_pred.yaxis.grid(True, color="#e0e0e0", linewidth=0.5, zorder=0)

    plt.tight_layout()

    if args.output:
        out = Path(args.output)
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")
    else:
        try:
            plt.show()
        except KeyboardInterrupt:
            pass
    plt.close("all")
