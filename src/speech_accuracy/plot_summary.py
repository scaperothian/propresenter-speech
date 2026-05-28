#!/usr/bin/env python3
"""
Generate a summary bar chart of slide-detection accuracy across models.

Reads accuracy from JSONL log files in a directory.  Each log must contain
a summary record (record_type=summary) with inference_accuracy, model/
embedding_mode, and audio_file fields.

Usage:
    speech-accuracy-plot-summary --logs-dir logs/drive2_rerun_20260527
    speech-accuracy-plot-summary --logs-dir logs/drive2_rerun_20260527 \\
        --extra-logs logs/speech_accuracy_Drive-2_20260525_180142.log \\
                     logs/speech_accuracy_Drive-2_20260525_203343.log \\
        --output summary.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_summary(log_path: Path) -> dict | None:
    with open(log_path) as f:
        for line in f:
            try:
                obj = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            if obj.get("record_type") == "summary":
                return obj
    return None


def model_label(summary: dict) -> str:
    mode = summary.get("embedding_mode", "")
    model = summary.get("model", "")
    if mode == "mert":
        return "MERT-v1-95M"
    if mode == "wav2vec2-alt":
        return "wav2vec2-large ALT"
    if model:
        return f"Whisper {model}"
    return mode or "unknown"


def audio_tag(summary: dict) -> str:
    """Derive a short tag from the audio filename."""
    audio = summary.get("audio_file", "")
    return Path(audio).stem if audio else "unknown"


def collect_results(log_paths: list[Path]) -> list[tuple[str, str, float]]:
    """Return (model_label, audio_tag, accuracy_pct) for every readable log."""
    rows = []
    for p in log_paths:
        s = read_summary(p)
        if s is None:
            print(f"  SKIP (no summary record): {p.name}")
            continue
        acc = round(s["inference_accuracy"] * 100, 1)
        rows.append((model_label(s), audio_tag(s), acc))
    return rows


def plot(
    rows: list[tuple[str, str, float]],
    title: str,
    output: Path,
) -> None:
    # Determine unique models (preserve insertion order) and unique tags.
    models_ordered: list[str] = []
    tags_seen: list[str] = []
    for model, tag, _ in rows:
        if model not in models_ordered:
            models_ordered.append(model)
        if tag not in tags_seen:
            tags_seen.append(tag)

    # Build a lookup {(model, tag): accuracy}.
    lookup: dict[tuple[str, str], float] = {(m, t): a for m, t, a in rows}

    # Colour palette — one per tag.
    palette = ["#4C9BE8", "#E87B4C", "#6DBF67", "#C47FD4", "#F2C94C"]
    tag_colours = {tag: palette[i % len(palette)] for i, tag in enumerate(tags_seen)}

    x = np.arange(len(models_ordered))
    n_tags = len(tags_seen)
    total_width = 0.7
    bar_width = total_width / max(n_tags, 1)
    offsets = np.linspace(-(total_width - bar_width) / 2,
                          (total_width - bar_width) / 2, n_tags)

    fig, ax = plt.subplots(figsize=(max(8, len(models_ordered) * 2.2), 5.5))

    for i, tag in enumerate(tags_seen):
        vals = [lookup.get((m, tag)) for m in models_ordered]
        bars = ax.bar(
            x + offsets[i], vals, bar_width,
            label=tag, color=tag_colours[tag], zorder=3,
        )
        for bar, val in zip(bars, vals):
            if val is None:
                ax.text(bar.get_x() + bar.get_width() / 2, 2, "n/a",
                        ha="center", va="bottom", fontsize=8, color="#888")
            else:
                ax.text(bar.get_x() + bar.get_width() / 2, val + 1.2,
                        f"{val:.1f}%", ha="center", va="bottom",
                        fontsize=8, fontweight="bold")

    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title(title, fontsize=12, pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models_ordered, fontsize=10)
    ax.set_ylim(0, 105)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=10)

    note = ("Whisper: text-embedding match (ASR).  "
            "MERT / wav2vec: audio-prototype similarity (no text).\n"
            "Thresholds — MERT: conf≥0.20 | margin≥0.05;  "
            "wav2vec: conf≥0.30 | margin≥0.08")
    fig.text(0.5, -0.04, note, ha="center", fontsize=8, color="#555")

    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output), dpi=150, bbox_inches="tight")
    print(f"Saved: {output}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot a summary bar chart of accuracy across models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--logs-dir", type=Path, required=True,
        help="Directory containing *.log JSONL files to include",
    )
    parser.add_argument(
        "--extra-logs", nargs="*", type=Path, default=[],
        metavar="LOG",
        help="Additional log files outside --logs-dir (e.g. existing Whisper-base logs)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output PNG path (default: <logs-dir>/summary_accuracy.png)",
    )
    parser.add_argument(
        "--title", default="Slide Detection Accuracy by Model & Audio",
        help="Chart title",
    )
    args = parser.parse_args()

    logs_dir = args.logs_dir.resolve()
    all_logs = sorted(logs_dir.glob("*.log")) + [p.resolve() for p in (args.extra_logs or [])]

    if not all_logs:
        print(f"No .log files found in {logs_dir}")
        return

    rows = collect_results(all_logs)
    if not rows:
        print("No valid summary records found.")
        return

    output = args.output or logs_dir / "summary_accuracy.png"
    plot(rows, args.title, output)


if __name__ == "__main__":
    main()
