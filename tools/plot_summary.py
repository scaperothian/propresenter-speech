#!/usr/bin/env python3
"""Generate a summary bar chart of Drive-2 accuracy across all models and audio types."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "drive2"

results = [
    ("Whisper base",        "spoken", 79.9),
    ("Whisper base",        "studio", 50.6),
    ("Whisper tiny",        "spoken", None),
    ("Whisper tiny",        "studio", None),
    ("MERT-v1-95M",         "spoken", 55.6),
    ("MERT-v1-95M",         "studio", 39.3),
    ("wav2vec2-large ALT",  "spoken", 47.2),
    ("wav2vec2-large ALT",  "studio", 31.2),
]

# Load Whisper tiny from logs if present
import json

def read_accuracy(log_path: Path) -> float | None:
    if not log_path.is_file():
        return None
    with open(log_path) as f:
        for line in f:
            obj = json.loads(line.strip())
            if obj.get("record_type") == "summary":
                return round(obj["inference_accuracy"] * 100, 1)
    return None

for i, (model, tag, acc) in enumerate(results):
    if model == "Whisper tiny":
        log = RESULTS_DIR / f"whisper_tiny_{tag}.log"
        loaded = read_accuracy(log)
        results[i] = (model, tag, loaded)

models_ordered = ["Whisper base", "Whisper tiny", "MERT-v1-95M", "wav2vec2-large ALT"]
spoken_vals = []
studio_vals = []

for m in models_ordered:
    s = next((acc for (mod, tag, acc) in results if mod == m and tag == "spoken"), None)
    t = next((acc for (mod, tag, acc) in results if mod == m and tag == "studio"), None)
    spoken_vals.append(s)
    studio_vals.append(t)

x = np.arange(len(models_ordered))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5.5))

spoken_color = "#4C9BE8"
studio_color = "#E87B4C"

bars_spoken = ax.bar(x - width / 2, spoken_vals, width, label="Spoken",
                     color=spoken_color, zorder=3)
bars_studio = ax.bar(x + width / 2, studio_vals, width, label="Studio",
                     color=studio_color, zorder=3)

def label_bars(bars):
    for bar in bars:
        h = bar.get_height()
        if h is None or h == 0:
            ax.text(bar.get_x() + bar.get_width() / 2, 2, "n/a",
                    ha="center", va="bottom", fontsize=9, color="#888")
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1.2,
                    f"{h:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

label_bars(bars_spoken)
label_bars(bars_studio)

ax.set_ylabel("Accuracy (%)", fontsize=11)
ax.set_title("Drive-2  —  Slide Detection Accuracy by Model & Audio Type", fontsize=12, pad=14)
ax.set_xticks(x)
ax.set_xticklabels(models_ordered, fontsize=10)
ax.set_ylim(0, 100)
ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
ax.set_axisbelow(True)
ax.legend(fontsize=10)

note = ("Whisper: text-embedding match (ASR).  "
        "MERT / wav2vec: audio-prototype similarity (no text).\n"
        "Thresholds — MERT: conf≥0.20 | margin≥0.05;  "
        "wav2vec: conf≥0.30 | margin≥0.08")
fig.text(0.5, -0.04, note, ha="center", fontsize=8, color="#555")

plt.tight_layout()
out = RESULTS_DIR / "summary_accuracy.png"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(str(out), dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
