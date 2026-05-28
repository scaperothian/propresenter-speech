#!/usr/bin/env python3
"""
Generate a section×section text-embedding similarity grid for a presentation.

Uses the same sentence-transformer model (all-MiniLM-L6-v2) that the
follow-semantic-words slide embedder uses, applied to each unique slide's full text.
Output is a PNG heatmap in the same style as the MERT/wav2vec pairwise plots.

Usage:
    speech-accuracy-pairwise \
        --ground-truth /path/to/Song.json \
        --output results/drive2/whisper_base_spoken_pairwise.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load_unique_slides(json_path: Path) -> list[tuple[str, str]]:
    """Return [(label, text)] for each unique slide text, in first-appearance order."""
    with open(json_path) as f:
        data = json.load(f)
    pres = data["presentation"]

    raw: list[tuple[float, str]] = []
    for group in pres["groups"]:
        for slide in group["slides"]:
            if not slide.get("enabled", True):
                continue
            text = slide.get("text", "").strip()
            if not text:
                continue
            t = None
            if "start time" in slide:
                starts = slide["start time"]
                t = float(starts[0] if isinstance(starts, list) else starts)
            elif "trigger time" in slide:
                triggers = slide["trigger time"]
                t = float(triggers[0] if isinstance(triggers, list) else triggers)
            if t is not None:
                raw.append((t, text))

    raw.sort(key=lambda e: e[0])

    seen: dict[str, None] = {}
    unique: list[tuple[str, str]] = []
    for _, text in raw:
        if text not in seen:
            seen[text] = None
            label = next(iter(text.splitlines()), "")[:20] or "(unlabeled)"
            unique.append((label, text))
    return unique


def _pairwise_cosine(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / np.maximum(norms, 1e-8)
    return normed @ normed.T


def plot_pairwise(
    M: np.ndarray,
    labels: list[str],
    title: str,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    n = len(labels)
    cell_size = max(1.5, n * 0.7)
    fig, ax = plt.subplots(figsize=(cell_size + 1, cell_size + 1.5))

    im = ax.imshow(M, vmin=-1.0, vmax=1.0, cmap="RdYlGn", aspect="equal")
    for i in range(n):
        for j in range(n):
            val = float(M[i, j])
            color = "white" if abs(val) > 0.7 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold")

    short = [lb[:14] for lb in labels]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(short, fontsize=8)
    plt.colorbar(im, ax=ax, label="Cosine similarity", fraction=0.046, pad=0.04)
    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Pairwise plot saved to {output_path}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Section×section text-embedding similarity grid for a presentation."
    )
    p.add_argument("--ground-truth", required=True, type=Path, metavar="JSON")
    p.add_argument("--output", required=True, type=Path, metavar="FILE")
    p.add_argument("--title", default=None,
                   help="Plot title (default: auto-derived from filename).")
    args = p.parse_args()

    gt_json = args.ground_truth.resolve()
    if not gt_json.is_file():
        p.error(f"Not found: {gt_json}")

    slides = _load_unique_slides(gt_json)
    if not slides:
        p.error("No enabled slides with text found in the JSON.")

    labels = [lb for lb, _ in slides]
    texts  = [tx for _, tx in slides]
    print(f"Building embeddings for {len(texts)} unique slides...")

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, show_progress_bar=False)

    M = _pairwise_cosine(np.array(embeddings))
    title = args.title or f"Section×section similarity (text) — {gt_json.stem}"
    plot_pairwise(M, labels, title, args.output)


if __name__ == "__main__":
    main()
