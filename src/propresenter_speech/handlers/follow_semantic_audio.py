"""
FollowSemanticAudioHandler: MERT-based audio embedding slide matching.

Startup
-------
Builds a per-slide prototype embedding from the reference audio supplied via
--ground-truth JSON (propresenter-train format).  For each slide the handler:
  1. Loads the reference audio file named in the JSON.
  2. Runs MERTPredictor over the slide's [start_sec, stop_sec] window.
  3. Mean-pools the MERT last-hidden-state frames → prototype[i]  shape [D].
  4. Computes global_mean = mean(all prototypes).

Inference
---------
For each live audio window the pipeline calls on_prediction(result, audio_time)
with an AudioEmbeddingResult.  The handler:
  1. Centres the live embedding: query = result.embedding - global_mean.
  2. Centres each prototype: centred_i = prototype[i] - global_mean.
  3. Computes cosine similarity of query vs each centred prototype.
  4. Best match above similarity_threshold (or margin above min_margin) → cue.

Mean-centering removes the shared musical background signal so cosine
similarity focuses on slide-discriminative features.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from propresenter_client.main import ProPresenterController

from ..predictors import AudioEmbeddingResult
from ..predictors.mert import MERTPredictor, _MERT_SAMPLE_RATE

logger = logging.getLogger(__name__)

DEFAULT_SIMILARITY_THRESHOLD = 0.4
DEFAULT_MIN_MARGIN = 0.15


class FollowSemanticAudioHandler:
    """
    Cues whichever slide best matches the live audio via MERT embeddings.
    Prototypes are built from reference audio at startup.
    """

    def __init__(
        self,
        pro_controller: ProPresenterController,
        mert_predictor: MERTPredictor,
        ground_truth_path: str,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        min_margin: float = DEFAULT_MIN_MARGIN,
        verbose: bool = False,
    ):
        self.pro_controller = pro_controller
        self.mert_predictor = mert_predictor
        self.ground_truth_path = ground_truth_path
        self.similarity_threshold = similarity_threshold
        self.min_margin = min_margin
        self.verbose = verbose

        self._prototypes: np.ndarray | None = None   # [N, D] centred prototypes
        self._global_mean: np.ndarray | None = None  # [D]
        self._slide_indices: list[int] = []          # ProPresenter 1-based slide numbers
        self._current_slide_idx: Optional[int] = None

    # ------------------------------------------------------------------
    # ModeHandler protocol
    # ------------------------------------------------------------------

    def on_startup(self) -> None:
        self._build_prototypes()

    def startup_description(self) -> str:
        n = len(self._slide_indices) if self._slide_indices else 0
        return (
            f"Follow-semantic-audio mode active — MERT embeddings, "
            f"{n} slide prototypes, "
            f"threshold={self.similarity_threshold:.2f}, "
            f"min_margin={self.min_margin:.2f}"
        )

    def on_prediction(self, result: AudioEmbeddingResult, _audio_time: float = 0.0) -> None:
        if self._prototypes is None or self._global_mean is None:
            return

        query = result.embedding - self._global_mean
        query_norm = np.linalg.norm(query)
        if query_norm < 1e-8:
            return

        scores = self._prototypes @ query / (
            np.linalg.norm(self._prototypes, axis=1) * query_norm + 1e-8
        )

        best = int(np.argmax(scores))
        confidence = float(scores[best])

        sorted_scores = np.sort(scores)[::-1]
        margin = float(sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) > 1 else confidence

        if self.verbose:
            logger.debug(
                "MERT match: slide_idx=%d  conf=%.3f  margin=%.3f",
                self._slide_indices[best], confidence, margin,
            )

        if confidence < self.similarity_threshold and margin < self.min_margin:
            return
        if self._slide_indices[best] == self._current_slide_idx:
            return

        target = self._slide_indices[best]
        ok = self.pro_controller.go_to_slide(target)
        if ok:
            self._current_slide_idx = target
            print(f"→ Slide {target} (audio conf: {confidence:.2f}, margin: {margin:.2f})")
        else:
            print(f"✗ Failed: go to slide {target}")

    # ------------------------------------------------------------------
    # Prototype building
    # ------------------------------------------------------------------

    def _build_prototypes(self) -> None:
        import json
        import soundfile as sf

        with open(self.ground_truth_path, encoding="utf-8") as f:
            data = json.load(f)

        pres = data["presentation"]
        audio_path = pres["id"]["audio"]
        raw_slides = [s for g in pres["groups"] for s in g["slides"]]
        enabled = [
            s for s in raw_slides
            if s.get("enabled", True) and s.get("text", "").strip()
        ]

        audio_full, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
        if audio_full.ndim > 1:
            audio_full = audio_full.mean(axis=1)
        duration = len(audio_full) / sample_rate
        print(
            f"Building MERT prototypes from: {audio_path}"
            f"  ({duration:.1f}s, {len(enabled)} slides)"
        )

        # Resample the full reference file to 24 kHz once, then slice by time.
        audio_24k = _resample(audio_full, sample_rate, _MERT_SAMPLE_RATE)

        self.mert_predictor.load()

        raw_prototypes: list[np.ndarray] = []
        slide_indices: list[int] = []

        for i, slide in enumerate(enabled):
            start_sec = _slide_start(slide)
            stop_sec = _slide_stop(slide, duration)
            start_frame = int(start_sec * _MERT_SAMPLE_RATE)
            stop_frame = int(stop_sec * _MERT_SAMPLE_RATE)
            chunk = audio_24k[start_frame:stop_frame]

            if len(chunk) < int(0.1 * _MERT_SAMPLE_RATE):
                logger.warning(
                    "Slide %d has very short reference audio (%.2fs) — skipping.",
                    i, stop_sec - start_sec,
                )
                continue

            embedding = self.mert_predictor.embed_24k(chunk)
            raw_prototypes.append(embedding)
            slide_indices.append(i + 1)  # ProPresenter slides are 1-based

        if not raw_prototypes:
            raise RuntimeError(
                "No slide prototypes could be built — check ground-truth JSON and audio path."
            )

        proto_matrix = np.stack(raw_prototypes)       # [N, D]
        global_mean = proto_matrix.mean(axis=0)       # [D]
        self._global_mean = global_mean
        self._prototypes = proto_matrix - global_mean  # [N, D] centred
        self._slide_indices = slide_indices
        print(f"Prototypes ready: {len(slide_indices)} slides, embedding dim={proto_matrix.shape[1]}")


# ------------------------------------------------------------------
# Ground-truth helpers
# ------------------------------------------------------------------

def _slide_start(slide: dict) -> float:
    if "start time" in slide:
        v = slide["start time"]
        return float(v[0] if isinstance(v, list) else v)
    v = slide["trigger time"]
    return float(v[0] if isinstance(v, list) else v)


def _slide_stop(slide: dict, audio_duration: float) -> float:
    if "stop time" in slide:
        v = slide["stop time"]
        t = float(v[0] if isinstance(v, list) else v)
        if t > 0.0:
            return t
    return audio_duration


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Linear-interpolation resample between arbitrary sample rates."""
    if orig_sr == target_sr:
        return audio
    n = int(len(audio) * target_sr / orig_sr)
    return np.interp(
        np.linspace(0, len(audio) - 1, n),
        np.arange(len(audio)),
        audio,
    ).astype(np.float32)
