"""
MLXDemucsSeparator: vocal isolation via demucs-mlx, the Apple-Silicon MLX port
of Demucs.  Runs on the Apple GPU, where PyTorch Demucs cannot (the MPS 2^16
conv limit forces the torch path to CPU), giving a much better real-time factor.

Requires the separation-mlx extra: poetry install --extras separation-mlx
(Apple Silicon only).
"""

from __future__ import annotations

import logging
import time

import numpy as np

from ..file_pipeline import _resample

logger = logging.getLogger(__name__)

DEFAULT_DEMUCS_MLX_MODEL = "htdemucs"
_DEMUCS_SAMPLE_RATE = 44_100
_PIPELINE_SAMPLE_RATE = 16_000


class MLXDemucsSeparator:
    """Isolates vocals from a 16 kHz mono chunk using demucs-mlx on the Apple GPU."""

    def __init__(
        self,
        model_name: str = DEFAULT_DEMUCS_MLX_MODEL,
        verbose: bool = False,
    ):
        self._model_name = model_name
        self._verbose = verbose
        self._sep = None
        self._segment_sec = 0.0
        self._split: bool | None = None

    @property
    def device(self) -> str:
        return "mlx"

    def load(self) -> None:
        try:
            from demucs_mlx import Separator
        except ImportError:
            raise ImportError(
                "separation-mlx extras required: poetry install --extras separation-mlx"
            )
        logger.info("Loading demucs-mlx model '%s' (Apple GPU)…", self._model_name)
        self._sep = Separator(model=self._model_name, shifts=1, overlap=0.25)
        self._segment_sec = float(getattr(self._sep.model, "segment", 0.0) or 0.0)
        logger.info("demucs-mlx model ready.")

    def separate(self, audio: np.ndarray) -> np.ndarray:
        if self._sep is None:
            self.load()
        import mlx.core as mx

        started = time.perf_counter()
        n_in = len(audio)
        # Mic capture is 16 kHz, so content above 8 kHz is already gone before
        # Demucs (trained on full-band 44.1 kHz audio) sees it.  Vocal energy
        # sits mostly below 8 kHz, so isolation still works well enough.
        upsampled = _resample(audio, _PIPELINE_SAMPLE_RATE, _DEMUCS_SAMPLE_RATE)
        stereo = np.stack([upsampled, upsampled], axis=0)
        ref_mean = float(stereo.mean())
        ref_std = float(stereo.std()) + 1e-8
        stereo = (stereo - ref_mean) / ref_std

        # split=False is faster but only valid up to the model's training segment
        # (7.8 s for htdemucs); longer windows must be chunked.
        split = self._segment_sec <= 0 or (n_in / _PIPELINE_SAMPLE_RATE) > self._segment_sec
        if split != self._split:
            self._sep.update_parameter(split=split)
            self._split = split

        try:
            _wav, stems = self._sep.separate_tensor(mx.array(stereo), return_mx=True)
            vocals = stems["vocals"]
            mx.eval(vocals)
        except Exception as exc:  # pragma: no cover - hard MLX failure
            raise RuntimeError(f"demucs-mlx separation failed: {exc}") from exc

        vocals_np = np.asarray(vocals, dtype=np.float32) * ref_std + ref_mean
        if vocals_np.ndim == 2:
            ch_axis = 0 if vocals_np.shape[0] < vocals_np.shape[1] else 1
            mono = vocals_np.mean(axis=ch_axis)
        else:
            mono = vocals_np
        out = _resample(mono.astype(np.float32), _DEMUCS_SAMPLE_RATE, _PIPELINE_SAMPLE_RATE)
        if len(out) >= n_in:
            out = out[:n_in]
        else:
            out = np.pad(out, (0, n_in - len(out)))
        if self._verbose:
            print(f"  separation: {(time.perf_counter() - started) * 1000:.0f} ms")
        return out.astype(np.float32)
