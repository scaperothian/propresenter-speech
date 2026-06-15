"""
DemucsSeparator: isolates the vocals stem from mixed audio via HTDemucs.

Requires the separation extra: poetry install --extras separation
"""

from __future__ import annotations

import logging
import time

import numpy as np

from ..file_pipeline import _resample

logger = logging.getLogger(__name__)

DEFAULT_DEMUCS_MODEL = "htdemucs"
_DEMUCS_SAMPLE_RATE = 44_100
_PIPELINE_SAMPLE_RATE = 16_000


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class DemucsSeparator:
    """Isolates vocals from a 16 kHz mono chunk using a pretrained Demucs model."""

    def __init__(
        self,
        model_name: str = DEFAULT_DEMUCS_MODEL,
        device: str = "auto",
        verbose: bool = False,
    ):
        self._model_name = model_name
        self._device_arg = device
        self._verbose = verbose
        self._model = None
        self._device: str | None = None
        self._vocals_idx: int | None = None
        self._segment_sec = 0.0

    @property
    def device(self) -> str | None:
        return self._device

    def load(self) -> None:
        try:
            import torch  # noqa: F401
            from demucs.apply import apply_model  # noqa: F401
            from demucs.pretrained import get_model
        except ImportError:
            raise ImportError("separation extras required: poetry install --extras separation")
        self._device = _resolve_device(self._device_arg)
        logger.info("Loading Demucs model '%s' on %s…", self._model_name, self._device)
        self._model = get_model(self._model_name)
        self._model.to(self._device)
        self._model.eval()
        self._vocals_idx = self._model.sources.index("vocals")
        self._segment_sec = float(getattr(self._model, "segment", 0.0))
        logger.info("Demucs model ready.")

    def separate(self, audio: np.ndarray) -> np.ndarray:
        if self._model is None:
            self.load()
        import torch
        from demucs.apply import apply_model

        started = time.perf_counter()
        n_in = len(audio)
        # Mic capture is 16 kHz, so content above 8 kHz is already gone before
        # Demucs (trained on full-band 44.1 kHz audio) sees it.  Vocal energy
        # sits mostly below 8 kHz, so isolation still works well enough.
        upsampled = _resample(audio, _PIPELINE_SAMPLE_RATE, _DEMUCS_SAMPLE_RATE)
        wav = torch.from_numpy(upsampled).to(self._device)
        wav = wav[None].expand(2, -1).contiguous()
        ref_mean = wav.mean()
        ref_std = wav.std() + 1e-8
        wav = (wav - ref_mean) / ref_std

        # split=False is ~20% faster but only valid up to the model's training
        # segment (7.8 s for htdemucs); longer windows must be chunked.
        split = self._segment_sec <= 0 or (n_in / _PIPELINE_SAMPLE_RATE) > self._segment_sec
        try:
            with torch.no_grad():
                stems = apply_model(
                    self._model,
                    wav[None],
                    device=self._device,
                    shifts=0,
                    split=split,
                    overlap=0.25,
                    progress=False,
                )[0]
        except RuntimeError:
            if self._device == "cpu":
                raise
            logger.warning(
                "Demucs inference failed on %s; falling back to cpu permanently.",
                self._device,
            )
            self._device = "cpu"
            self._model.to("cpu")
            return self.separate(audio)

        vocals = stems[self._vocals_idx] * ref_std + ref_mean
        mono = vocals.mean(dim=0).cpu().numpy().astype(np.float32)
        out = _resample(mono, _DEMUCS_SAMPLE_RATE, _PIPELINE_SAMPLE_RATE)
        if len(out) >= n_in:
            out = out[:n_in]
        else:
            out = np.pad(out, (0, n_in - len(out)))
        if self._verbose:
            print(f"  separation: {(time.perf_counter() - started) * 1000:.0f} ms")
        return out
