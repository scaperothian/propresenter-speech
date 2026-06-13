"""
Backend-resolving factory for source separators.

Shared by the live CLI (`main._build_separator`) and the offline accuracy
evaluator so both resolve `--separation-backend` identically, including the
`auto` preference for demucs-mlx with graceful fallback to torch Demucs.
"""

from __future__ import annotations

import importlib.util
import logging

from .base import SourceSeparator
from .demucs import DemucsSeparator
from .demucs_mlx import MLXDemucsSeparator

logger = logging.getLogger(__name__)


def _mlx_available() -> bool:
    return importlib.util.find_spec("demucs_mlx") is not None


def build_separator(
    backend: str,
    model_name: str,
    device: str = "auto",
    verbose: bool = False,
    log=print,
) -> SourceSeparator:
    """Construct and load a separator for ``backend`` ('auto' | 'demucs' | 'demucs-mlx').

    'auto' prefers demucs-mlx when importable, falling back to torch Demucs if
    the MLX package is absent or fails to load.
    """
    resolved = backend
    if backend == "auto":
        resolved = "demucs-mlx" if _mlx_available() else "demucs"

    if resolved == "demucs-mlx":
        log(f"Loading demucs-mlx '{model_name}' (Apple GPU) — first run may download weights…")
        separator = MLXDemucsSeparator(model_name=model_name, verbose=verbose)
        try:
            separator.load()
        except Exception as exc:
            if backend != "auto":
                raise
            logger.warning("demucs-mlx unavailable (%s); falling back to torch Demucs.", exc)
            resolved = "demucs"
        else:
            log("demucs-mlx ready (device: mlx).")
            return separator

    log(f"Loading Demucs '{model_name}' — this may take a moment on first run…")
    separator = DemucsSeparator(model_name=model_name, device=device, verbose=verbose)
    separator.load()
    log(f"Demucs ready (device: {separator.device}).")
    return separator
