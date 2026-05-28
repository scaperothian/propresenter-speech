"""
Wav2VecAltPredictor: wraps the SpeechBrain ALT wav2vec2 checkpoint for lyric ASR.

Loads the fine-tuned wav2vec2 encoder (wav2vec2.ckpt) and the downstream CTC
head (model.ckpt) from a SpeechBrain ALT training run without requiring
SpeechBrain at inference time.

Architecture (from train_wav2vec2_tb.py):
  wav2vec2 encoder → last hidden state + LayerNorm
  → VanillaNN(1024→1024) + LeakyReLU
  → ctc_lin(1024→31)
  → CTC greedy decode

Requires the torch extra: poetry install --extras torch
"""

from __future__ import annotations

import collections
import logging
from pathlib import Path

import numpy as np

from .base import TranscriptionResult
from ..slide_follower import extract_words

logger = logging.getLogger(__name__)

_BASE_MODEL = "facebook/wav2vec2-large-960h-lv60-self"
_HIDDEN_DIM = 1024
_SAMPLE_RATE = 16_000

# Default: sibling wav2vec-alt-experiment directory on this machine.
# Override via --wav2vec-ckpt-dir or ckpt_dir constructor arg.
_DEFAULT_CKPT_DIR = Path(
    "/Users/das/wav2vec-alt-experiment/model/save/downloaded/model/save"
    "/CKPT+2022-05-13+09-25-17+00"
)

# Reconstructed from training config: blank=0, bos=1, eos=2, then space/apostrophe/A-Z.
# The label_encoder.txt was not included in the checkpoint download; this ordering
# matches SpeechBrain's standard English char-level encoder for this model family.
_DEFAULT_VOCAB: list[str] = [
    "<blank>", "<bos>", "<eos>",
    " ", "'",
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
]


class Wav2VecAltPredictor:
    """
    Transcribes audio using the SpeechBrain DALI/ALT fine-tuned wav2vec2 checkpoint.

    The checkpoint was fine-tuned for lyric transcription (Gu & Ou, ISMIR 2022) on
    the DALI v2 dataset, starting from facebook/wav2vec2-large-960h-lv60-self.
    """

    def __init__(
        self,
        ckpt_dir: Path | str | None = None,
        vocab: list[str] | None = None,
        verbose: bool = False,
    ):
        self._ckpt_dir = Path(ckpt_dir) if ckpt_dir else _DEFAULT_CKPT_DIR
        self._vocab = vocab if vocab is not None else _DEFAULT_VOCAB
        self._verbose = verbose
        self._model = None
        self._output_layer_norm = None
        self._enc = None
        self._ctc_head = None
        self._leaky_relu = None
        self._word_buffer: collections.deque[str] = collections.deque(maxlen=200)

    def load(self) -> None:
        try:
            import torch
            import torch.nn as nn
            from transformers import Wav2Vec2Model
        except ImportError:
            raise ImportError("torch extras required: poetry install --extras torch")

        logging.getLogger("transformers").setLevel(logging.WARNING)

        # ── wav2vec2 encoder ──────────────────────────────────────────────────
        logger.info("Loading base wav2vec2 architecture: %s", _BASE_MODEL)
        model = Wav2Vec2Model.from_pretrained(_BASE_MODEL)

        wav2vec_ckpt = self._ckpt_dir / "wav2vec2.ckpt"
        if not wav2vec_ckpt.exists():
            raise FileNotFoundError(
                f"wav2vec2.ckpt not found at {wav2vec_ckpt}\n"
                "Set --wav2vec-ckpt-dir to the SpeechBrain CKPT+... directory."
            )
        logger.info("Loading ALT fine-tuned encoder weights: %s", wav2vec_ckpt)
        raw = torch.load(wav2vec_ckpt, map_location="cpu", weights_only=True)
        sd = {(k[len("model."):] if k.startswith("model.") else k): v for k, v in raw.items()}
        model.load_state_dict(sd, strict=False)
        model.eval()
        self._model = model

        self._output_layer_norm = nn.LayerNorm(_HIDDEN_DIM)
        self._output_layer_norm.eval()

        # ── CTC head (enc + ctc_lin) from model.ckpt ─────────────────────────
        model_ckpt = self._ckpt_dir / "model.ckpt"
        if not model_ckpt.exists():
            raise FileNotFoundError(f"model.ckpt not found at {model_ckpt}")
        logger.info("Loading CTC head weights: %s", model_ckpt)
        head_sd = torch.load(model_ckpt, map_location="cpu", weights_only=True)

        enc = nn.Linear(_HIDDEN_DIM, _HIDDEN_DIM)
        enc.weight = nn.Parameter(head_sd["0.linear.w.weight"])
        enc.bias = nn.Parameter(head_sd["0.linear.w.bias"])
        enc.eval()
        self._enc = enc

        n_classes = head_sd["3.w.weight"].shape[0]
        ctc_head = nn.Linear(_HIDDEN_DIM, n_classes)
        ctc_head.weight = nn.Parameter(head_sd["3.w.weight"])
        ctc_head.bias = nn.Parameter(head_sd["3.w.bias"])
        ctc_head.eval()
        self._ctc_head = ctc_head

        self._leaky_relu = nn.LeakyReLU()
        logger.info("Wav2VecAlt model ready.")

    def predict(self, audio: np.ndarray) -> TranscriptionResult:
        if self._model is None:
            self.load()
        import torch

        # Per-utterance normalisation — matches SpeechBrain's do_normalize=True path
        audio_t = torch.tensor(audio, dtype=torch.float32)
        audio_t = (audio_t - audio_t.mean()) / (audio_t.std() + 1e-7)

        with torch.no_grad():
            out = self._model(audio_t.unsqueeze(0), output_hidden_states=True)
            features = self._output_layer_norm(out.hidden_states[-1])  # [1, T, 1024]
            features = self._leaky_relu(self._enc(features))
            logits = self._ctc_head(features)  # [1, T, n_classes]

        ids = logits.argmax(dim=-1)[0].tolist()

        # CTC greedy decode: collapse repeats, strip blank/bos/eos
        _specials = {"<blank>", "<bos>", "<eos>"}
        chars: list[str] = []
        prev: int | None = None
        for idx in ids:
            if idx != prev:
                token = self._vocab[idx] if idx < len(self._vocab) else ""
                if token and token not in _specials:
                    chars.append(token)
            prev = idx

        text = "".join(chars).strip()
        if text:
            if self._verbose:
                print(f"  heard: {text!r}")
            self._word_buffer.extend(extract_words(text))
        return TranscriptionResult(text=text, word_buffer=self._word_buffer)
