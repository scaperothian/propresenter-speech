from collections import deque
from typing import Protocol, runtime_checkable


@runtime_checkable
class ModeHandler(Protocol):
    def on_startup(self) -> None: ...
    def startup_description(self) -> str: ...
    def on_transcription(self, text: str, word_buffer: deque, audio_time: float = 0.0) -> None:
        # audio_time is T_snap: the audio file position (seconds) at the END of the
        # window that was transcribed.  It is computed from frame counts, not from
        # wall clock, so it carries no Whisper processing latency.
        # Handlers that do not need positional information ignore this parameter.
        # In mic mode audio_time is always 0.0 (no file position exists).
        ...
