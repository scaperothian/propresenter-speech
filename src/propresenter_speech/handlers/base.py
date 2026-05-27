from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ModeHandler(Protocol):
    def on_startup(self) -> None: ...
    def startup_description(self) -> str: ...
    def on_prediction(self, result: Any, audio_time: float = 0.0) -> None:
        # audio_time is T_snap: the audio file position (seconds) at the END of
        # the window that was processed.  Derived from frame counts, not wall
        # clock, so it carries no model processing latency.
        # Handlers that do not need positional information ignore this parameter.
        # In mic mode audio_time is always 0.0 (no file position exists).
        ...
