"""
Mic-mode audio pipeline.

Microphone
    │
    ▼
AudioPipeline      (sounddevice ring buffer, poll every poll_interval s)
    │  audio chunk
    ▼
Predictor.predict(chunk) → result
    │
    ▼
ModeHandler.on_prediction(result, audio_time)

For file-based processing (accuracy evaluation) see file_pipeline.FilePipeline.
"""

import collections
import logging
import threading
import time
from typing import TYPE_CHECKING, Optional

import numpy as np
import sounddevice as sd

if TYPE_CHECKING:
    from .handlers.base import ModeHandler
    from .predictor import Predictor

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
DEFAULT_WINDOW_SECONDS = 2.0
DEFAULT_POLL_INTERVAL = 0.2
COMMAND_COOLDOWN = 1.8


class _BasePipeline:
    """Shared audio-chunk dispatch for mic and file pipelines."""

    def __init__(
        self,
        predictor: "Predictor",
        handler: "ModeHandler",
        window_seconds: float,
        poll_interval: float,
    ):
        self.predictor = predictor
        self.handler = handler
        self.poll_interval = poll_interval
        self._window_frames = int(window_seconds * SAMPLE_RATE)
        self._model_busy = False
        self._running = False

    def _process(self, audio: np.ndarray, audio_time: float = 0.0) -> None:
        self._model_busy = True
        try:
            result = self.predictor.predict(audio)
            if result is None:
                return
            self.handler.on_prediction(result, audio_time)
        finally:
            self._model_busy = False


class AudioPipeline(_BasePipeline):
    """Mic-mode pipeline: sounddevice ring buffer → Predictor → ModeHandler."""

    def __init__(
        self,
        predictor: "Predictor",
        handler: "ModeHandler",
        device: Optional[int] = None,
        window_seconds: float = DEFAULT_WINDOW_SECONDS,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
    ):
        super().__init__(predictor, handler, window_seconds, poll_interval)
        self.device = device
        self._ring: collections.deque = collections.deque(maxlen=self._window_frames)

    def run(self) -> None:
        self.handler.on_startup()
        print(self.handler.startup_description())
        self._running = True
        self._run_mic()

    def _run_mic(self) -> None:
        device_info = sd.query_devices(self.device if self.device is not None else sd.default.device[0])
        print(f"Audio input: [{device_info['index']}] {device_info['name']}")

        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=self.device,
            blocksize=int(SAMPLE_RATE * self.poll_interval),
            callback=self._sd_callback,
        )
        stream.start()

        threading.Thread(
            target=self._poll_loop, daemon=True, name="pipeline-poll"
        ).start()

        print("Press 'q' + Enter or Ctrl-C to stop.\n")
        self._wait_for_stop()

        stream.stop()
        stream.close()

    def _sd_callback(self, indata, _frames, _time_info, status) -> None:
        if status:
            logger.debug("sounddevice status: %s", status)
        self._ring.extend(indata[:, 0].tolist())

    def _poll_loop(self) -> None:
        while self._running:
            time.sleep(self.poll_interval)
            if self._model_busy or len(self._ring) < SAMPLE_RATE * 0.5:
                continue
            audio = np.array(list(self._ring), dtype=np.float32)
            threading.Thread(
                target=self._process,
                args=(audio, 0.0),
                daemon=True,
                name="pipeline-model",
            ).start()

    def _wait_for_stop(self) -> None:
        stop = threading.Event()
        threading.Thread(
            target=self._keyboard_listener, args=(stop,), daemon=True
        ).start()
        try:
            while not stop.is_set():
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            print("\nStopping…")
            self._running = False

    @staticmethod
    def _keyboard_listener(stop: threading.Event) -> None:
        while not stop.is_set():
            try:
                if input().strip().lower() == "q":
                    stop.set()
            except EOFError:
                break


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

def list_input_devices() -> list[dict]:
    devices = sd.query_devices()
    return [
        {"index": i, "name": d["name"], "channels": d["max_input_channels"]}
        for i, d in enumerate(devices)
        if d["max_input_channels"] > 0
    ]


def list_output_devices() -> list[dict]:
    devices = sd.query_devices()
    return [
        {"index": i, "name": d["name"], "channels": d["max_output_channels"]}
        for i, d in enumerate(devices)
        if d["max_output_channels"] > 0
    ]
