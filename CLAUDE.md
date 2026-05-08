# propresenter-speech — Claude context

## What this project does

Real-time voice control for ProPresenter slides using Whisper ASR via
faster-whisper (CTranslate2 backend — no PyTorch, no torch ABI issues).  A
CLI tool captures microphone audio on macOS, runs it through Whisper for
transcription, parses the resulting text for slide commands, and calls the
ProPresenter HTTP API to advance, retreat, or jump to a specific slide.

## Dependency layout

| Concern | Module | Notes |
|---------|--------|-------|
| Whisper transcription | `src/propresenter_speech/transcriber.py` | lazy-loads via faster-whisper; no PyTorch; models from HuggingFace |
| Command parsing | `src/propresenter_speech/command_parser.py` | pure Python, no I/O |
| Audio capture / VAD | `src/propresenter_speech/audio_capture.py` | sounddevice + energy-based VAD |
| Pipeline orchestration | `src/propresenter_speech/speech_controller.py` | wires the three above together |
| CLI entry point | `src/propresenter_speech/main.py` | argparse, calls `SpeechController.run()` |
| ProPresenter HTTP client | `../propresenter-slides/src/propresenter_slides/main.py` | imported via path dependency |

## Project conventions

- **Python 3.10+** — use native `list[...]` / `tuple[...]` / `X | Y` type hints.
- **Poetry** for dependency management (`pyproject.toml`); run `poetry install` before anything else.
- **No comments** unless the WHY is non-obvious.  Self-documenting names preferred.
- Tests live in `tests/`; all external I/O is mocked — no network, mic, or GPU needed.

## Running the project

```bash
# Install deps (first time, or after pyproject.toml changes)
poetry install

# Verify ProPresenter is running locally on port 1025, then:
poetry run propresenter-speech

# Common flags
poetry run propresenter-speech --model small         # better accuracy
poetry run propresenter-speech --verbose             # print transcriptions
poetry run propresenter-speech --list-devices        # show audio input devices
poetry run propresenter-speech --device 2            # use device index 2
poetry run propresenter-speech --host 192.168.1.5    # remote ProPresenter host
```

## Running tests

```bash
poetry run pytest                  # all tests
poetry run pytest tests/test_command_parser.py -v
poetry run pytest -k "Whisper"
```

Tests are entirely unit-level — no sounddevice, no Whisper model, no
ProPresenter server required.  `faster_whisper.WhisperModel` is patched in
`test_transcriber.py` (patch the source module, not the transcriber module,
because the import is deferred inside `Transcriber.load()`).

## Adding new voice commands

1. Add regex patterns to the appropriate `_*_PATTERNS` list in
   `command_parser.py` (or create a new `CommandType` variant).
2. Handle the new `CommandType` in `SpeechController._execute()`.
3. Add unit tests in `tests/test_command_parser.py` and
   `tests/test_speech_controller.py`.

## Audio tuning

If Whisper is triggering on background noise:
- Raise `--silence-threshold` (e.g. `0.02`–`0.05`).
- Increase `--silence-duration` to require a longer pause before a segment is sent.

If commands are being cut off mid-utterance:
- Lower `--silence-duration` slightly (e.g. `0.5`).

## Planned future modes

- **Keyword wake-word** — only activate on "Hey ProPresenter" or similar.
- **Continuous transcription mode** — stream a live transcript alongside commands.
- **Web-based Whisper** — swap `Transcriber` backend to a hosted API if on-device
  latency is too high.
- **Presentation name commands** — "open sermon slides", "switch to announcements".
