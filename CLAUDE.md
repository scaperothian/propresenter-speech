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
| Mode enum | `src/propresenter_speech/modes.py` | `Mode.PRESENTATION` / `Mode.FOLLOW` |
| Follow-mode slide tracking | `src/propresenter_speech/slide_follower.py` | fetches slide text, extracts trigger words |
| Pipeline orchestration | `src/propresenter_speech/speech_controller.py` | mode-aware dispatch; wires all components |
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
poetry run propresenter-speech                          # presentation mode (default)
poetry run propresenter-speech --mode follow            # follow mode
poetry run propresenter-speech --mode follow --trigger-words 2  # use last 2 words as trigger

# Common flags
poetry run propresenter-speech --model small         # better accuracy
poetry run propresenter-speech --verbose             # print transcriptions + trigger words
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

## Adding new modes

1. Add a new value to the `Mode` enum in `modes.py`.
2. Add the mode choice to `--mode` in `main.py` and wire up any new
   dependencies (e.g. a new controller class analogous to `SlideFollower`).
3. Branch on the new `Mode` value in `SpeechController._handle_segment()`.
4. Add tests in `test_speech_controller.py`.

## Follow mode — how it works

1. On startup `SlideFollower.refresh()` fetches the active presentation from
   `v1/presentation/active` (fallback: `v1/status/slide`) and recursively
   scans the response for text fields.
2. The last N words (default 1, configurable via `--trigger-words`) are stored
   as trigger words after stripping RTF/HTML markup.
3. Each transcribed segment is checked against both the `CommandParser` **and**
   `SlideFollower.matches()`.  An explicit command always takes priority.
4. On a trigger match or explicit command that changes the slide,
   `SlideFollower.refresh()` is called to load the trigger words for the
   new slide.

Note: slide text availability depends on what the ProPresenter Network API
returns for the active presentation.  If text cannot be retrieved, the follower
logs a warning and only explicit commands work until the next successful refresh.

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

## Modes summary

| `--mode` | Behaviour |
|----------|-----------|
| `presentation` (default) | Responds to explicit voice commands only |
| `follow` | Auto-advances on slide trigger words **and** accepts all explicit commands |
