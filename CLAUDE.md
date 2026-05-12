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
| Audio capture / VAD | `src/propresenter_speech/audio_capture.py` | `AudioCapture` (mic) and `AudioFileCapture` (WAV/FLAC/OGG); energy-based VAD |
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
poetry run propresenter-speech --audio-file audio/pledge_of_allegiance.wav  # process file instead of mic
```

## Running tests

```bash
poetry run pytest                  # all tests
poetry run pytest tests/test_command_parser.py -v
poetry run pytest -k "Whisper"
```

Most tests are unit-level — no sounddevice, no Whisper model, no ProPresenter
server required.  `faster_whisper.WhisperModel` is patched in `test_transcriber.py`
(patch the source module, not the transcriber module, because the import is deferred
inside `Transcriber.load()`).

`tests/test_integration_follow.py` has two classes:
- `TestFollowModeTriggerOrder` — unit tests for `refresh_after_advance()` trigger
  sequencing; always runs, no external resources.
- `TestFollowModeAudio` — end-to-end test using `audio/pledge_of_allegiance.wav` and
  Whisper `tiny`; auto-skipped when the audio file is absent.

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

1. **Startup validation** — `SlideFollower.validate()` is called before the mic
   starts.  It calls `ProPresenterController.get_active_presentation_uuid()` to
   get the UUID of the current presentation, then `get_presentation_details(uuid)`
   to confirm the response contains at least one slide with a `"text"` field.
   It also caches the UUID and details so the first `refresh()` call reuses them.
   If any step fails the program exits with a clear error message.
2. **Trigger-word extraction** — `SlideFollower.refresh()` drives this flow:
   - `get_active_presentation_uuid()` — resolves the current presentation UUID.
   - `get_presentation_details(uuid)` — fetches full slide data; result is **cached**
     for the class lifetime and only re-fetched when the UUID changes.
   - `get_slide_index()` — `GET /v1/presentation/slide_index?chunked=false` returns
     the zero-based index of the active slide.
   - `find_slides(details)[index]["text"]` — reads the slide text directly.
   - Falls back to `GET /v1/status/slide` + recursive text search on any failure.
3. The last N words of the slide text (default 1, `--trigger-words`) become the
   trigger.  Each transcribed segment is checked against both `CommandParser` **and**
   `SlideFollower.matches()`.  An explicit command always takes priority.
4. On a trigger match, `SpeechController._handle_follow()` loops: it calls
   `next_slide()` then `SlideFollower.refresh_after_advance()` which increments the
   locally cached slide index **without** calling `GET /v1/presentation/slide_index`
   again.  This avoids the race condition where the API still returns the pre-advance
   index immediately after advancing.  The loop continues while trigger words from the
   new slide also appear in the same transcript (so a single audio segment can advance
   through multiple slides).  The loop exits when the end of the presentation is
   reached (triggers cleared) or the next slide's text no longer matches.
   On an explicit command, a plain `SlideFollower.refresh()` is used instead.

`ProPresenterController` (in `propresenter-slides`) owns all knowledge of the API
response shape: `get_active_presentation_uuid()` and `_extract_uuid()` parse the
UUID; `find_slides()` recursively locates the `"slides"` list; `get_slide_index()`
wraps `GET /v1/presentation/slide_index`.

ProPresenter API reference: `http://<propresenter-ip>:1025/v1/doc/index.html#`

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
