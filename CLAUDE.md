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
| Shared audio pipeline | `src/propresenter_speech/audio_pipeline.py` | `AudioPipeline` — ring-buffer mic capture OR file chunking (tqdm progress bar, optional speaker playback); polls Whisper on a timer; calls `ModeHandler.on_transcription()` |
| Mode enum | `src/propresenter_speech/modes.py` | `Mode.PRESENTATION` / `Mode.FOLLOW` / `Mode.FOLLOW_ENHANCED` |
| Mode handler protocol | `src/propresenter_speech/handlers/base.py` | `ModeHandler` Protocol — `on_startup()`, `startup_description()`, `on_transcription()` |
| Presentation mode | `src/propresenter_speech/handlers/presentation.py` | `PresentationHandler` — parses explicit voice commands only |
| Follow mode | `src/propresenter_speech/handlers/follow.py` | `FollowHandler` — explicit commands + trigger-word auto-advance via `SlideFollower`; cooldown prevents double-advance on overlapping windows; explicit commands use `refresh_after_advance`/`refresh_to_slide` to avoid API race condition |
| Follow-enhanced mode | `src/propresenter_speech/handlers/follow_enhanced.py` | `FollowEnhancedHandler` — semantic cosine-similarity match via `SlideEmbedder`; jumps to best-matching slide freely |
| Follow-mode slide tracking | `src/propresenter_speech/slide_follower.py` | fetches slide text, extracts trigger phrase via `--trigger-index` / `--trigger-words`; `refresh_after_advance()` and `refresh_to_slide()` avoid race with API propagation delay |
| Semantic slide index | `src/propresenter_speech/slide_embedder.py` | `SlideEmbedder` — dense cosine similarity over sentence-transformers all-MiniLM-L6-v2 embeddings; `find_slide_with_margin()` returns `(index, score, margin)` |
| CLI entry point | `src/propresenter_speech/main.py` | argparse, builds handler + `AudioPipeline`, calls `.run()` |
| ProPresenter HTTP client | `../propresenter-client/src/propresenter_client/main.py` | imported via path dependency |

## Project conventions

- **Python 3.11+** — use native `list[...]` / `tuple[...]` / `X | Y` type hints.
- **Poetry** for dependency management (`pyproject.toml`); run `poetry install` before anything else.
- **No comments** unless the WHY is non-obvious.  Self-documenting names preferred.
- Tests live in `tests/`; all external I/O is mocked — no network, mic, or GPU needed.

## Running the project

```bash
# Install deps (first time, or after pyproject.toml changes)
poetry install

# Verify ProPresenter is running locally on port 1025, then:
poetry run propresenter-speech                          # presentation mode (default)
poetry run propresenter-speech --mode follow            # follow mode (default: 2-word phrase ending at second-to-last word)
poetry run propresenter-speech --mode follow --trigger-index=-1 --trigger-words=1  # trigger on last word only
poetry run propresenter-speech --mode follow-enhanced          # semantic embedding mode

# Common flags
poetry run propresenter-speech --model small            # better accuracy
poetry run propresenter-speech --verbose                # print transcriptions + match info
poetry run propresenter-speech --list-devices           # show all input + output audio devices
poetry run propresenter-speech --device 2               # use device index 2
poetry run propresenter-speech --host 192.168.1.5       # remote ProPresenter host
poetry run propresenter-speech --window-seconds 3.0     # longer audio context for Whisper
poetry run propresenter-speech --poll-interval 0.1      # faster response
poetry run propresenter-speech --audio-file audio/pledge_of_allegiance.wav            # process file (tqdm progress bar)
poetry run propresenter-speech --audio-file audio/pledge_of_allegiance.wav --playback # process + play through speakers
poetry run propresenter-speech --audio-file audio/sermon.wav --playback --output-device 3  # specific output device
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

`tests/test_transcriber_performance.py` measures the Whisper real-time factor (RTF):
- Loads the real `tiny` and `base` models (no mocking).
- Reports `avg_ms` and `RTF = transcription_time / audio_duration` per model.
- `RTF < 1.0` means the pipeline can keep up in real time; `RTF > 1.0` means lag.
- Run with `poetry run pytest tests/test_transcriber_performance.py -v -s` to see output.

## Adding new voice commands

1. Add regex patterns to the appropriate `_*_PATTERNS` list in
   `command_parser.py` (or create a new `CommandType` variant).
2. Handle the new `CommandType` in the relevant handler's `_execute()` method
   (`PresentationHandler`, `FollowHandler`, or both).
3. Add unit tests in `tests/test_command_parser.py` and the relevant handler test module.

## Adding new modes

1. Add a new value to the `Mode` enum in `modes.py`.
2. Create `src/propresenter_speech/handlers/<mode_name>.py` implementing the `ModeHandler` protocol
   (`on_startup`, `startup_description`, `on_transcription`).
3. Export the new class from `handlers/__init__.py`.
4. Add the mode choice to `--mode` in `main.py` and wire the handler into the dispatch block.
5. Add tests in `tests/test_handler_<mode_name>.py`.

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
3. `--trigger-index` (default `-2`, second-to-last word) sets the anchor position
   within the slide text; `--trigger-words` (default `2`) sets how many consecutive
   words ending at that position form the trigger phrase.  Each transcribed segment is
   checked against both `CommandParser` **and** `SlideFollower.matches()`.  An explicit
   command always takes priority.
4. On a trigger match, `FollowHandler.on_transcription()` loops: it calls
   `next_slide()` then `SlideFollower.refresh_after_advance()` which increments the
   locally cached slide index **without** calling `GET /v1/presentation/slide_index`
   again.  This avoids the race condition where the API still returns the pre-advance
   index immediately after advancing.  The loop continues while trigger words from the
   new slide also appear in the same transcript (so a single audio segment can advance
   through multiple slides).  The loop exits when the end of the presentation is
   reached (triggers cleared) or the next slide's text no longer matches.
   On an explicit command, a plain `SlideFollower.refresh()` is used instead.
   A `COMMAND_COOLDOWN` (imported from `audio_pipeline`) prevents the overlapping
   rolling window from re-triggering the same command or advance.

`ProPresenterController` (in `propresenter-client`) owns all knowledge of the API
response shape: `get_active_presentation_uuid()` and `_extract_uuid()` parse the
UUID; `find_slides()` recursively locates the `"slides"` list; `get_slide_index()`
wraps `GET /v1/presentation/slide_index`.

ProPresenter API reference: `http://<propresenter-ip>:1025/v1/doc/index.html#`

## Follow-enhanced mode — how it works

`FollowEnhancedHandler` runs through the shared `AudioPipeline` like all other modes.

1. **Startup** — `main.py:_build_follow_enhanced_handler()` fetches all slides from the active
   presentation, filters to those with text, calls `SlideEmbedder.build()` to compute dense
   embeddings, then returns the handler.  Whisper is loaded by `main()` before `AudioPipeline.run()`.
2. **Ring buffer** — a `sounddevice.InputStream` fills a `collections.deque` capped at
   `window_seconds × SAMPLE_RATE` frames.  Each audio block is appended in the stream callback.
3. **Poll loop** — a background thread wakes every `poll_interval` seconds.  If Whisper is not
   busy and the buffer holds at least 0.5 s of audio it snapshots the buffer and spawns a
   transcription thread.
4. **Transcription + matching** — `Transcriber.transcribe()` runs on the snapshot.  The resulting
   words are appended to a rolling `word_buffer` (capped at 200 words).  The last `context_words`
   words form the query string for `SlideEmbedder.find_slide_with_margin()`.
5. **Cue logic** — a slide is cued when:
   - `confidence >= similarity_threshold`, **or** `margin >= min_margin`
   - the result is a different slide from the currently cued one
   `ProPresenterController.go_to_slide(slide_idx + 1)` is called directly; explicit voice commands
   are **not** parsed in this mode.

## Audio tuning

If Whisper is triggering on background noise:
- Use a directional or headset microphone.
- Increase `--window-seconds` so Whisper has more context and is less sensitive to brief noise bursts.

If commands lag or are missed:
- Lower `--poll-interval` (e.g. `0.1`) for faster Whisper polling.
- Lower `--window-seconds` (e.g. `1.0`) for a shorter, more focused context window.

## Planned future modes

- **Keyword wake-word** — only activate on "Hey ProPresenter" or similar.
- **Continuous transcript display** — stream a live transcript alongside commands.
- **Web-based Whisper** — swap `Transcriber` backend to a hosted API if on-device
  latency is too high.
- **Presentation name commands** — "open sermon slides", "switch to announcements".

## Modes summary

| `--mode` | Behaviour |
|----------|-----------|
| `presentation` (default) | Responds to explicit voice commands only |
| `follow` | Auto-advances on slide trigger words **and** accepts all explicit commands |
| `follow-enhanced` | Semantic embedding match against all slides; cues whichever slide best matches recent speech; does **not** parse explicit commands |
