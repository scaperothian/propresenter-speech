# propresenter-speech — Claude context

## What this project does

Real-time voice control for ProPresenter slides using ASR (Whisper by default,
with optional wav2vec2 and MERT backends).  A CLI tool captures microphone audio
on macOS, runs it through the selected predictor, and calls the ProPresenter
HTTP API to advance, retreat, or jump to a specific slide.

## Dependency layout

| Concern | Module | Notes |
|---------|--------|-------|
| Whisper transcription | `src/propresenter_speech/transcriber.py` | lazy-loads via faster-whisper; no PyTorch; models from HuggingFace |
| Predictor protocol | `src/propresenter_speech/predictors/base.py` | `Predictor` Protocol — `predict(audio) -> Any`; `TranscriptionResult(text, word_buffer)`; `AudioEmbeddingResult(embedding)` |
| Whisper predictor | `src/propresenter_speech/predictors/whisper.py` | `WhisperPredictor` — wraps `Transcriber`, owns the 200-word rolling `_word_buffer`, returns `TranscriptionResult`; always returns a result (even for empty transcriptions) |
| Wav2Vec2 predictor | `src/propresenter_speech/predictors/wav2vec.py` | `Wav2VecPredictor` — HuggingFace `Wav2Vec2ForCTC`; requires torch extra |
| Wav2Vec2-ALT predictor | `src/propresenter_speech/predictors/wav2vec_alt.py` | `Wav2VecAltPredictor` — SpeechBrain lyric-tuned checkpoint loaded without SpeechBrain at inference; requires torch extra |
| MERT predictor | `src/propresenter_speech/predictors/mert.py` | `MERTPredictor` — `m-a-p/MERT-v1-95M`; resamples 16 kHz → 24 kHz internally; returns `AudioEmbeddingResult`; `embed_24k()` is used by `FollowSemanticAudioHandler` for prototype building; requires torch extra |
| Command parsing | `src/propresenter_speech/command_parser.py` | pure Python, no I/O |
| Source-separator protocol | `src/propresenter_speech/separation/base.py` | `SourceSeparator` Protocol — `separate(audio) -> audio`; 16 kHz float32 mono in/out, equal length; decoupled from predictors/handlers so implementations are swappable |
| Demucs separator | `src/propresenter_speech/separation/demucs.py` | `DemucsSeparator` — htdemucs/htdemucs_ft vocal isolation; upsamples 16 kHz → 44.1 kHz, fakes stereo, takes `vocals` stem, downmixes back; auto device (cuda > mps > cpu) with permanent cpu fallback on runtime failure; requires separation extra |
| Mic audio pipeline | `src/propresenter_speech/audio_pipeline.py` | `_BasePipeline` (model-agnostic: `predictor`, optional `separator` applied before `predict()`, `_model_busy`) + `AudioPipeline` — ring-buffer mic capture; polls `Predictor.predict()` on a timer; calls `ModeHandler.on_prediction()` |
| File audio pipeline | `src/propresenter_speech/file_pipeline.py` | `FilePipeline(_BasePipeline)` — sliding-window file processing used by the accuracy evaluator; `_resample` lives here |
| Mode enum | `src/propresenter_speech/modes.py` | `Mode.PRESENTATION` / `Mode.FOLLOW_TRIGGER_WORDS` / `Mode.FOLLOW_SEMANTIC_WORDS` / `Mode.FOLLOW_SEMANTIC_AUDIO` |
| Mode handler protocol | `src/propresenter_speech/handlers/base.py` | `ModeHandler` Protocol — `on_startup()`, `startup_description()`, `on_prediction(result: Any, audio_time: float)` |
| Presentation mode | `src/propresenter_speech/handlers/presentation.py` | `PresentationHandler` — parses explicit voice commands only |
| Follow-trigger-words mode | `src/propresenter_speech/handlers/follow_trigger_words.py` | `FollowTriggerWordsHandler` — explicit commands + trigger-word auto-advance via `SlideFollower`; cooldown prevents double-advance on overlapping windows; explicit commands use `refresh_after_advance`/`refresh_to_slide` to avoid API race condition |
| Follow-semantic-words mode | `src/propresenter_speech/handlers/follow_semantic_words.py` | `FollowSemanticWordsHandler` — semantic cosine-similarity match via `SlideEmbedder`; jumps to best-matching slide freely; does not parse explicit commands |
| Follow-semantic-audio mode | `src/propresenter_speech/handlers/follow_semantic_audio.py` | `FollowSemanticAudioHandler` — builds per-slide MERT prototype embeddings from `--ground-truth` reference audio at startup; matches live audio via mean-centered cosine similarity; does not parse explicit commands |
| Follow-mode slide tracking | `src/propresenter_speech/slide_follower.py` | fetches slide text, extracts trigger phrase via `--trigger-index` / `--trigger-words`; `refresh_after_advance()` and `refresh_to_slide()` avoid race with API propagation delay |
| Semantic slide index | `src/propresenter_speech/slide_embedder.py` | `SlideEmbedder` — one embedding per slide; `WordWindowEmbedder` — one embedding per word position (configurable stride), slides passed as `(slide_idx, text)` pairs in chronological order so repeated sections resolve correctly; both expose `find_slide_with_margin()` and `avg_words_per_slide` |
| CLI entry point | `src/propresenter_speech/main.py` | argparse, builds predictor + handler + `AudioPipeline`, calls `.run()`; `_build_predictor()` factory for text-based modes; `_build_separator()` — `--source-separation auto` enables Demucs only in follow-semantic-words mode; audio mode builds `MERTPredictor` directly and reuses it as both handler and pipeline predictor |
| Accuracy evaluator | `src/speech_accuracy/evaluator.py` | `load_ground_truth()`, `AccuracyHandler`, `AccuracyEvaluator`, `EvaluationResult`; feeds audio through `FilePipeline` + `WhisperPredictor` with `AccuracyHandler` in place of the normal slide-cue handler; mirrors `FollowSemanticWordsHandler` gate (confidence OR margin threshold) via `_current_pred_idx`; scores each inference call against propresenter-train JSON ground truth using T_snap timing |
| Accuracy CLI | `src/speech_accuracy/main.py` | `speech-accuracy` (single file) and `speech-accuracy-batch` (batch) entry points; JSONL event logging includes `audio_file`, `ground_truth_file`, `embedding_mode`, `source_separation`, `similarity_threshold`, `min_margin`; `--source-separation on` (default off so baselines stay comparable) isolates vocals before transcription |
| Accuracy plot | `src/speech_accuracy/plot.py` | `speech-accuracy-plot` — offline 4-panel matplotlib visualiser: waveform, confidence, margin, predicted slide index; reads JSONL log + audio file + ground-truth JSON |
| Multi-model eval runner | `src/speech_accuracy/run_eval.py` | `speech-accuracy-run-eval` CLI entry point; runs Whisper, MERT, and wav2vec-alt evaluations against one or more ground-truth JSON files; `--ground-truth` accepts any propresenter-train JSON; tags derived from filename stems; `--whisper-models` selects model sizes; `TRANSFORMERS_OFFLINE=1` set for MERT subprocess |
| Pairwise similarity chart | `src/speech_accuracy/whisper_pairwise.py` | `speech-accuracy-pairwise` CLI entry point; section×section text-embedding similarity grid using `all-MiniLM-L6-v2`; outputs PNG heatmap |
| Summary chart | `src/speech_accuracy/plot_summary.py` | `speech-accuracy-plot-summary` CLI entry point; reads all `*.log` files in `--logs-dir`, auto-detects model/tag from summary records, plots grouped bar chart; accepts `--extra-logs` for logs outside the main dir |
| Whisper benchmark | `tools/benchmark_whisper.py` | standalone script (no propresenter-speech imports); measures per-call latency and RTF for all Whisper model variants using synthetic audio; recommends largest real-time-capable model |
| Wav2Vec2 benchmark | `tools/benchmark_wav2vec2.py` | standalone; `facebook/wav2vec2-large-960h-lv60-self`; measures per-call latency and RTF |
| Wav2Vec2-ALT benchmark | `tools/benchmark_wav2vec2_alt.py` | standalone; loads SpeechBrain CKPT+... checkpoint; `--ckpt-dir` selects path |
| MERT benchmark | `tools/benchmark_mert.py` | standalone; `m-a-p/MERT-v1-95M`; 24 kHz synthetic audio; `--model` selects variant |
| Demucs benchmark | `tools/benchmark_demucs.py` | standalone; measures the full vocal-isolation path (16 kHz → 44.1 kHz → separate → 16 kHz); `--model` / `--device` selectable |
| All-models benchmark | `tools/benchmark_all.py` | standalone rollup; runs Whisper + Wav2Vec2 + Wav2Vec2-ALT + MERT + Demucs in one pass; prints unified comparison table; each backend skippable with `--skip-*` |
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

# Install torch-dependent backends (wav2vec2, wav2vec2-alt, MERT)
poetry install --extras torch

# Install Demucs source separation (vocal isolation)
poetry install --extras separation

# Verify ProPresenter is running locally on port 1025, then:
poetry run propresenter-speech                                   # presentation mode (default)
poetry run propresenter-speech --mode follow-trigger-words       # trigger-word auto-advance
poetry run propresenter-speech --mode follow-trigger-words --trigger-index=-1 --trigger-words=1
poetry run propresenter-speech --mode follow-semantic-words      # semantic text embedding (slide embedder)
poetry run propresenter-speech --mode follow-semantic-words --embedding-mode word-window
poetry run propresenter-speech --mode follow-semantic-words --embedding-mode word-window --embedding-stride 2
poetry run propresenter-speech --mode follow-semantic-audio \
  --ground-truth ../propresenter-train/output/Song.json          # MERT audio embedding

# ASR backend (default: whisper)
poetry run propresenter-speech --asr-backend wav2vec2
poetry run propresenter-speech --asr-backend wav2vec2-alt

# Source separation (--source-separation auto is the default: Demucs vocal
# isolation runs only in follow-semantic-words mode; on/off force it anywhere)
poetry run propresenter-speech --mode follow-semantic-words                          # separation auto-on
poetry run propresenter-speech --mode follow-semantic-words --source-separation off
poetry run propresenter-speech --mode presentation --source-separation on
poetry run propresenter-speech --source-separation on --separation-model htdemucs_ft --separation-device cpu

# Common flags
poetry run propresenter-speech --model small            # better Whisper accuracy
poetry run propresenter-speech --verbose                # print transcriptions + match info
poetry run propresenter-speech --list-devices           # show all input + output audio devices
poetry run propresenter-speech --device 2               # use device index 2
poetry run propresenter-speech --host 192.168.1.5       # remote ProPresenter host
poetry run propresenter-speech --window-seconds 3.0     # longer audio context for Whisper
poetry run propresenter-speech --poll-interval 0.1      # faster response
```

## Accuracy evaluation tools

`speech-accuracy` and `speech-accuracy-batch` are CLI entry points in the `speech_accuracy`
package (`src/speech_accuracy/`) that measure follow-semantic-words matching accuracy offline.
`speech_accuracy` is a separate package within the same Poetry project — it imports from
`propresenter_speech` but `propresenter_speech` has no dependency on it.

```bash
# Single presentation
poetry run speech-accuracy \
  --ground-truth ../propresenter-train/output/"Mary Had A Little Lamb.json" \
  --model base --verbose

# All presentations in a directory
poetry run speech-accuracy-batch \
  --ground-truth-dir ../propresenter-train/output/ --model base

# A/B the Demucs vocal-isolation impact (default off so baselines stay comparable;
# the JSONL summary records "source_separation": "htdemucs" | "htdemucs_ft" | "off")
poetry run speech-accuracy \
  --ground-truth ../propresenter-train/output/"Mary Had A Little Lamb.json" \
  --model base --source-separation on
```

**Ground-truth JSON** lives in `../propresenter-train/output/`.  Two timing formats are
normalised automatically: `"start time"`/`"stop time"` and `"trigger time"` (stop derived
from next slide's start).  Slides with `"enabled": false` are excluded.

**Evaluation strategy** — `AccuracyEvaluator` feeds the audio file through `FilePipeline`
with `WhisperPredictor` and `AccuracyHandler` replacing the normal slide-cue handler.
`FilePipeline` uses a sliding window identical to mic mode: advances by `poll_interval` each
step, always transcribes the trailing `window_seconds` of audio (e.g. ~900 steps for a
3-minute file at `poll_interval=0.2s`).  `audio_time` passed to `on_prediction()` is T_snap
— the audio file position (seconds) at the END of the transcribed window, derived from frame
counts with no wall-clock jitter.  `AccuracyHandler` mirrors `FollowSemanticWordsHandler`'s gate:
the predicted slide only updates when `confidence >= similarity_threshold OR margin >=
min_margin`; otherwise the last accepted prediction is held.

**Output** — per-inference JSONL log (one object per step + a summary record at the end)
plus a printed accuracy table.  See `src/speech_accuracy/speech-accuracy.md` for the full reference.

```bash
# Visualise a log file (interactive)
poetry run speech-accuracy-plot --log speech_accuracy_MyPresentation_20260525_120000.log

# Save to PNG
poetry run speech-accuracy-plot --log results.log --output plot.png
```

**Multi-model evaluation runner** — `tools/run_eval.py` runs Whisper, MERT, and wav2vec-alt
evaluations against any number of ground-truth JSON files in a single pass and prints a
combined accuracy table.

```bash
# Evaluate two audio versions of the same song
poetry run speech-accuracy-run-eval \
  --ground-truth path/to/Song_spoken.json path/to/Song_studio.json \
  --results-dir logs/song_results \
  --whisper-models tiny base

# Skip MERT/wav2vec, evaluate Whisper only
poetry run speech-accuracy-run-eval \
  --ground-truth path/to/Song.json \
  --skip-mert --skip-wav2vec
```

**Summary chart** — `speech-accuracy-plot-summary` reads all `.log` files in a directory and
generates a grouped bar chart (one bar group per model, one bar per audio tag).

```bash
poetry run speech-accuracy-plot-summary --logs-dir logs/song_results

# Include existing logs from outside the dir (e.g. a previously run Whisper base)
poetry run speech-accuracy-plot-summary \
  --logs-dir logs/song_results \
  --extra-logs logs/speech_accuracy_Song_20260525_180142.log
```

**Pairwise similarity** — `speech-accuracy-pairwise` generates a section×section
text-embedding heatmap for a single presentation.

```bash
poetry run speech-accuracy-pairwise \
  --ground-truth path/to/Song.json \
  --output results/song_pairwise.png
```

**Benchmarks** — standalone scripts in `tools/` (no Poetry install needed beyond deps):

```bash
python tools/benchmark_whisper.py                        # all Whisper sizes
python tools/benchmark_wav2vec2.py                       # Wav2Vec2ForCTC
python tools/benchmark_wav2vec2_alt.py --ckpt-dir PATH   # Wav2Vec2-ALT
python tools/benchmark_mert.py                           # MERT-v1-95M
python tools/benchmark_demucs.py                         # Demucs vocal isolation
python tools/benchmark_demucs.py --model htdemucs_ft --device cpu

# Unified comparison table across all backends
python tools/benchmark_all.py
python tools/benchmark_all.py --duration 3.0 --runs 10
python tools/benchmark_all.py --skip-wav2vec --skip-wav2vec-alt
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
   (`PresentationHandler`, `FollowTriggerWordsHandler`, or both).
3. Add unit tests in `tests/test_command_parser.py` and the relevant handler test module.

## Adding new modes

1. Add a new value to the `Mode` enum in `modes.py`.
2. Create `src/propresenter_speech/handlers/<mode_name>.py` implementing the `ModeHandler` protocol
   (`on_startup`, `startup_description`, `on_prediction`).
3. Export the new class from `handlers/__init__.py`.
4. Add the mode choice to `--mode` in `main.py` and wire the handler into the dispatch block.
5. Add tests in `tests/test_handler_<mode_name>.py`.

## Follow-trigger-words mode — how it works

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
4. On a trigger match, `FollowTriggerWordsHandler.on_prediction()` loops: it calls
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

## Follow-semantic-words mode — how it works

`FollowSemanticWordsHandler` runs through the shared `AudioPipeline` like all other modes.

1. **Startup** — `main.py:_build_follow_semantic_words_handler()` fetches all slides from the active
   presentation, filters to those with text, calls `SlideEmbedder.build()` to compute dense
   embeddings, then auto-computes `context_words` as the average word count per slide
   (via `SlideEmbedder.avg_words_per_slide`) unless `--context-words` was explicitly passed.
   This means the query n-gram naturally mirrors how much text is on a typical slide.
2. **Ring buffer** — a `sounddevice.InputStream` fills a `collections.deque` capped at
   `window_seconds × SAMPLE_RATE` frames.  Each audio block is appended in the stream callback.
3. **Poll loop** — a background thread wakes every `poll_interval` seconds.  If the predictor is not
   busy and the buffer holds at least 0.5 s of audio it snapshots the buffer and spawns a
   transcription thread.
4. **Transcription + matching** — `WhisperPredictor.predict()` runs on the snapshot.  New
   words are appended to its internal 200-word rolling `_word_buffer`.  The result
   (`TranscriptionResult`) is passed to `FollowSemanticWordsHandler.on_prediction()`, which takes
   the last `context_words` words from `result.word_buffer` as the query string for
   `SlideEmbedder.find_slide_with_margin()`.
5. **Cue logic** — a slide is cued when:
   - `confidence >= similarity_threshold`, **or** `margin >= min_margin`
   - the result is a different slide from the currently cued one
   `ProPresenterController.go_to_slide(slide_idx + 1)` is called directly; explicit voice commands
   are **not** parsed in this mode.

## Follow-semantic-audio mode — how it works

`FollowSemanticAudioHandler` uses MERT audio embeddings instead of text.

1. **Startup** — `MERTPredictor` is loaded first (`m-a-p/MERT-v1-95M`; ~95 M params; requires
   torch extra).  `FollowSemanticAudioHandler.on_startup()` then loads the `--ground-truth` JSON,
   reads the reference audio file, resamples to 24 kHz, and for each enabled slide:
   - slices the audio by `[start_sec, stop_sec]`
   - calls `MERTPredictor.embed_24k(chunk)` — mean-pools the last hidden state → `[D]` prototype
   The same `MERTPredictor` instance serves as both the pipeline predictor and the prototype builder
   (no double-loading of weights).
2. **Mean centering** — after all prototypes are built, `global_mean = mean(all prototypes)` is
   computed.  Each prototype and every live query is centred by subtracting `global_mean` before
   cosine similarity.  This suppresses the shared musical background signal (drums, keys, etc.)
   so matching focuses on the slide-discriminative features.
3. **Live inference** — `AudioPipeline` feeds 16 kHz mic audio to `MERTPredictor.predict()`,
   which resamples to 24 kHz internally and returns `AudioEmbeddingResult(embedding)`.
   `FollowSemanticAudioHandler.on_prediction()` centres the embedding and computes cosine
   similarity against all centred prototypes.  Best match above threshold → `go_to_slide()`.
4. **Cue logic** — same gate as follow-semantic-words: `confidence >= similarity_threshold` **or**
   `margin >= min_margin`; skips if the best match is the currently cued slide.

## Source separation — how it works

`DemucsSeparator` is an optional preprocessing stage applied by `_BasePipeline._process()`
to each audio window **before** `Predictor.predict()`.  It is fully decoupled from
predictors, handlers, and embedders — anything implementing the `SourceSeparator`
protocol (`separate(audio) -> audio`) can replace it.

1. **Enablement** — `--source-separation auto` (default) turns it on only in
   `follow-semantic-words` mode; `on`/`off` force it for any mode.  `main.py:_build_separator()`
   loads the model up front so the first mic window isn't delayed.
2. **Per-window isolation** — each 16 kHz window is upsampled to 44.1 kHz (Demucs's native
   rate), duplicated mono → stereo, normalised, run through `demucs.apply.apply_model`,
   and the `vocals` stem is denormalised, downmixed to mono, and resampled back to 16 kHz
   at the original length.  Mic capture is 16 kHz so content above 8 kHz is lost before
   Demucs sees it; vocal energy sits mostly below 8 kHz, so isolation still works.
3. **Latency behaviour** — separation runs inside the `_model_busy`-guarded worker thread,
   so a slow window causes skipped polls (fewer updates per second), never a backlog.
   Windows shorter than the model's training segment (7.8 s for htdemucs) use `split=False`,
   which roughly halves latency.  Measured on an Apple-Silicon CPU (torch 2.7.0): ~3.2 s per
   2 s window (RTF ≈ 1.6) — live mode yields a slide-match update every ~3–4 s rather than
   every poll.  `htdemucs_ft` (a 4-model bag, ~4× slower) is realistically for offline
   evaluation only.  Device resolution is cuda > mps > cpu; **htdemucs does not run on MPS**
   (`Output channels > 65536 not supported`, torch 2.7) — the separator catches the failure,
   logs a warning, and falls back to cpu permanently.
4. **First run** — htdemucs (~80 MB) / htdemucs_ft (~320 MB) download from
   dl.fbaipublicfiles.com into the torch-hub cache.

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
| `follow-trigger-words` | Auto-advances on slide trigger words **and** accepts all explicit commands |
| `follow-semantic-words` | Semantic text embedding match; cues whichever slide best matches recent speech; does **not** parse explicit commands |
| `follow-semantic-audio` | MERT audio embedding match against per-slide prototypes built from reference audio; does **not** parse explicit commands; requires `--ground-truth` |
