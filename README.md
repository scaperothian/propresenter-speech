# propresenter-speech

Voice-controlled slide advancement for [ProPresenter](https://renewedvision.com/propresenter/)
using [Whisper](https://github.com/openai/whisper) ASR via
[faster-whisper](https://github.com/SYSTRAN/faster-whisper), running on-device.

Three modes of operation:

- **`presentation` mode** (default) — respond to explicit voice commands: "next slide", "previous slide", "go to slide five".
- **`follow` mode** — automatically advance when the last word(s) of the active slide are heard, while also accepting all explicit commands.
- **`follow-enhanced` mode** — continuously matches recent speech to slide text using sentence-transformer embeddings (all-MiniLM-L6-v2); cues whichever slide best matches what was just said, and can jump forward or backward freely.

---

## Requirements

| Requirement | Notes |
|-------------|-------|
| macOS | Tested on macOS 13+; built-in or headset mic both work |
| Python 3.11+ | Managed via Poetry |
| [Poetry](https://python-poetry.org) | `pip install poetry` or `brew install poetry` |
| [ffmpeg](https://ffmpeg.org) | Not required for the mic pipeline; only needed if passing audio files directly |
| ProPresenter 7 | Network API must be enabled: Preferences → Network → Enable Network |
| `../propresenter-slides` | Sibling directory — the HTTP client library |

> **Note:** Whisper model weights (~74 MB for `base`) are downloaded automatically on first run
> from HuggingFace and cached in `~/.cache/huggingface/hub/`.
>
> **`follow-enhanced` note:** The sentence-transformer model (`all-MiniLM-L6-v2`, ~80 MB) is also
> downloaded automatically on first use of `--mode follow-enhanced`.

---

## Installation

```bash
# Clone (or cd into) this repo
cd propresenter-speech

# Install all dependencies (creates a .venv automatically)
poetry install

# Verify the CLI is available
poetry run propresenter-speech --help
```

---

## Quick start

1. Open ProPresenter and enable the Network API  
   *(Preferences → Network → Enable Network, default port 1025)*

2. Run the CLI:

```bash
# Explicit-command mode (default)
poetry run propresenter-speech

# Follow mode — auto-advances on the last word of each slide
poetry run propresenter-speech --mode follow

# Follow-enhanced mode — semantic matching against all slides
poetry run propresenter-speech --mode follow-enhanced
```

3. Speak into your mic:

| Say | Action | Modes |
|-----|--------|-------|
| "next slide" | Advance one slide | both |
| "previous slide" | Go back one slide | both |
| "go back" | Go back one slide | both |
| "go to slide five" | Jump to slide 5 | both |
| "slide number 12" | Jump to slide 12 | both |
| "jump to slide 3" | Jump to slide 3 | both |
| *(last word of slide)* | Auto-advance | follow only |
| *(any slide text)* | Jump to best-matching slide | follow-enhanced only |

Press **Ctrl-C** or type `q` + Enter to stop.

---

## CLI options

```
propresenter-speech [options]

ProPresenter connection:
  --host HOST           ProPresenter hostname or IP (default: localhost)
  --port PORT           ProPresenter API port (default: 1025)
  --timeout TIMEOUT     HTTP request timeout in seconds (default: 5)

Presentation selection:
  --presentation NAME   Activate a presentation by name before listening
                        (case-insensitive substring match; skipped if omitted)
  --library NAME        Library to search when --presentation is given (default: Default)

Operation mode:
  --mode {presentation,follow,follow-enhanced}
                        presentation: explicit commands only (default)
                        follow: auto-advance on slide trigger words + explicit commands
                        follow-enhanced: semantic embedding search — cues whichever slide best matches recent speech
  --trigger-words N     (follow mode) words from end of slide text to use as trigger (default: 1)
  --context-words N     (follow-enhanced) recent spoken words used to form the query n-gram (default: 3)
  --similarity-threshold FLOAT
                        (follow-enhanced) minimum hybrid score to trigger a slide cue (default: 0.4)
  --min-margin FLOAT    (follow-enhanced) minimum gap between best and second-best score to trigger
                        even when below --similarity-threshold (default: 0.15)
  --window-seconds SECS (follow-enhanced) rolling audio window length in seconds (default: 2.0)
  --poll-interval SECS  (follow-enhanced) seconds between Whisper inference calls (default: 0.2)


Whisper ASR:
  --model {tiny,base,small,medium,large}
                        Whisper model size (default: base)
                        tiny < base < small < medium < large
                        Smaller = faster; larger = more accurate

Audio capture:
  --device DEVICE       Input device index; see --list-devices (default: system default)
  --silence-threshold   RMS energy threshold 0–1 for speech vs. silence (default: 0.01)
  --silence-duration    Seconds of silence to close a speech segment (default: 0.8)
  --list-devices        Print available input devices and exit
  --audio-file PATH     Process a WAV/FLAC/OGG file instead of the microphone
                        (resampled to 16 kHz automatically; program idles when done)

Misc:
  --verbose             Print transcriptions and trigger words to stdout
  --log-level {DEBUG,INFO,WARNING,ERROR}
```

### Examples

```bash
# Activate a specific presentation, then listen for commands
poetry run propresenter-speech --presentation "Sermon Slides"

# Activate from a non-default library
poetry run propresenter-speech --presentation "How Great Thou Art" --library Songs

# Activate a presentation and use follow mode
poetry run propresenter-speech --presentation "Sermon Slides" --mode follow

# Follow mode using the last 2 words as trigger (fewer false positives)
poetry run propresenter-speech --mode follow --trigger-words 2

# Follow-enhanced mode — semantic slide matching from active presentation
poetry run propresenter-speech --mode follow-enhanced

# Follow-enhanced with stricter matching (raise threshold) or more context
poetry run propresenter-speech --mode follow-enhanced --similarity-threshold 0.55 --context-words 5

# Use a more accurate model
poetry run propresenter-speech --model small

# Show what Whisper is hearing and which trigger words are active
poetry run propresenter-speech --mode follow --verbose

# List audio input devices, then use device #2
poetry run propresenter-speech --list-devices
poetry run propresenter-speech --device 2

# Connect to ProPresenter on another machine
poetry run propresenter-speech --host 192.168.1.10

# Reduce false positives in a noisy room
poetry run propresenter-speech --silence-threshold 0.03
```

---

## Architecture

**`presentation` / `follow` pipeline:**

```
Microphone
    │
    ▼
AudioCapture          (sounddevice — 100 ms chunks, energy VAD)
    │  speech segments (numpy float32 arrays)
    ▼
Transcriber           (faster-whisper — base model by default)
    │  transcribed text
    ▼
SpeechController  ── mode-aware dispatch ──────────────────────────┐
    │                                                               │
    │  presentation mode                                      follow mode
    │  (explicit commands only)               (trigger words + explicit commands)
    ▼                                                               │
CommandParser                                               SlideFollower
    │  Command(type, slide_number)               (caches presentation details;
    ▼                                             gets slide index each refresh;
ProPresenterController  (HTTP GET to ProPresenter Network API)      reads slide["text"] directly)
```

**`follow-enhanced` pipeline:**

```
Microphone
    │
    ▼
sounddevice InputStream  →  ring buffer (rolling WINDOW_SECONDS of PCM)
                                  │
                           timer thread (every POLL_INTERVAL s, when Whisper is free)
                                  │   audio snapshot
                                  ▼
                              Transcriber (Whisper)
                                  │   text → rolling word deque
                                  ▼
                              SlideEmbedder.find_slide_with_margin()
                                  │   (slide_index, confidence, margin)
                                  ▼
                              ProPresenterController.go_to_slide()  ← only on new match
```

`SlideEmbedder` builds dense cosine-similarity scores over `all-MiniLM-L6-v2` embeddings at startup. Slide indices are preserved so slides without text are skipped cleanly.

**Follow mode slide-text flow (per `refresh()`):**

| Step | Endpoint |
|------|----------|
| 1. Resolve presentation UUID | `GET /v1/presentation/active` |
| 2. Fetch + cache presentation details | `GET /v1/presentation/{uuid}` |
| 3. Get current slide index | `GET /v1/presentation/slide_index?chunked=false` |
| 4. Read `slides[index]["text"]` | — (from cached details) |
| fallback | `GET /v1/status/slide` |

Presentation details are cached for the lifetime of the `SlideFollower` and only re-fetched when the active presentation UUID changes.

After each auto-advance `refresh_after_advance()` is used instead of `refresh()`: it increments the locally cached slide index rather than calling `GET /v1/presentation/slide_index` again, avoiding a race condition where the API returns the pre-advance value. A single audio segment can therefore advance through multiple slides in one pass.

Full ProPresenter API reference: `http://<propresenter-ip>:1025/v1/doc/index.html#`

All components are injected into `SpeechController` — easy to swap (e.g. replace
`Transcriber` with a web-hosted model, or add new `Mode` variants).

---

## Running the tests

Most tests are fully unit-level — no microphone, GPU, or ProPresenter instance needed.

```bash
poetry run pytest          # run all tests (unit + integration)
poetry run pytest -v       # verbose output
poetry run pytest tests/test_command_parser.py -v
```

`tests/test_integration_follow.py` contains two test classes:
- `TestFollowModeTriggerOrder` — pure unit tests for `refresh_after_advance()` trigger sequencing; always runs.
- `TestFollowModeAudio` — end-to-end tests using the real WAV file in `audio/` and the Whisper `tiny` model; automatically skipped when the audio file is absent.

---

## Tuning tips

**Whisper triggers on background noise**  
Raise `--silence-threshold` (try `0.02`–`0.05`).  You can also switch to a
quieter environment or use a directional/headset microphone.

**Commands are cut off** (e.g. "go to slide" fires before you finish)  
Lower `--silence-duration` slightly (e.g. `0.5`) or raise it to give yourself
more time (e.g. `1.2`).

**Whisper is too slow** (noticeable lag)  
Switch to `--model tiny`.  faster-whisper uses CTranslate2 and int8 quantisation
by default, so `base` typically responds in under 1 second on Apple Silicon.

**"go to slide" picks up the wrong number**  
Speak clearly and pause briefly after the number.  You can also try `--model small`
for better word-error rate on numbers.

---

## Future modes (planned)

- Wake-word activation ("Hey ProPresenter")  
- Continuous transcript display  
- Open/switch presentation by name  
- Web-based Whisper backend (option for slower Macs)

## Modes reference

| `--mode` | Auto-advance | Explicit commands | Requires slide text from API |
|----------|-------------|-------------------|-------------------------------|
| `presentation` | No | Yes | No |
| `follow` | Yes (on trigger words, sequential) | Yes | Yes (exits on startup if unavailable) |
| `follow-enhanced` | Yes (semantic match, any slide) | No | Yes (builds embeddings at startup) |

---

## License

MIT
