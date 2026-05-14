# propresenter-speech

Voice-controlled slide advancement for [ProPresenter](https://renewedvision.com/propresenter/)
using [Whisper](https://github.com/openai/whisper) ASR via
[faster-whisper](https://github.com/SYSTRAN/faster-whisper), running on-device.

Four modes of operation:

- **`presentation` mode** (default) — respond to explicit voice commands: "next slide", "previous slide", "go to slide five".
- **`follow` mode** — automatically advance when a trigger phrase near the end of the active slide is heard, while also accepting all explicit commands.
- **`follow-enhanced` mode** — continuously matches recent speech to slide text using sentence-transformer embeddings (all-MiniLM-L6-v2); cues whichever slide best matches what was just said, and can jump forward or backward freely.
- **`follow-enhanced-plus` mode** — combines both: embedding search runs first and jumps to the best-matching slide when confident; falls back to trigger-word matching for sequential advances when the match is ambiguous. Accepts explicit commands.

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
> **`follow-enhanced` / `follow-enhanced-plus` note:** The sentence-transformer model (`all-MiniLM-L6-v2`, ~80 MB) is also
> downloaded automatically on first use of either embedding mode.

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

# Follow-enhanced-plus — embedding search + trigger-word fallback
poetry run propresenter-speech --mode follow-enhanced-plus
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
| *(any slide text)* | Jump to best-matching slide | follow-enhanced, follow-enhanced-plus |

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
  --mode {presentation,follow,follow-enhanced,follow-enhanced-plus}
                        presentation: explicit commands only (default)
                        follow: auto-advance on slide trigger words + explicit commands
                        follow-enhanced: semantic embedding search — cues whichever slide best matches recent speech
                        follow-enhanced-plus: embedding search + trigger-word fallback — jumps when confident, advances sequentially when ambiguous
  --trigger-words N     (follow, follow-enhanced-plus) number of words in the trigger phrase (default: 2)
  --trigger-index I     (follow, follow-enhanced-plus) pythonic index of the anchor trigger word;
                        -2 = second-to-last word (default), -1 = last word
  --context-words N     (follow-enhanced, follow-enhanced-plus) recent spoken words used to form the query n-gram (default: 3)
  --similarity-threshold FLOAT
                        (follow-enhanced, follow-enhanced-plus) minimum score to trigger a slide cue (default: 0.4)
  --min-margin FLOAT    (follow-enhanced, follow-enhanced-plus) minimum gap between best and second-best score
                        to trigger even when below --similarity-threshold (default: 0.15)

Whisper ASR:
  --model {tiny,base,small,medium,large}
                        Whisper model size (default: base)
                        tiny < base < small < medium < large
                        Smaller = faster; larger = more accurate

Audio pipeline:
  --device DEVICE       Input device index; see --list-devices (default: system default)
  --window-seconds SECS Rolling audio window length fed to Whisper, all modes (default: 2.0)
  --poll-interval SECS  Seconds between Whisper inference calls, all modes (default: 0.2)
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

# Follow mode — trigger on the last word only (override default second-to-last, 2-word phrase)
poetry run propresenter-speech --mode follow --trigger-index=-1 --trigger-words=1

# Follow-enhanced mode — semantic slide matching from active presentation
poetry run propresenter-speech --mode follow-enhanced

# Follow-enhanced with stricter matching (raise threshold) or more context
poetry run propresenter-speech --mode follow-enhanced --similarity-threshold 0.55 --context-words 5

# Follow-enhanced-plus — embedding + trigger words (all embedding and trigger flags apply)
poetry run propresenter-speech --mode follow-enhanced-plus --presentation "Sermon Slides"

# Use a more accurate model
poetry run propresenter-speech --model small

# Show what Whisper is hearing and which trigger words are active
poetry run propresenter-speech --mode follow --verbose

# List audio input devices, then use device #2
poetry run propresenter-speech --list-devices
poetry run propresenter-speech --device 2

# Connect to ProPresenter on another machine
poetry run propresenter-speech --host 192.168.1.10

# Reduce false positives in a noisy room (longer window gives Whisper more context)
poetry run propresenter-speech --window-seconds 3.0
```

---

## Architecture

**`presentation` / `follow` pipeline:**

All three modes share a single `AudioPipeline` (ring buffer + Whisper polling). Mode logic lives exclusively in a `ModeHandler` class.

```
Microphone (or audio file)
    │
    ▼
AudioPipeline          (sounddevice ring buffer, poll every --poll-interval s)
    │  text + rolling word buffer
    ▼
ModeHandler.on_transcription()
    │
    ├── PresentationHandler  →  CommandParser  →  ProPresenterController
    │
    ├── FollowHandler        →  CommandParser + SlideFollower  →  ProPresenterController
    │                           (SlideFollower caches presentation details,
    │                            tracks slide index, extracts trigger words)
    │
    ├── FollowEnhancedHandler    →  SlideEmbedder.find_slide_with_margin()  →  ProPresenterController
    │                               (cosine similarity over all-MiniLM-L6-v2 embeddings,
    │                                built at startup from active presentation slides)
    │
    └── FollowEnhancedPlusHandler → SlideEmbedder (jump when confident)
                                    + SlideFollower (trigger-word fallback)
                                    + CommandParser (explicit commands)  →  ProPresenterController
```

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

All components are injected into `AudioPipeline` — easy to swap (e.g. replace
`Transcriber` with a web-hosted model, or add new `ModeHandler` variants).

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
Use a directional or headset microphone, or raise `--window-seconds` so Whisper
gets more context and is less likely to misfire on a short noise burst.

**Commands are cut off or lag**  
Lower `--poll-interval` (e.g. `0.1`) for faster response, or raise
`--window-seconds` (e.g. `3.0`) to give Whisper more audio context.

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
| `follow-enhanced-plus` | Yes (embedding jump or trigger-word advance) | Yes | Yes (embeddings + trigger words at startup) |

---

## License

MIT
