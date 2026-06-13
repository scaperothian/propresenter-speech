# propresenter-speech

Voice-controlled slide advancement for [ProPresenter](https://renewedvision.com/propresenter/)
using [Whisper](https://github.com/openai/whisper) ASR via
[faster-whisper](https://github.com/SYSTRAN/faster-whisper), running on-device.

Four modes of operation:

- **`presentation` mode** (default) — respond to explicit voice commands: "next slide", "previous slide", "go to slide five".
- **`follow-trigger-words` mode** — automatically advance when a trigger phrase near the end of the active slide is heard, while also accepting all explicit commands.
- **`follow-semantic-words` mode** — continuously matches recent speech to slide text using sentence-transformer embeddings (all-MiniLM-L6-v2); cues whichever slide best matches what was just said, and can jump forward or backward freely.
- **`follow-semantic-audio` mode** — matches live microphone audio to per-slide MERT embeddings built from reference audio at startup; no transcription required.

---

## Requirements

| Requirement | Notes |
|-------------|-------|
| macOS | Tested on macOS 13+; built-in or headset mic both work |
| Python 3.11+ | Managed via Poetry |
| [Poetry](https://python-poetry.org) | `pip install poetry` or `brew install poetry` |
| ProPresenter 7 | Network API must be enabled: Preferences → Network → Enable Network |
| `../propresenter-client` | Sibling directory — the HTTP client library |

> **Note:** Whisper model weights (~74 MB for `base`) are downloaded automatically on first run
> from HuggingFace and cached in `~/.cache/huggingface/hub/`.
>
> **`follow-semantic-words` note:** The sentence-transformer model (`all-MiniLM-L6-v2`, ~80 MB) is also
> downloaded automatically on first use.
>
> **`follow-semantic-audio` note:** The MERT model (`m-a-p/MERT-v1-95M`, ~380 MB) requires the
> torch extra: `poetry install --extras torch`.  A propresenter-train ground-truth JSON file is
> also required to build slide prototypes at startup.
>
> **Source-separation note:** Vocal isolation is optional.  On Apple Silicon, install the
> `separation-mlx` extra to run Demucs on the Apple GPU (`demucs-mlx`) — roughly 3× faster than
> the torch `separation` extra, which is CPU-only on Mac (htdemucs can't run on MPS).  The default
> `--separation-backend auto` uses demucs-mlx when installed and falls back to torch Demucs.

---

## Installation

```bash
# Clone (or cd into) this repo
cd propresenter-speech

# Install all dependencies (creates a .venv automatically)
poetry install

# For wav2vec2 / wav2vec2-alt / MERT backends (requires PyTorch 2.7.0+)
poetry install --extras torch

# For Demucs source separation (vocal isolation before ASR — torch, CPU-only on Mac)
poetry install --extras separation

# For demucs-mlx source separation (Apple GPU — much better real-time factor on Apple Silicon)
poetry install --extras separation-mlx

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

# Follow-trigger-words mode — auto-advances on trigger phrase near end of each slide
poetry run propresenter-speech --mode follow-trigger-words

# Follow-semantic-words mode — semantic text matching against all slides
poetry run propresenter-speech --mode follow-semantic-words

# Follow-semantic-audio mode — MERT audio matching (requires ground-truth JSON)
poetry run propresenter-speech --mode follow-semantic-audio \
  --ground-truth ../propresenter-train/output/Song.json
```

3. Speak into your mic:

| Say | Action | Modes |
|-----|--------|-------|
| "next slide" | Advance one slide | presentation, follow-trigger-words |
| "previous slide" | Go back one slide | presentation, follow-trigger-words |
| "go back" | Go back one slide | presentation, follow-trigger-words |
| "go to slide five" | Jump to slide 5 | presentation, follow-trigger-words |
| "slide number 12" | Jump to slide 12 | presentation, follow-trigger-words |
| *(trigger phrase near end of slide)* | Auto-advance | follow-trigger-words |
| *(any slide text)* | Jump to best-matching slide | follow-semantic-words |
| *(audio matching a slide)* | Jump to best-matching slide | follow-semantic-audio |

Press **Ctrl-C** to stop.

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
  --mode {presentation,follow-trigger-words,follow-semantic-words,follow-semantic-audio}
                        presentation:          explicit commands only (default)
                        follow-trigger-words:  auto-advance on slide trigger phrase + explicit commands
                        follow-semantic-words: text embedding search — cues whichever slide best matches recent speech
                        follow-semantic-audio: MERT audio embedding search — cues whichever slide best matches live audio
  --trigger-words N     (follow-trigger-words) number of words in the trigger phrase (default: 2)
  --trigger-index I     (follow-trigger-words) pythonic index of the anchor trigger word;
                        -2 = second-to-last word (default), -1 = last word
  --context-words N     (follow-semantic-words) recent spoken words used to form the query n-gram
                        (default: avg words/slide, computed at startup from the loaded presentation)
  --similarity-threshold FLOAT
                        (follow-semantic-words/audio) minimum score to trigger a slide cue (default: 0.4)
  --min-margin FLOAT    (follow-semantic-words/audio) minimum gap between best and second-best score
                        to trigger even when below --similarity-threshold (default: 0.15)
  --embedding-mode {slide,word-window}
                        (follow-semantic-words) slide: one embedding per slide (default)
                        word-window: one embedding per word position — finer resolution,
                                     better for repeated sections (choruses)
  --embedding-stride N  (word-window) words to advance between successive windows (default: 1)
  --ground-truth PATH   (follow-semantic-audio) path to propresenter-train ground-truth JSON;
                        used to build per-slide MERT prototype embeddings at startup

ASR backend:
  --asr-backend {whisper,wav2vec2,wav2vec2-alt}
                        whisper:      faster-whisper CTranslate2 (default)
                        wav2vec2:     HuggingFace Wav2Vec2ForCTC (requires --extras torch)
                        wav2vec2-alt: SpeechBrain ALT lyric-tuned checkpoint (requires --extras torch)
  --model {tiny,base,small,medium,large}
                        (whisper) Whisper model size (default: base)
  --wav2vec-ckpt-dir PATH
                        (wav2vec2-alt) path to SpeechBrain CKPT+... directory

Source separation:
  --source-separation {auto,on,off}
                        auto: Demucs vocal isolation on in follow-semantic-words mode,
                              off otherwise (default)
                        on:   always isolate vocals before ASR (requires --extras separation)
                        off:  never isolate
  --separation-backend {auto,demucs,demucs-mlx}
                        auto:       demucs-mlx (Apple GPU) if installed, else torch demucs (default)
                        demucs:     torch Demucs (--extras separation)
                        demucs-mlx: MLX Demucs on the Apple GPU (--extras separation-mlx)
  --separation-model {htdemucs,htdemucs_ft}
                        Demucs model (default: htdemucs; htdemucs_ft: higher quality, ~4x slower)
  --separation-device {auto,cpu,mps,cuda}
                        Torch device for the demucs backend; ignored by demucs-mlx (default: auto)

Audio pipeline:
  --device DEVICE       Input device index; see --list-devices (default: system default)
  --window-seconds SECS Rolling audio window length (default: 2.0)
  --poll-interval SECS  Seconds between inference calls (default: 0.2)
  --list-devices        Print available input and output audio devices and exit

Misc:
  --verbose             Print transcriptions and match info to stdout
  --log-level {DEBUG,INFO,WARNING,ERROR}
```

### Examples

```bash
# Activate a specific presentation, then listen for commands
poetry run propresenter-speech --presentation "Sermon Slides"

# Activate from a non-default library
poetry run propresenter-speech --presentation "How Great Thou Art" --library Songs

# Activate a presentation and use follow-trigger-words mode
poetry run propresenter-speech --presentation "Sermon Slides" --mode follow-trigger-words

# Trigger on the last word only (override default second-to-last, 2-word phrase)
poetry run propresenter-speech --mode follow-trigger-words --trigger-index=-1 --trigger-words=1

# Semantic text matching from active presentation
poetry run propresenter-speech --mode follow-semantic-words

# Semantic text matching with stricter thresholds or explicit context window
poetry run propresenter-speech --mode follow-semantic-words --similarity-threshold 0.55 --context-words 8

# Word-window embedder (better for songs with repeated choruses)
poetry run propresenter-speech --mode follow-semantic-words --embedding-mode word-window
poetry run propresenter-speech --mode follow-semantic-words --embedding-mode word-window --embedding-stride 2

# MERT audio embedding mode (requires ground-truth JSON and torch extra)
poetry run propresenter-speech --mode follow-semantic-audio \
  --ground-truth ../propresenter-train/output/"How Great Thou Art.json"

# Source separation: auto-on in follow-semantic-words; disable it, or force it in another mode
poetry run propresenter-speech --mode follow-semantic-words --source-separation off
poetry run propresenter-speech --mode presentation --source-separation on

# Separation backend: auto picks the Apple-GPU demucs-mlx when installed (best RTF), else torch
poetry run propresenter-speech --mode follow-semantic-words --separation-backend demucs-mlx

# Use a more accurate Whisper model
poetry run propresenter-speech --model small

# Show what Whisper is hearing and which trigger words are active
poetry run propresenter-speech --mode follow-trigger-words --verbose

# List all input and output audio devices, then use device #2 for input
poetry run propresenter-speech --list-devices
poetry run propresenter-speech --device 2

# Connect to ProPresenter on another machine
poetry run propresenter-speech --host 192.168.1.10

# Reduce false positives in a noisy room (longer window gives Whisper more context)
poetry run propresenter-speech --window-seconds 3.0
```

---

## Architecture

All modes share a single `AudioPipeline` (ring buffer + predictor polling). The pipeline is
model-agnostic: it calls a `Predictor` to convert audio to a result, then hands that result
to a `ModeHandler`.

```
Microphone
    │
    ▼
AudioPipeline          (sounddevice ring buffer, poll every --poll-interval s)
    │  audio chunk
    ▼
SourceSeparator.separate()   (optional, --source-separation; vocal isolation)
    │                         DemucsSeparator (torch) or MLXDemucsSeparator (Apple GPU)
    ▼
Predictor.predict()
    │
    ├── WhisperPredictor     → TranscriptionResult(text, word_buffer)
    ├── Wav2VecPredictor     → TranscriptionResult(text, word_buffer)
    ├── Wav2VecAltPredictor  → TranscriptionResult(text, word_buffer)
    └── MERTPredictor        → AudioEmbeddingResult(embedding)
    │
    ▼
ModeHandler.on_prediction()
    │
    ├── PresentationHandler         →  CommandParser  →  ProPresenterController
    │
    ├── FollowTriggerWordsHandler   →  CommandParser + SlideFollower  →  ProPresenterController
    │                                  (SlideFollower caches presentation details,
    │                                   tracks slide index, extracts trigger words)
    │
    ├── FollowSemanticWordsHandler  →  SlideEmbedder.find_slide_with_margin()  →  ProPresenterController
    │                                  (cosine similarity over all-MiniLM-L6-v2 embeddings,
    │                                   built at startup from active presentation slides)
    │
    └── FollowSemanticAudioHandler  →  mean-centered cosine vs MERT prototypes  →  ProPresenterController
                                       (prototypes built at startup from --ground-truth reference audio;
                                        same MERTPredictor instance used for both prototype building and live inference)
```

**Follow-trigger-words slide-text flow (per `refresh()`):**

| Step | Endpoint |
|------|----------|
| 1. Resolve presentation UUID | `GET /v1/presentation/active` |
| 2. Fetch + cache presentation details | `GET /v1/presentation/{uuid}` |
| 3. Get current slide index | `GET /v1/presentation/slide_index?chunked=false` |
| 4. Read `slides[index]["text"]` | — (from cached details) |
| fallback | `GET /v1/status/slide` |

After each auto-advance `refresh_after_advance()` increments the locally cached slide index
rather than calling `GET /v1/presentation/slide_index` again, avoiding a race condition where
the API returns the pre-advance value.

Full ProPresenter API reference: `http://<propresenter-ip>:1025/v1/doc/index.html#`

---

## Running the tests

Most tests are fully unit-level — no microphone, GPU, or ProPresenter instance needed.

```bash
poetry run pytest          # run all tests
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

**follow-semantic-audio matches the wrong slide**  
Try raising `--similarity-threshold` or `--min-margin` to require more confident
matches.  Check that the ground-truth JSON covers the full audio range — slides with
very short reference windows (< 0.1 s) are skipped during prototype building.

---

## Performance evaluation

The `speech-accuracy`, `speech-accuracy-batch`, and `speech-accuracy-plot` CLI tools measure
and visualise how well the follow-semantic-words pipeline matches slides against ground-truth timing
data from [`propresenter-train`](../propresenter-train).

See [src/speech_accuracy/speech-accuracy.md](src/speech_accuracy/speech-accuracy.md) for the full reference.

---

## Planned features

- Wake-word activation ("Hey ProPresenter")  
- Continuous transcript display  
- Open/switch presentation by name  
- Web-based Whisper backend (option for slower Macs)
- HMM-based slide prediction — model slide transitions as a hidden Markov chain for temporally coherent cuing
- RL-based slide prediction — train a policy to advance slides based on audio/text state and timing reward signals

## Modes reference

| `--mode` | Auto-advance | Explicit commands | Extra requirements |
|----------|-------------|-------------------|--------------------|
| `presentation` | No | Yes | — |
| `follow-trigger-words` | Yes (trigger phrase, sequential) | Yes | Active presentation in ProPresenter |
| `follow-semantic-words` | Yes (text match, any slide) | No | Active presentation in ProPresenter |
| `follow-semantic-audio` | Yes (audio match, any slide) | No | `--ground-truth` JSON + torch extra |

---

## License

MIT
