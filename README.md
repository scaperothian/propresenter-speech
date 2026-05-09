# propresenter-speech

Voice-controlled slide advancement for [ProPresenter](https://renewedvision.com/propresenter/)
using [Whisper](https://github.com/openai/whisper) ASR via
[faster-whisper](https://github.com/SYSTRAN/faster-whisper), running on-device.

Two modes of operation:

- **`presentation` mode** (default) — respond to explicit voice commands: "next slide", "previous slide", "go to slide five".
- **`follow` mode** — automatically advance when the last word(s) of the active slide are heard, while also accepting all explicit commands.

---

## Requirements

| Requirement | Notes |
|-------------|-------|
| macOS | Tested on macOS 13+; built-in or headset mic both work |
| Python 3.10+ | Managed via Poetry |
| [Poetry](https://python-poetry.org) | `pip install poetry` or `brew install poetry` |
| [ffmpeg](https://ffmpeg.org) | Not required for the mic pipeline; only needed if passing audio files directly |
| ProPresenter 7 | Network API must be enabled: Preferences → Network → Enable Network |
| `../propresenter-slides` | Sibling directory — the HTTP client library |

> **Note:** Whisper model weights (~74 MB for `base`) are downloaded automatically on first run
> from HuggingFace and cached in `~/.cache/huggingface/hub/`.

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

Press **Ctrl-C** to stop.

---

## CLI options

```
propresenter-speech [options]

ProPresenter connection:
  --host HOST           ProPresenter hostname or IP (default: localhost)
  --port PORT           ProPresenter API port (default: 1025)
  --timeout TIMEOUT     HTTP request timeout in seconds (default: 5)

Operation mode:
  --mode {presentation,follow}
                        presentation: explicit commands only (default)
                        follow: auto-advance on slide trigger words + explicit commands
  --trigger-words N     (follow mode) words from end of slide to use as trigger (default: 1)

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

Misc:
  --verbose             Print transcriptions and trigger words to stdout
  --log-level {DEBUG,INFO,WARNING,ERROR}
```

### Examples

```bash
# Follow mode — auto-advance on the last word of each slide
poetry run propresenter-speech --mode follow

# Follow mode using the last 2 words as trigger (less false positives)
poetry run propresenter-speech --mode follow --trigger-words 2

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
    │  Command(type, slide_number)               (fetches slide text from PP API,
    ▼                                             extracts last N words as triggers)
ProPresenterController  (HTTP GET to ProPresenter Network API)
```

All components are injected into `SpeechController` — easy to swap (e.g. replace
`Transcriber` with a web-hosted model, or add new `Mode` variants).

---

## Running the tests

Tests are fully unit-level — no microphone, GPU, or ProPresenter instance needed.

```bash
poetry run pytest          # run all tests
poetry run pytest -v       # verbose output
poetry run pytest tests/test_command_parser.py -v
```

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
| `follow` | Yes (on trigger words) | Yes | Yes (degrades gracefully if unavailable) |

---

## License

MIT
