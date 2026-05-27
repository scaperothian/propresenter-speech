# speech-accuracy

Offline accuracy evaluation tool for the `follow-enhanced` slide-matching pipeline.

Given a ground-truth JSON file from [`propresenter-train`](../../propresenter-train), it runs
Whisper transcription + sentence-transformer embedding inference over the audio file and
scores each prediction against the known slide timings.

---

## How it works

1. **Load ground truth** — reads slide texts and timestamps from a `propresenter-train` JSON file.  
   Two timing formats are supported and normalised automatically:
   - `"start time"` / `"stop time"` (e.g. *Mary Had A Little Lamb*, *Your Way Is Better*)
   - `"trigger time"` only, stop derived from next slide's start (e.g. *The Pledge of Allegiance*)
   
   Slides with `"enabled": false` are excluded, matching follow-enhanced startup behaviour.

2. **Build embeddings** — constructs a `SlideEmbedder` (sentence-transformers `all-MiniLM-L6-v2`)
   from the slide texts, exactly as the live pipeline does.  `context_words` defaults to the
   average word count per slide unless overridden with `--context-words`.

3. **Run the audio pipeline** — audio is processed through the real `AudioPipeline`
   (the same code used in production) with `interactive=False` so it runs at full CPU
   speed rather than real-time.  The pipeline walks the file in sequential non-overlapping
   `--window-seconds` chunks (e.g. 90 calls for a 3-minute file at 2 s/chunk).
   Each chunk yields a T_snap value — the audio file position (seconds) at the END of
   the transcribed window, derived from frame counts with no wall-clock jitter.

4. **Score every inference call** — at each T_snap the *raw*
   `SlideEmbedder.find_slide_with_margin()` result is compared against the ground-truth
   slide active at that exact audio position, independent of cue-threshold or same-slide
   suppression.  This makes every chunk individually evaluable.

5. **Log + summarise** — each inference event is written as a JSON object to a `.log` file
   (one JSON object per line) for later analysis.  A human-readable summary is printed to stdout.

---

## CLI: `speech-accuracy`

Evaluate a single presentation.

```
speech-accuracy --ground-truth PATH [options]

Required:
  --ground-truth PATH       propresenter-train ground-truth JSON file

Whisper / audio:
  --model {tiny,base,...}   Whisper model size (default: base)
  --window-seconds SECS     Rolling audio window width (default: 2.0)
  --poll-interval SECS      Step between inference calls (default: 0.2)
  --context-words N         Words from word buffer used as query
                            (default: avg words/slide, computed from presentation)

Matching (informational — all raw predictions are logged regardless):
  --similarity-threshold F  Minimum confidence (default: 0.4)
  --min-margin F            Minimum score gap (default: 0.15)

Output:
  --log-file PATH           Event log (default: speech_accuracy_<name>_<ts>.log)
  --verbose                 Print every inference step to stdout
  --log-level LEVEL         Logging verbosity (default: WARNING)
```

### Examples

```bash
# Basic evaluation with default settings
poetry run speech-accuracy \
  --ground-truth ../propresenter-train/output/"Mary Had A Little Lamb.json"

# Faster run with tiny model, verbose step output
poetry run speech-accuracy \
  --ground-truth ../propresenter-train/output/"Your Way Is Better.json" \
  --model tiny --verbose

# Tune context window, save log to a specific file
poetry run speech-accuracy \
  --ground-truth ../propresenter-train/output/"The Pledge of Allegiance.json" \
  --context-words 6 \
  --log-file results/pledge_base.log
```

---

## CLI: `speech-accuracy-batch`

Batch evaluation over every JSON file in a directory.  Whisper is loaded once and
reused across all presentations for efficiency.

```
speech-accuracy-batch --ground-truth-dir DIR [options]

Required:
  --ground-truth-dir DIR    Directory of propresenter-train JSON files

All other options are the same as speech-accuracy.
```

### Example

```bash
poetry run speech-accuracy-batch \
  --ground-truth-dir ../propresenter-train/output/ \
  --model base
```

Output:

```
────────────────────────────────────────────────────────────
  AGGREGATE SUMMARY
────────────────────────────────────────────────────────────
  Presentation                    Accuracy  Missed
  ──────────────────────────────  ────────  ──────
  Mary Had A Little Lamb            87.5%       0
  Your Way Is Better                72.1%       2
  The Pledge of Allegiance          91.3%       0
  ──────────────────────────────  ────────  ──────
  OVERALL                           82.4%
```

---

## JSONL log format

Each inference call is written as one JSON object per line:

```json
{"audio_time": 14.4, "query": "mary went her lamb was sure", "gt_slide_idx": 1,
 "gt_slide_text": "Everywhere that Mary went\n...", "pred_slide_idx": 1,
 "confidence": 0.83, "margin": 0.31, "is_correct": true}
```

A `"record_type": "summary"` object is appended at the end of each run with aggregate
metrics and per-slide breakdowns for programmatic analysis.

### Analysing logs

```python
import json

events = []
with open("speech_accuracy_Mary_Had_A_Little_Lamb_20260523_120000.log") as f:
    for line in f:
        obj = json.loads(line)
        if "record_type" not in obj:          # skip summary record
            events.append(obj)

# Per-slide accuracy
from collections import defaultdict
by_slide = defaultdict(list)
for e in events:
    if e["gt_slide_idx"] >= 0:
        by_slide[e["gt_slide_idx"]].append(e["is_correct"])

for idx, results in sorted(by_slide.items()):
    acc = sum(results) / len(results)
    print(f"slide {idx}: {acc:.0%}  ({sum(results)}/{len(results)})")
```

---

## Metrics explained

| Metric | Definition |
|--------|------------|
| **Inference accuracy** | Fraction of steps where embedder top-1 == ground-truth slide |
| **Per-slide accuracy** | Same, scoped to steps where that slide is the ground truth |
| **Detection latency** | Seconds from slide start to first correct prediction (None = never) |
| **Missed slides** | Slides with zero correct predictions across all steps |

Detection latency is bounded below by `window_seconds` (the first inference call can only
happen once a full window of audio has been heard).

---

## Ground-truth JSON schema

Files live in `../propresenter-train/output/` and have this structure:

```json
{
  "presentation": {
    "id": {
      "name": "Song Name",
      "audio": "/absolute/path/to/audio.wav"
    },
    "groups": [
      {
        "slides": [
          {
            "enabled": true,
            "text": "Slide text here",
            "start time": 0.0,
            "stop time": 10.8
          }
        ]
      }
    ]
  }
}
```

`"trigger time"` (no `stop time`) is also accepted; stop is derived from the next
slide's `trigger time`.
