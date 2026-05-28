"""
Ground-truth loader and evaluation engine for speech-accuracy.

Evaluation strategy
-------------------
AccuracyEvaluator feeds an audio file through FilePipeline with an
AccuracyHandler in place of the normal slide-cue handler.  This means
evaluation uses exactly the same Whisper transcription loop, word buffer, and
chunk sizing as production follow-semantic-words mode.

Timing: T_snap
--------------
audio_time received in on_transcription() is T_snap — the audio file position
(seconds) at the END of the transcribed window, derived from frame counts:

    T_snap = frame_pos / SAMPLE_RATE

It carries no wall-clock jitter and no Whisper processing latency — Whisper
may take 0.3–0.8 s to return, but by the time on_transcription() fires,
audio_time still refers to the window that was snapshotted, not the current
moment.

Consequence for accuracy measurement: ground-truth lookup uses the exact audio
position the model was reasoning about, independent of how long inference took.

File-mode chunking (sliding window)
-------------------------------------
FilePipeline advances by poll_interval each step and transcribes the trailing
window_seconds of audio — identical to mic mode's ring buffer.  For a 3-minute
file at window_seconds=2.0, poll_interval=0.2 this is 900 calls; at
poll_interval=0.05 it is 3,600 calls.  Unlike mic mode there is no
_whisper_busy throttle, so every step is evaluated regardless of Whisper speed.

Ground-truth JSON schema (../propresenter-train/output/)
---------------------------------------------------------
Two timing variants are normalised to a unified list:
  - "start time" / "stop time"   (Mary Had A Little Lamb, Your Way Is Better)
  - "trigger time" only          (The Pledge of Allegiance)
Slides with enabled=false are excluded, matching follow-semantic-words startup.
"""

from __future__ import annotations

import collections
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from propresenter_speech.audio_pipeline import DEFAULT_WINDOW_SECONDS, DEFAULT_POLL_INTERVAL
from propresenter_speech.file_pipeline import FilePipeline
from propresenter_speech.predictors import TranscriptionResult, WhisperPredictor
from propresenter_speech.slide_embedder import SlideEmbedder, WordWindowEmbedder
from propresenter_speech.transcriber import Transcriber

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GroundTruthSlide:
    idx: int          # 0-based index in the filtered (enabled) slide list
    text: str
    start_sec: float
    stop_sec: float   # exclusive upper bound; 0.0 = sentinel (last slide, use audio duration)


@dataclass
class InferenceEvent:
    audio_time: float       # T_snap: end of transcribed window (seconds into file)
    query: str
    gt_slide_idx: int       # ground-truth index at T_snap (-1 = no slide active yet)
    gt_slide_text: str
    pred_slide_idx: int     # embedder best match (-1 = empty query or no index)
    confidence: float
    margin: float
    is_correct: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SlideMetrics:
    slide_idx: int
    slide_text: str
    total_steps: int
    correct_steps: int
    detection_latency_sec: Optional[float]  # secs from slide start to first correct hit


@dataclass
class EvaluationResult:
    presentation_name: str
    audio_path: str
    total_steps: int
    correct_steps: int
    inference_accuracy: float
    context_words: int
    window_seconds: float
    poll_interval: float
    model_name: str
    per_slide: list[SlideMetrics]
    events: list[InferenceEvent]

    @property
    def missed_slides(self) -> list[SlideMetrics]:
        return [s for s in self.per_slide if s.correct_steps == 0 and s.total_steps > 0]


# ---------------------------------------------------------------------------
# Ground-truth loading
# ---------------------------------------------------------------------------

def load_ground_truth(json_path: str | Path) -> tuple[list[GroundTruthSlide], str, str]:
    """
    Parse a propresenter-train JSON file.

    Time values ("trigger time", "start time", "stop time") may be a list of
    floats (new format — one entry per occurrence of the slide in the song) or
    a single float (legacy format).  Each occurrence expands into its own
    GroundTruthSlide entry so repeated sections (choruses) are represented as
    distinct entries in chronological order.

    Returns:
        slides      — list of GroundTruthSlide sorted by start_sec
        audio_path  — absolute path to the audio file
        name        — presentation name
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    pres = data["presentation"]
    name = pres["id"]["name"]
    audio_path = pres["id"]["audio"]

    raw_slides = [s for g in pres["groups"] for s in g["slides"]]
    enabled = [s for s in raw_slides if s.get("enabled", True)]

    # Accumulate (start_sec, stop_sec, idx, text) — one tuple per occurrence.
    # stop_sec=0.0 is the sentinel meaning "derive from next entry".
    raw_entries: list[tuple[float, float, int, str]] = []

    for i, s in enumerate(enabled):
        text = s.get("text", "").strip()
        if not text:
            continue

        if "start time" in s:
            starts = s["start time"]
            if not isinstance(starts, list):
                starts = [starts]
            stops_raw = s.get("stop time", [])
            if not isinstance(stops_raw, list):
                stops_raw = [stops_raw]
            for j, start in enumerate(starts):
                stop = float(stops_raw[j]) if j < len(stops_raw) else 0.0
                raw_entries.append((float(start), stop, i, text))
        else:
            triggers = s["trigger time"]
            if not isinstance(triggers, list):
                triggers = [triggers]
            for t in triggers:
                raw_entries.append((float(t), 0.0, i, text))

    # Sort chronologically so repeated sections appear in playback order.
    raw_entries.sort(key=lambda e: e[0])

    slides: list[GroundTruthSlide] = [
        GroundTruthSlide(idx=idx, text=text, start_sec=start, stop_sec=stop)
        for start, stop, idx, text in raw_entries
    ]

    # Derive stop_sec from the next entry's start for trigger-time occurrences.
    for j in range(len(slides) - 1):
        if slides[j].stop_sec == 0.0:
            slides[j].stop_sec = slides[j + 1].start_sec

    # Last slide: stop_sec stays 0.0 (sentinel — evaluator uses audio duration)

    return slides, audio_path, name


def ground_truth_at(
    slides: list[GroundTruthSlide], t: float, audio_duration: float
) -> GroundTruthSlide | None:
    """Return the slide active at time t, or None if t is before the first slide."""
    for slide in reversed(slides):
        stop = slide.stop_sec if slide.stop_sec > 0.0 else audio_duration
        if slide.start_sec <= t < stop:
            return slide
    return None


# ---------------------------------------------------------------------------
# AccuracyHandler — ModeHandler implementation for evaluation
# ---------------------------------------------------------------------------

class AccuracyHandler:
    """
    Drop-in ModeHandler that records inference accuracy instead of cueing slides.

    Plugged into AudioPipeline in place of the normal follow-semantic-words handler.
    Receives audio_time (T_snap) via on_transcription() and uses it for exact
    ground-truth lookup.
    """

    def __init__(
        self,
        ground_truth: list[GroundTruthSlide],
        embedder: SlideEmbedder | WordWindowEmbedder,
        context_words: int,
        audio_duration: float,
        similarity_threshold: float = 0.4,
        min_margin: float = 0.15,
        verbose: bool = False,
    ):
        self.ground_truth = ground_truth
        self.embedder = embedder
        self.context_words = context_words
        self.audio_duration = audio_duration
        self.similarity_threshold = similarity_threshold
        self.min_margin = min_margin
        self.verbose = verbose
        self.events: list[InferenceEvent] = []
        self._current_pred_idx: int = -1

    def on_startup(self) -> None:
        pass

    def startup_description(self) -> str:
        return f"Accuracy evaluation active — {len(self.ground_truth)} slides"

    def on_prediction(self, result: TranscriptionResult, audio_time: float = 0.0) -> None:
        query_words = list(result.word_buffer)[-self.context_words:]
        query = " ".join(query_words) if len(query_words) >= 2 else ""

        raw_idx, confidence, margin = (
            self.embedder.find_slide_with_margin(query)
            if query
            else (-1, 0.0, 0.0)
        )

        # Mirror FollowEnhancedHandler's gate: only update the current predicted
        # slide when the match clears at least one threshold.  Below both thresholds
        # the prediction stays on the last accepted slide, same as live behaviour.
        if confidence >= self.similarity_threshold or margin >= self.min_margin:
            self._current_pred_idx = raw_idx
        pred_idx = self._current_pred_idx

        gt = ground_truth_at(self.ground_truth, audio_time, self.audio_duration)
        gt_idx = gt.idx if gt is not None else -1
        gt_text = gt.text if gt is not None else ""
        is_correct = gt_idx >= 0 and pred_idx == gt_idx

        event = InferenceEvent(
            audio_time=round(audio_time, 3),
            query=query,
            gt_slide_idx=gt_idx,
            gt_slide_text=gt_text,
            pred_slide_idx=pred_idx,
            confidence=round(confidence, 4),
            margin=round(margin, 4),
            is_correct=is_correct,
        )
        self.events.append(event)
        logger.info("inference_event %s", json.dumps(event.to_dict()))

        if self.verbose:
            from tqdm import tqdm
            status = "✓" if is_correct else "✗"
            tqdm.write(
                f"  {status} t={audio_time:.1f}s  gt={gt_idx}  pred={pred_idx}"
                f"  conf={confidence:.3f}  margin={margin:.3f}  q={query!r}"
            )


# ---------------------------------------------------------------------------
# AccuracyEvaluator
# ---------------------------------------------------------------------------

class AccuracyEvaluator:
    """
    Drives AudioPipeline over an audio file with an AccuracyHandler and
    returns a scored EvaluationResult.
    """

    def __init__(
        self,
        transcriber: Transcriber,
        embedder: SlideEmbedder | WordWindowEmbedder,
        ground_truth: list[GroundTruthSlide],
        context_words: int,
        window_seconds: float = DEFAULT_WINDOW_SECONDS,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        similarity_threshold: float = 0.4,
        min_margin: float = 0.15,
        verbose: bool = False,
    ):
        self.transcriber = transcriber
        self.embedder = embedder
        self.ground_truth = ground_truth
        self.context_words = context_words
        self.window_seconds = window_seconds
        self.poll_interval = poll_interval
        self.similarity_threshold = similarity_threshold
        self.min_margin = min_margin
        self.verbose = verbose

    def evaluate(self, audio_path: str, presentation_name: str, model_name: str) -> EvaluationResult:
        import soundfile as sf

        # Read duration without decoding the full file so AccuracyHandler can
        # resolve the sentinel stop_sec=0.0 on the last ground-truth slide.
        audio_duration = sf.info(audio_path).duration

        handler = AccuracyHandler(
            ground_truth=self.ground_truth,
            embedder=self.embedder,
            context_words=self.context_words,
            audio_duration=audio_duration,
            similarity_threshold=self.similarity_threshold,
            min_margin=self.min_margin,
            verbose=self.verbose,
        )

        FilePipeline(
            predictor=WhisperPredictor(self.transcriber),
            handler=handler,
            audio_file=audio_path,
            window_seconds=self.window_seconds,
            poll_interval=self.poll_interval,
        ).run()

        return self._build_result(
            events=handler.events,
            presentation_name=presentation_name,
            audio_path=audio_path,
            model_name=model_name,
        )

    def _build_result(
        self,
        events: list[InferenceEvent],
        presentation_name: str,
        audio_path: str,
        model_name: str,
    ) -> EvaluationResult:
        total = len(events)
        correct = sum(1 for e in events if e.is_correct)
        accuracy = correct / total if total else 0.0

        per_slide: list[SlideMetrics] = []
        for slide in self.ground_truth:
            # Scope to this occurrence's time window so that repeated slides
            # (choruses) produce separate metrics rather than being merged.
            stop = slide.stop_sec if slide.stop_sec > 0.0 else float("inf")
            slide_events = [
                e for e in events
                if e.gt_slide_idx == slide.idx and slide.start_sec <= e.audio_time < stop
            ]
            slide_correct = sum(1 for e in slide_events if e.is_correct)
            first_hit = next((e for e in slide_events if e.is_correct), None)
            latency = (
                round(first_hit.audio_time - slide.start_sec, 3)
                if first_hit else None
            )
            per_slide.append(SlideMetrics(
                slide_idx=slide.idx,
                slide_text=slide.text,
                total_steps=len(slide_events),
                correct_steps=slide_correct,
                detection_latency_sec=latency,
            ))

        return EvaluationResult(
            presentation_name=presentation_name,
            audio_path=audio_path,
            total_steps=total,
            correct_steps=correct,
            inference_accuracy=round(accuracy, 4),
            context_words=self.context_words,
            window_seconds=self.window_seconds,
            poll_interval=self.poll_interval,
            model_name=model_name,
            per_slide=per_slide,
            events=events,
        )


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

def print_summary(result: EvaluationResult) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {result.presentation_name}")
    print(f"  model={result.model_name}  window={result.window_seconds}s"
          f"  poll={result.poll_interval}s  context={result.context_words} words")
    print(f"  accuracy: {result.correct_steps}/{result.total_steps}"
          f"  ({result.inference_accuracy * 100:.1f}%)")
    print(f"{'=' * 60}")
    print(f"  {'Slide':<4}  {'Acc':>6}  {'Latency':>9}  Text")
    print(f"  {'-'*4}  {'-'*6}  {'-'*9}  {'-'*30}")
    for sm in result.per_slide:
        pct = f"{sm.correct_steps / sm.total_steps * 100:.0f}%" if sm.total_steps else "  n/a"
        lat = f"{sm.detection_latency_sec:.2f}s" if sm.detection_latency_sec is not None else "   miss"
        text_preview = sm.slide_text.replace("\n", " ")[:40]
        print(f"  {sm.slide_idx:<4}  {pct:>6}  {lat:>9}  {text_preview!r}")
    if result.missed_slides:
        print(f"\n  MISSED ({len(result.missed_slides)}):")
        for sm in result.missed_slides:
            print(f"    slide {sm.slide_idx}: {sm.slide_text.replace(chr(10), ' ')[:60]!r}")
    print()
