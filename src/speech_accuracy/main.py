"""
CLI entry point for speech-accuracy.

Evaluates follow-semantic-words slide matching accuracy against a propresenter-train
ground-truth JSON file.

Usage:
  speech-accuracy --ground-truth ../propresenter-train/output/song.json
  speech-accuracy --ground-truth ../propresenter-train/output/song.json --model tiny --verbose
  speech-accuracy --ground-truth ../propresenter-train/output/song.json --log-file results.log
  speech-accuracy-batch   --ground-truth-dir ../propresenter-train/output/ --model base
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from .evaluator import (
    AccuracyEvaluator,
    EvaluationResult,
    load_ground_truth,
    print_summary,
)
from propresenter_speech.audio_pipeline import DEFAULT_POLL_INTERVAL, DEFAULT_WINDOW_SECONDS
from propresenter_speech.handlers.follow_semantic_words import DEFAULT_MIN_MARGIN, DEFAULT_SIMILARITY_THRESHOLD
from propresenter_speech.separation import DEFAULT_DEMUCS_MODEL, DemucsSeparator
from propresenter_speech.slide_embedder import SlideEmbedder, WordWindowEmbedder
from propresenter_speech.transcriber import Transcriber


# ---------------------------------------------------------------------------
# Structured JSONL logging
# ---------------------------------------------------------------------------

class _JsonlHandler(logging.FileHandler):
    """Writes each LogRecord's message to a .jsonl file, one JSON object per line."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.stream.write(record.getMessage() + "\n")
            self.flush()
        except Exception:
            self.handleError(record)


def _setup_jsonl_logger(log_path: str) -> logging.Logger:
    """Return a dedicated logger that writes inference_event lines to log_path."""
    handler = _JsonlHandler(log_path, mode="a", encoding="utf-8")
    log = logging.getLogger("speech_accuracy.events")
    log.setLevel(logging.INFO)
    log.addHandler(handler)
    log.propagate = False
    return log


# ---------------------------------------------------------------------------
# Shared argument parser helpers
# ---------------------------------------------------------------------------

class _Formatter(argparse.RawTextHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=DEFAULT_WINDOW_SECONDS,
        dest="window_seconds",
        metavar="SECS",
        help="Rolling audio window fed to Whisper (seconds)",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=DEFAULT_POLL_INTERVAL,
        dest="poll_interval",
        metavar="SECS",
        help="Step size between inference calls (seconds)",
    )
    parser.add_argument(
        "--context-words",
        type=int,
        default=None,
        dest="context_words",
        metavar="N",
        help=(
            "Words from rolling buffer used as embedding query\n"
            "(default: avg words/slide, computed from presentation)"
        ),
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=DEFAULT_SIMILARITY_THRESHOLD,
        dest="similarity_threshold",
        metavar="FLOAT",
        help="Minimum confidence score (informational only; all raw predictions logged)",
    )
    parser.add_argument(
        "--min-margin",
        type=float,
        default=DEFAULT_MIN_MARGIN,
        dest="min_margin",
        metavar="FLOAT",
        help="Minimum margin between best and second-best score (informational only)",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        dest="log_file",
        metavar="PATH",
        help=(
            "JSONL file for per-inference event log\n"
            "(default: speech_accuracy_<name>_<timestamp>.log)"
        ),
    )
    parser.add_argument(
        "--embedding-mode",
        default="slide",
        choices=["slide", "word-window"],
        dest="embedding_mode",
        help=(
            "slide: one embedding per slide (default)\n"
            "word-window: one embedding per word position — finer resolution,\n"
            "             slides sorted by timestamp so choruses appear in order"
        ),
    )
    parser.add_argument(
        "--embedding-stride",
        type=int,
        default=1,
        dest="embedding_stride",
        metavar="N",
        help="(word-window mode) words to advance between successive windows (default: 1)",
    )
    parser.add_argument(
        "--source-separation",
        default="off",
        choices=["on", "off"],
        dest="source_separation",
        help=(
            "on: isolate vocals with Demucs before transcription\n"
            "(default off so results stay comparable with existing baselines)"
        ),
    )
    parser.add_argument(
        "--separation-model",
        default=DEFAULT_DEMUCS_MODEL,
        choices=["htdemucs", "htdemucs_ft"],
        dest="separation_model",
        help="Demucs model (htdemucs_ft: higher quality, ~4x slower)",
    )
    parser.add_argument(
        "--separation-device",
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        dest="separation_device",
        help="Torch device for Demucs inference",
    )
    parser.add_argument("--verbose", action="store_true", help="Print every inference step to stdout")
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        dest="log_level",
    )


# ---------------------------------------------------------------------------
# Single-file evaluation
# ---------------------------------------------------------------------------

def _build_separator(args) -> DemucsSeparator | None:
    if args.source_separation != "on":
        return None
    print(f"Loading Demucs '{args.separation_model}' — this may take a moment on first run…")
    separator = DemucsSeparator(
        model_name=args.separation_model,
        device=args.separation_device,
        verbose=args.verbose,
    )
    separator.load()
    print(f"Demucs ready (device: {separator.device}).")
    return separator


def _separation_tag(args) -> str:
    return args.separation_model if args.source_separation == "on" else "off"


def _build_evaluator(ground_truth, args) -> tuple[AccuracyEvaluator, Transcriber, int]:
    slide_texts = [s.text for s in ground_truth]

    context_words = args.context_words
    if context_words is None:
        total_words = sum(len(t.split()) for t in slide_texts)
        context_words = max(1, round(total_words / len(slide_texts)))
        print(f"Context words: {context_words} (avg words/slide)")

    if args.embedding_mode == "word-window":
        from propresenter_speech.slide_embedder import WordWindowEmbedder
        # Sort by start_sec so repeated sections (choruses) appear in the word
        # continuum in chronological playback order, not slide-list order.
        ordered = sorted(ground_truth, key=lambda s: s.start_sec)
        embedder: SlideEmbedder | WordWindowEmbedder = WordWindowEmbedder(
            stride=args.embedding_stride
        )
        embedder.load()
        embedder.build([(s.idx, s.text) for s in ordered], context_words=context_words)
    else:
        embedder = SlideEmbedder()
        embedder.load()
        embedder.build(slide_texts, slide_indices=[s.idx for s in ground_truth])

    transcriber = Transcriber(model_name=args.model)
    transcriber.load()

    evaluator = AccuracyEvaluator(
        transcriber=transcriber,
        embedder=embedder,
        ground_truth=ground_truth,
        context_words=context_words,
        window_seconds=args.window_seconds,
        poll_interval=args.poll_interval,
        similarity_threshold=args.similarity_threshold,
        min_margin=args.min_margin,
        verbose=args.verbose,
        separator=_build_separator(args),
    )
    return evaluator, transcriber, context_words


def _default_log_path(name: str) -> str:
    safe = name.replace(" ", "_").replace("/", "-")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"speech_accuracy_{safe}_{ts}.log"


def _write_result_jsonl(
    result: EvaluationResult,
    log_path: str,
    gt_path: str = "",
    embedding_mode: str = "",
    similarity_threshold: float = 0.4,
    min_margin: float = 0.15,
    source_separation: str = "off",
) -> None:
    """Append all inference events + a summary record to the JSONL log."""
    with open(log_path, "a", encoding="utf-8") as f:
        for event in result.events:
            f.write(json.dumps(event.to_dict()) + "\n")
        summary = {
            "record_type": "summary",
            "presentation": result.presentation_name,
            "audio_file": result.audio_path,
            "ground_truth_file": gt_path,
            "embedding_mode": embedding_mode,
            "source_separation": source_separation,
            "model": result.model_name,
            "window_seconds": result.window_seconds,
            "poll_interval": result.poll_interval,
            "context_words": result.context_words,
            "similarity_threshold": similarity_threshold,
            "min_margin": min_margin,
            "total_steps": result.total_steps,
            "correct_steps": result.correct_steps,
            "inference_accuracy": result.inference_accuracy,
            "per_slide": [
                {
                    "idx": s.slide_idx,
                    "correct": s.correct_steps,
                    "total": s.total_steps,
                    "latency_sec": s.detection_latency_sec,
                    "text_preview": s.slide_text.replace("\n", " ")[:60],
                }
                for s in result.per_slide
            ],
        }
        f.write(json.dumps(summary) + "\n")


def speech_accuracy_main() -> None:
    sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(
        prog="speech-accuracy",
        description="Evaluate follow-semantic-words slide matching against a ground-truth JSON file.",
        formatter_class=_Formatter,
    )
    parser.add_argument(
        "--ground-truth",
        required=True,
        metavar="PATH",
        dest="ground_truth",
        help="Path to a propresenter-train ground-truth JSON file",
    )
    _add_common_args(parser)
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    gt_path = Path(args.ground_truth)
    if not gt_path.is_file():
        print(f"Error: ground-truth file not found: {gt_path}")
        sys.exit(1)

    print(f"Loading ground truth: {gt_path.name}")
    ground_truth, audio_path, name = load_ground_truth(gt_path)

    if not Path(audio_path).is_file():
        print(f"Error: audio file not found: {audio_path}")
        sys.exit(1)

    print(f"Presentation: {name!r}  ({len(ground_truth)} slides)")
    print(f"Audio: {audio_path}")
    print("Loading models…")

    evaluator, _, _context_words = _build_evaluator(ground_truth, args)

    log_path = args.log_file or _default_log_path(name)
    print(f"Logging inference events to: {log_path}\n")

    result = evaluator.evaluate(audio_path, name, args.model)

    _write_result_jsonl(
        result, log_path,
        gt_path=str(gt_path),
        embedding_mode=args.embedding_mode,
        similarity_threshold=args.similarity_threshold,
        min_margin=args.min_margin,
        source_separation=_separation_tag(args),
    )
    print_summary(result)
    print(f"Full event log: {log_path}")


# ---------------------------------------------------------------------------
# Multi-file evaluation
# ---------------------------------------------------------------------------

def evaluate_all_main() -> None:
    sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(
        prog="speech-accuracy-batch",
        description="Run speech-accuracy over every JSON file in a directory and print an aggregate summary.",
        formatter_class=_Formatter,
    )
    parser.add_argument(
        "--ground-truth-dir",
        required=True,
        metavar="DIR",
        dest="ground_truth_dir",
        help="Directory containing propresenter-train ground-truth JSON files",
    )
    _add_common_args(parser)
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    gt_dir = Path(args.ground_truth_dir)
    json_files = sorted(gt_dir.glob("*.json"))
    if not json_files:
        print(f"Error: no JSON files found in {gt_dir}")
        sys.exit(1)

    print(f"Found {len(json_files)} ground-truth file(s) in {gt_dir}\n")

    # Load Whisper once and reuse across all presentations
    print("Loading Whisper model…")
    transcriber = Transcriber(model_name=args.model)
    transcriber.load()
    print("Whisper ready.\n")

    separator = _build_separator(args)

    results: list[EvaluationResult] = []

    for gt_path in json_files:
        print(f"─── {gt_path.name} ───")
        try:
            ground_truth, audio_path, name = load_ground_truth(gt_path)
        except Exception as exc:
            print(f"  Skipping (load error): {exc}\n")
            continue

        if not Path(audio_path).is_file():
            print(f"  Skipping (audio not found): {audio_path}\n")
            continue

        slide_texts = [s.text for s in ground_truth]
        embedder = SlideEmbedder()
        embedder.load()
        embedder.build(slide_texts)

        context_words = args.context_words
        if context_words is None:
            context_words = embedder.avg_words_per_slide
            print(f"  Context words: {context_words} (avg words/slide)")

        evaluator = AccuracyEvaluator(
            transcriber=transcriber,
            embedder=embedder,
            ground_truth=ground_truth,
            context_words=context_words,
            window_seconds=args.window_seconds,
            poll_interval=args.poll_interval,
            similarity_threshold=args.similarity_threshold,
            min_margin=args.min_margin,
            verbose=args.verbose,
            separator=separator,
        )

        log_path = args.log_file or _default_log_path(name)
        print(f"  Logging to: {log_path}")

        result = evaluator.evaluate(audio_path, name, args.model)
        _write_result_jsonl(
            result, log_path,
            gt_path=str(gt_path),
            embedding_mode=args.embedding_mode,
            similarity_threshold=args.similarity_threshold,
            min_margin=args.min_margin,
            source_separation=_separation_tag(args),
        )
        print_summary(result)
        results.append(result)

    if not results:
        print("No results to summarise.")
        return

    _print_aggregate(results)


def _print_aggregate(results: list[EvaluationResult]) -> None:
    print("=" * 60)
    print("  AGGREGATE SUMMARY")
    print("=" * 60)
    print(f"  {'Presentation':<30}  {'Accuracy':>8}  {'Missed':>6}")
    print(f"  {'-'*30}  {'-'*8}  {'-'*6}")
    total_steps = 0
    total_correct = 0
    for r in results:
        pct = f"{r.inference_accuracy * 100:.1f}%"
        missed = len(r.missed_slides)
        name = r.presentation_name[:30]
        print(f"  {name:<30}  {pct:>8}  {missed:>6}")
        total_steps += r.total_steps
        total_correct += r.correct_steps
    overall = total_correct / total_steps if total_steps else 0.0
    print(f"  {'─'*30}  {'─'*8}  {'─'*6}")
    print(f"  {'OVERALL':<30}  {overall * 100:.1f}%")
    print()
