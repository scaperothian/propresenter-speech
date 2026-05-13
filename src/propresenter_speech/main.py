"""
CLI entry point for propresenter-speech.

Usage examples:
  propresenter-speech
  propresenter-speech --presentation "Sermon Slides"
  propresenter-speech --presentation "Worship" --library Songs --mode follow
  propresenter-speech --mode follow --trigger-words 2
  propresenter-speech --host 192.168.1.10 --port 1025 --model small --verbose
  propresenter-speech --list-devices
  propresenter-speech --device 2 --silence-threshold 0.02
"""

import argparse
import logging
import sys
from pathlib import Path

from propresenter_slides.main import ProPresenterController

from .audio_capture import AudioCapture, AudioFileCapture, list_input_devices
from .command_parser import CommandParser
from .follow_enhanced_controller import (
    FollowEnhancedController,
    DEFAULT_CONTEXT_WORDS,
    DEFAULT_MIN_MARGIN,
    DEFAULT_POLL_INTERVAL,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_WINDOW_SECONDS,
)
from .modes import Mode
from .slide_embedder import SlideEmbedder, DEFAULT_BM25_WEIGHT
from .slide_follower import SlideFollower
from .speech_controller import SpeechController
from .transcriber import Transcriber


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="propresenter-speech",
        description="Control ProPresenter slides with your voice using Whisper ASR.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ProPresenter connection
    conn = parser.add_argument_group("ProPresenter connection")
    conn.add_argument("--host", default="localhost", help="ProPresenter hostname or IP")
    conn.add_argument("--port", type=int, default=1025, help="ProPresenter API port")
    conn.add_argument("--timeout", type=int, default=5, help="HTTP request timeout (seconds)")

    # Presentation selection
    pres_grp = parser.add_argument_group("Presentation selection")
    pres_grp.add_argument(
        "--presentation",
        default=None,
        metavar="NAME",
        help="Activate a presentation by name before listening (case-insensitive substring match)",
    )
    pres_grp.add_argument(
        "--library",
        default="Default",
        metavar="NAME",
        help="Library to search when --presentation is given",
    )

    # Mode
    mode_grp = parser.add_argument_group("Operation mode")
    mode_grp.add_argument(
        "--mode",
        default="presentation",
        choices=["presentation", "follow", "follow-enhanced"],
        help=(
            "presentation: respond to explicit voice commands only. "
            "follow: also auto-advance when the last word(s) of the active slide are heard. "
            "follow-enhanced: semantic embedding search — cues whichever slide best matches recent speech."
        ),
    )
    mode_grp.add_argument(
        "--trigger-words",
        type=int,
        default=1,
        dest="trigger_words",
        metavar="N",
        help="(follow mode) number of words from the end of the slide text to use as trigger (default: 1)",
    )
    mode_grp.add_argument(
        "--context-words",
        type=int,
        default=DEFAULT_CONTEXT_WORDS,
        dest="context_words",
        metavar="N",
        help="(follow-enhanced) recent spoken words to form the query n-gram (default: %(default)s)",
    )
    mode_grp.add_argument(
        "--similarity-threshold",
        type=float,
        default=DEFAULT_SIMILARITY_THRESHOLD,
        dest="similarity_threshold",
        metavar="FLOAT",
        help="(follow-enhanced) minimum hybrid score to trigger a slide cue (default: %(default)s)",
    )
    mode_grp.add_argument(
        "--min-margin",
        type=float,
        default=DEFAULT_MIN_MARGIN,
        dest="min_margin",
        metavar="FLOAT",
        help=(
            "(follow-enhanced) minimum gap between best and second-best score to trigger "
            "even when below --similarity-threshold (default: %(default)s)"
        ),
    )
    mode_grp.add_argument(
        "--window-seconds",
        type=float,
        default=DEFAULT_WINDOW_SECONDS,
        dest="window_seconds",
        metavar="SECS",
        help="(follow-enhanced) rolling audio window length in seconds (default: %(default)s)",
    )
    mode_grp.add_argument(
        "--poll-interval",
        type=float,
        default=DEFAULT_POLL_INTERVAL,
        dest="poll_interval",
        metavar="SECS",
        help="(follow-enhanced) seconds between Whisper inference calls (default: %(default)s)",
    )
    mode_grp.add_argument(
        "--bm25-weight",
        type=float,
        default=DEFAULT_BM25_WEIGHT,
        dest="bm25_weight",
        metavar="FLOAT",
        help="(follow-enhanced) BM25 share of the hybrid score 0.0=dense-only, 1.0=BM25-only (default: %(default)s)",
    )

    # Whisper
    whisper_grp = parser.add_argument_group("Whisper ASR")
    whisper_grp.add_argument(
        "--model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (smaller = faster, larger = more accurate)",
    )

    # Audio
    audio_grp = parser.add_argument_group("Audio capture")
    audio_grp.add_argument(
        "--device",
        type=int,
        default=None,
        help="Input audio device index (see --list-devices; default: system default)",
    )
    audio_grp.add_argument(
        "--silence-threshold",
        type=float,
        default=0.01,
        dest="silence_threshold",
        help="RMS energy threshold (0–1) separating speech from silence",
    )
    audio_grp.add_argument(
        "--silence-duration",
        type=float,
        default=0.8,
        dest="silence_duration",
        help="Seconds of silence required to close a speech segment",
    )
    audio_grp.add_argument(
        "--list-devices",
        action="store_true",
        dest="list_devices",
        help="Print available input audio devices and exit",
    )
    audio_grp.add_argument(
        "--audio-file",
        default=None,
        metavar="PATH",
        dest="audio_file",
        help="Process an audio file instead of the microphone (WAV/FLAC/OGG; resampled to 16 kHz automatically)",
    )

    # Misc
    parser.add_argument("--verbose", action="store_true", help="Print transcribed text to stdout")
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        dest="log_level",
        help="Logging verbosity",
    )

    return parser


def _run_follow_enhanced(pro, args) -> None:
    """Build slide embeddings then hand off to FollowEnhancedController."""
    library_data = pro.get_library(args.library)
    if library_data is None:
        print(f"Error: Could not query '{args.library}' library.")
        sys.exit(1)

    uuid = pro.get_active_presentation_uuid()
    if not uuid:
        print("Error: Could not retrieve active presentation UUID. Make sure a presentation is active in ProPresenter.")
        sys.exit(1)

    details = pro.get_presentation_details(uuid)
    if not details:
        print(f"Error: Could not retrieve presentation details for UUID {uuid}.")
        sys.exit(1)

    slides = pro.find_slides(details)
    if not slides:
        print("Error: No slides found in the active presentation.")
        sys.exit(1)

    # Keep only slides that have text, but remember their original 0-based index
    # so go_to_slide() targets the right ProPresenter slide even when some are skipped.
    indexed_texts = [
        (i, s.get("text", "").strip())
        for i, s in enumerate(slides)
        if isinstance(s, dict) and s.get("text", "").strip()
    ]
    if not indexed_texts:
        print("Error: No slides with text found in the active presentation.")
        sys.exit(1)

    slide_indices = [i for i, _ in indexed_texts]
    slide_texts = [t for _, t in indexed_texts]
    print(f"Found {len(slide_texts)} slides with text (out of {len(slides)} total). Building embeddings…")
    embedder = SlideEmbedder(bm25_weight=args.bm25_weight)
    embedder.load()
    embedder.build(slide_texts, slide_indices=slide_indices)
    print("Embeddings ready.")

    print("Loading Whisper model — this may take a moment on first run…")
    transcriber = Transcriber(model_name=args.model)
    transcriber.load()
    print("Whisper ready.")

    controller = FollowEnhancedController(
        transcriber=transcriber,
        pro_controller=pro,
        slide_embedder=embedder,
        device=args.device,
        window_seconds=args.window_seconds,
        poll_interval=args.poll_interval,
        context_words=args.context_words,
        similarity_threshold=args.similarity_threshold,
        min_margin=args.min_margin,
        verbose=args.verbose,
    )
    controller.run()


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    parser = _build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.audio_file and not Path(args.audio_file).is_file():
        print(f"Error: Audio file not found: {args.audio_file}")
        sys.exit(1)

    if args.list_devices:
        devices = list_input_devices()
        if not devices:
            print("No input audio devices found.")
        else:
            print("Available input audio devices:")
            for d in devices:
                print(f"  [{d['index']}] {d['name']}  ({d['channels']} ch)")
        sys.exit(0)

    mode = Mode(args.mode)

    # Verify ProPresenter is reachable before loading the (large) Whisper model.
    pro = ProPresenterController(host=args.host, port=args.port, timeout=args.timeout)
    status = pro.get_status()
    if status is None:
        print(
            f"Error: Cannot reach ProPresenter at {args.host}:{args.port}.\n"
            "Make sure ProPresenter is running and the Network API is enabled."
        )
        sys.exit(1)
    print(f"Connected to ProPresenter at {args.host}:{args.port}")

    if args.presentation:
        library_data = pro.get_library(args.library)
        if library_data is None:
            print(f"Error: Could not query '{args.library}' library at {args.host}:{args.port}.")
            sys.exit(1)
        uuid = pro.find_presentation_uuid_by_name(args.presentation, library_data)
        if uuid is None:
            print(f"Error: Presentation '{args.presentation}' not found in '{args.library}' library.")
            sys.exit(1)
        if not pro.activate_presentation(uuid):
            print(f"Error: Failed to activate '{args.presentation}'.")
            sys.exit(1)
        print(f"Activated '{args.presentation}'.")

    if mode == Mode.FOLLOW_ENHANCED:
        _run_follow_enhanced(pro, args)
        return

    slide_follower = SlideFollower(pro, trigger_word_count=args.trigger_words) if mode == Mode.FOLLOW else None

    if args.audio_file:
        audio_source = AudioFileCapture(
            file_path=args.audio_file,
            silence_threshold=args.silence_threshold,
            silence_duration=args.silence_duration,
        )
    else:
        audio_source = AudioCapture(
            device=args.device,
            silence_threshold=args.silence_threshold,
            silence_duration=args.silence_duration,
        )

    controller = SpeechController(
        transcriber=Transcriber(model_name=args.model),
        command_parser=CommandParser(),
        pro_controller=pro,
        audio_capture=audio_source,
        mode=mode,
        slide_follower=slide_follower,
        verbose=args.verbose,
    )
    controller.run()


if __name__ == "__main__":
    main()
