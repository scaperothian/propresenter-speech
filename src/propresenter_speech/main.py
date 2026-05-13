"""
CLI entry point for propresenter-speech.

Usage examples:
  propresenter-speech
  propresenter-speech --presentation "Sermon Slides"
  propresenter-speech --presentation "Worship" --library Songs --mode follow
  propresenter-speech --mode follow --trigger-words 2
  propresenter-speech --mode follow-enhanced
  propresenter-speech --host 192.168.1.10 --port 1025 --model small --verbose
  propresenter-speech --list-devices
"""

import argparse
import logging
import sys
from pathlib import Path

from propresenter_slides.main import ProPresenterController

from .audio_pipeline import (
    AudioPipeline,
    DEFAULT_POLL_INTERVAL,
    DEFAULT_WINDOW_SECONDS,
    list_input_devices,
)
from .command_parser import CommandParser
from .handlers import FollowEnhancedHandler, FollowHandler, PresentationHandler
from .handlers.follow_enhanced import (
    DEFAULT_CONTEXT_WORDS,
    DEFAULT_MIN_MARGIN,
    DEFAULT_SIMILARITY_THRESHOLD,
)
from .modes import Mode
from .slide_embedder import SlideEmbedder
from .slide_follower import SlideFollower
from .transcriber import Transcriber


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="propresenter-speech",
        description="Control ProPresenter slides with your voice using Whisper ASR.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    conn = parser.add_argument_group("ProPresenter connection")
    conn.add_argument("--host", default="localhost", help="ProPresenter hostname or IP")
    conn.add_argument("--port", type=int, default=1025, help="ProPresenter API port")
    conn.add_argument("--timeout", type=int, default=5, help="HTTP request timeout (seconds)")

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
        help="(follow mode) number of words from the end of the slide text to use as trigger",
    )
    mode_grp.add_argument(
        "--context-words",
        type=int,
        default=DEFAULT_CONTEXT_WORDS,
        dest="context_words",
        metavar="N",
        help="(follow-enhanced) recent spoken words to form the query n-gram",
    )
    mode_grp.add_argument(
        "--similarity-threshold",
        type=float,
        default=DEFAULT_SIMILARITY_THRESHOLD,
        dest="similarity_threshold",
        metavar="FLOAT",
        help="(follow-enhanced) minimum hybrid score to trigger a slide cue",
    )
    mode_grp.add_argument(
        "--min-margin",
        type=float,
        default=DEFAULT_MIN_MARGIN,
        dest="min_margin",
        metavar="FLOAT",
        help="(follow-enhanced) minimum gap between best and second-best score to trigger",
    )

    whisper_grp = parser.add_argument_group("Whisper ASR")
    whisper_grp.add_argument(
        "--model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (smaller = faster, larger = more accurate)",
    )

    audio_grp = parser.add_argument_group("Audio pipeline")
    audio_grp.add_argument(
        "--device",
        type=int,
        default=None,
        help="Input audio device index (see --list-devices; default: system default)",
    )
    audio_grp.add_argument(
        "--window-seconds",
        type=float,
        default=DEFAULT_WINDOW_SECONDS,
        dest="window_seconds",
        metavar="SECS",
        help="Rolling audio window length fed to Whisper",
    )
    audio_grp.add_argument(
        "--poll-interval",
        type=float,
        default=DEFAULT_POLL_INTERVAL,
        dest="poll_interval",
        metavar="SECS",
        help="Seconds between Whisper inference calls",
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

    parser.add_argument("--verbose", action="store_true", help="Print transcribed text to stdout")
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        dest="log_level",
        help="Logging verbosity",
    )

    return parser


def _build_follow_enhanced_handler(pro: ProPresenterController, args) -> FollowEnhancedHandler:
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

    embedder = SlideEmbedder()
    embedder.load()
    embedder.build(slide_texts, slide_indices=slide_indices)
    print("Embeddings ready.")

    return FollowEnhancedHandler(
        pro_controller=pro,
        slide_embedder=embedder,
        context_words=args.context_words,
        similarity_threshold=args.similarity_threshold,
        min_margin=args.min_margin,
        verbose=args.verbose,
    )


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

    pro = ProPresenterController(host=args.host, port=args.port, timeout=args.timeout)
    if pro.get_status() is None:
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
        handler = _build_follow_enhanced_handler(pro, args)
    elif mode == Mode.FOLLOW:
        handler = FollowHandler(
            pro_controller=pro,
            command_parser=CommandParser(),
            slide_follower=SlideFollower(pro, trigger_word_count=args.trigger_words),
            verbose=args.verbose,
        )
    else:
        handler = PresentationHandler(
            pro_controller=pro,
            command_parser=CommandParser(),
            verbose=args.verbose,
        )

    print("Loading Whisper model — this may take a moment on first run…")
    transcriber = Transcriber(model_name=args.model)
    transcriber.load()
    print("Whisper ready.")

    AudioPipeline(
        transcriber=transcriber,
        handler=handler,
        device=args.device,
        audio_file=args.audio_file,
        window_seconds=args.window_seconds,
        poll_interval=args.poll_interval,
        verbose=args.verbose,
    ).run()


if __name__ == "__main__":
    main()
