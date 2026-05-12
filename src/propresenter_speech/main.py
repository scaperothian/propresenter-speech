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
from .modes import Mode
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
        choices=["presentation", "follow"],
        help=(
            "presentation: respond to explicit voice commands only. "
            "follow: also auto-advance when the last word(s) of the active slide are heard."
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
