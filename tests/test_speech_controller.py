"""
Unit tests for SpeechController.

All external dependencies (ProPresenterController, Transcriber, CommandParser,
AudioCapture) are replaced with MagicMocks — no network, audio hardware, or
Whisper model required.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, call, patch

from propresenter_speech.command_parser import Command, CommandType
from propresenter_speech.modes import Mode
from propresenter_speech.speech_controller import SpeechController


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_controller(**kwargs) -> SpeechController:
    defaults = {
        "transcriber": MagicMock(),
        "command_parser": MagicMock(),
        "pro_controller": MagicMock(),
        "audio_capture": MagicMock(),
        "verbose": False,
    }
    defaults.update(kwargs)
    return SpeechController(**defaults)


@pytest.fixture
def ctrl():
    return _make_controller()


def _audio() -> np.ndarray:
    return np.zeros(16_000, dtype=np.float32)


# ---------------------------------------------------------------------------
# _handle_segment — the core dispatch logic
# ---------------------------------------------------------------------------

class TestHandleSegment:
    def test_next_slide_calls_pro_controller(self, ctrl):
        ctrl.transcriber.transcribe.return_value = "next slide"
        ctrl.command_parser.parse.return_value = Command(CommandType.NEXT_SLIDE)
        ctrl.pro_controller.next_slide.return_value = True

        ctrl._handle_segment(_audio())

        ctrl.pro_controller.next_slide.assert_called_once()

    def test_previous_slide_calls_pro_controller(self, ctrl):
        ctrl.transcriber.transcribe.return_value = "previous slide"
        ctrl.command_parser.parse.return_value = Command(CommandType.PREVIOUS_SLIDE)
        ctrl.pro_controller.previous_slide.return_value = True

        ctrl._handle_segment(_audio())

        ctrl.pro_controller.previous_slide.assert_called_once()

    def test_go_to_slide_calls_pro_controller_with_number(self, ctrl):
        ctrl.transcriber.transcribe.return_value = "go to slide five"
        ctrl.command_parser.parse.return_value = Command(CommandType.GO_TO_SLIDE, slide_number=5)
        ctrl.pro_controller.go_to_slide.return_value = True

        ctrl._handle_segment(_audio())

        ctrl.pro_controller.go_to_slide.assert_called_once_with(5)

    def test_unknown_command_does_not_call_pro_controller(self, ctrl):
        ctrl.transcriber.transcribe.return_value = "some random noise"
        ctrl.command_parser.parse.return_value = Command(CommandType.UNKNOWN)

        ctrl._handle_segment(_audio())

        ctrl.pro_controller.next_slide.assert_not_called()
        ctrl.pro_controller.previous_slide.assert_not_called()
        ctrl.pro_controller.go_to_slide.assert_not_called()

    def test_empty_transcription_skips_parsing(self, ctrl):
        ctrl.transcriber.transcribe.return_value = ""

        ctrl._handle_segment(_audio())

        ctrl.command_parser.parse.assert_not_called()
        ctrl.pro_controller.next_slide.assert_not_called()

    def test_whitespace_only_transcription_skips_parsing(self, ctrl):
        ctrl.transcriber.transcribe.return_value = "   "

        ctrl._handle_segment(_audio())

        ctrl.command_parser.parse.assert_not_called()


# ---------------------------------------------------------------------------
# _execute — per-command output and pro_controller calls
# ---------------------------------------------------------------------------

class TestExecute:
    def test_next_slide_success_message(self, ctrl, capsys):
        ctrl.pro_controller.next_slide.return_value = True
        ctrl._execute(Command(CommandType.NEXT_SLIDE), "next slide")
        assert "Next slide" in capsys.readouterr().out

    def test_next_slide_failure_message(self, ctrl, capsys):
        ctrl.pro_controller.next_slide.return_value = False
        ctrl._execute(Command(CommandType.NEXT_SLIDE), "next slide")
        assert "Failed" in capsys.readouterr().out

    def test_previous_slide_success_message(self, ctrl, capsys):
        ctrl.pro_controller.previous_slide.return_value = True
        ctrl._execute(Command(CommandType.PREVIOUS_SLIDE), "previous slide")
        assert "Previous slide" in capsys.readouterr().out

    def test_previous_slide_failure_message(self, ctrl, capsys):
        ctrl.pro_controller.previous_slide.return_value = False
        ctrl._execute(Command(CommandType.PREVIOUS_SLIDE), "previous slide")
        assert "Failed" in capsys.readouterr().out

    def test_go_to_slide_success_message(self, ctrl, capsys):
        ctrl.pro_controller.go_to_slide.return_value = True
        ctrl._execute(Command(CommandType.GO_TO_SLIDE, slide_number=7), "slide 7")
        out = capsys.readouterr().out
        assert "7" in out

    def test_go_to_slide_failure_message(self, ctrl, capsys):
        ctrl.pro_controller.go_to_slide.return_value = False
        ctrl._execute(Command(CommandType.GO_TO_SLIDE, slide_number=7), "slide 7")
        assert "Failed" in capsys.readouterr().out

    def test_unknown_command_produces_no_output(self, ctrl, capsys):
        ctrl._execute(Command(CommandType.UNKNOWN), "blah blah")
        assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# Verbose mode
# ---------------------------------------------------------------------------

class TestVerboseMode:
    def test_verbose_prints_transcription(self, capsys):
        ctrl = _make_controller(verbose=True)
        ctrl.transcriber.transcribe.return_value = "next slide"
        ctrl.command_parser.parse.return_value = Command(CommandType.NEXT_SLIDE)
        ctrl.pro_controller.next_slide.return_value = True

        ctrl._handle_segment(_audio())

        out = capsys.readouterr().out
        assert "next slide" in out

    def test_non_verbose_omits_transcription(self, ctrl, capsys):
        ctrl.transcriber.transcribe.return_value = "next slide"
        ctrl.command_parser.parse.return_value = Command(CommandType.NEXT_SLIDE)
        ctrl.pro_controller.next_slide.return_value = True

        ctrl._handle_segment(_audio())

        out = capsys.readouterr().out
        assert "heard:" not in out


# ---------------------------------------------------------------------------
# _enqueue_segment — queue overflow behaviour
# ---------------------------------------------------------------------------

class TestEnqueueSegment:
    def test_enqueue_puts_segment_in_queue(self, ctrl):
        audio = _audio()
        ctrl._enqueue_segment(audio)
        assert not ctrl._segment_queue.empty()

    def test_enqueue_drops_oldest_when_full(self, ctrl):
        first = np.ones(100, dtype=np.float32)
        second = np.ones(100, dtype=np.float32) * 2

        # Fill queue beyond capacity
        for _ in range(ctrl._segment_queue.maxsize + 1):
            ctrl._enqueue_segment(first)

        ctrl._enqueue_segment(second)

        # Queue should still be within its size limit
        assert ctrl._segment_queue.qsize() <= ctrl._segment_queue.maxsize


# ---------------------------------------------------------------------------
# Follow mode — _handle_follow
# ---------------------------------------------------------------------------

def _make_follow_controller(**kwargs) -> SpeechController:
    follower = MagicMock()
    follower.has_triggers = True
    follower.trigger_words = ["grace"]
    follower.matches.return_value = False
    defaults = {
        "transcriber": MagicMock(),
        "command_parser": MagicMock(),
        "pro_controller": MagicMock(),
        "audio_capture": MagicMock(),
        "mode": Mode.FOLLOW,
        "slide_follower": follower,
        "verbose": False,
    }
    defaults.update(kwargs)
    return SpeechController(**defaults)


class TestFollowMode:
    def test_trigger_match_calls_next_slide(self):
        ctrl = _make_follow_controller()
        ctrl.slide_follower.matches.return_value = True
        ctrl.pro_controller.next_slide.return_value = True
        ctrl.command_parser.parse.return_value = Command(CommandType.UNKNOWN)
        ctrl.transcriber.transcribe.return_value = "amazing grace"

        ctrl._handle_segment(_audio())

        ctrl.pro_controller.next_slide.assert_called_once()

    def test_trigger_match_refreshes_follower(self):
        ctrl = _make_follow_controller()
        ctrl.slide_follower.matches.return_value = True
        ctrl.pro_controller.next_slide.return_value = True
        ctrl.command_parser.parse.return_value = Command(CommandType.UNKNOWN)
        ctrl.transcriber.transcribe.return_value = "amazing grace"

        ctrl._handle_segment(_audio())

        ctrl.slide_follower.refresh.assert_called_once()

    def test_no_trigger_match_does_not_advance(self):
        ctrl = _make_follow_controller()
        ctrl.slide_follower.matches.return_value = False
        ctrl.command_parser.parse.return_value = Command(CommandType.UNKNOWN)
        ctrl.transcriber.transcribe.return_value = "some random words"

        ctrl._handle_segment(_audio())

        ctrl.pro_controller.next_slide.assert_not_called()

    def test_explicit_command_works_in_follow_mode(self):
        ctrl = _make_follow_controller()
        ctrl.command_parser.parse.return_value = Command(CommandType.NEXT_SLIDE)
        ctrl.transcriber.transcribe.return_value = "next slide"
        ctrl.pro_controller.next_slide.return_value = True

        ctrl._handle_segment(_audio())

        ctrl.pro_controller.next_slide.assert_called_once()

    def test_explicit_command_refreshes_follower(self):
        ctrl = _make_follow_controller()
        ctrl.command_parser.parse.return_value = Command(CommandType.PREVIOUS_SLIDE)
        ctrl.transcriber.transcribe.return_value = "previous slide"
        ctrl.pro_controller.previous_slide.return_value = True

        ctrl._handle_segment(_audio())

        ctrl.slide_follower.refresh.assert_called_once()

    def test_go_to_slide_command_works_in_follow_mode(self):
        ctrl = _make_follow_controller()
        ctrl.command_parser.parse.return_value = Command(CommandType.GO_TO_SLIDE, slide_number=3)
        ctrl.transcriber.transcribe.return_value = "go to slide three"
        ctrl.pro_controller.go_to_slide.return_value = True

        ctrl._handle_segment(_audio())

        ctrl.pro_controller.go_to_slide.assert_called_once_with(3)
        ctrl.slide_follower.refresh.assert_called_once()

    def test_trigger_match_prints_follow_indicator(self, capsys):
        ctrl = _make_follow_controller()
        ctrl.slide_follower.matches.return_value = True
        ctrl.slide_follower.trigger_words = ["grace"]
        ctrl.pro_controller.next_slide.return_value = True
        ctrl.command_parser.parse.return_value = Command(CommandType.UNKNOWN)
        ctrl.transcriber.transcribe.return_value = "grace"

        ctrl._handle_segment(_audio())

        out = capsys.readouterr().out
        assert "follow" in out.lower()

    def test_follow_mode_retries_refresh_when_no_triggers(self):
        ctrl = _make_follow_controller()
        ctrl.slide_follower.has_triggers = False
        ctrl.slide_follower.matches.return_value = False
        ctrl.command_parser.parse.return_value = Command(CommandType.UNKNOWN)
        ctrl.transcriber.transcribe.return_value = "some words"

        ctrl._handle_segment(_audio())

        ctrl.slide_follower.refresh.assert_called_once()

    def test_none_slide_follower_does_not_crash_in_follow_mode(self, capsys):
        ctrl = _make_follow_controller(slide_follower=None)
        ctrl.command_parser.parse.return_value = Command(CommandType.UNKNOWN)
        ctrl.transcriber.transcribe.return_value = "some words"

        ctrl._handle_segment(_audio())  # should not raise

    def test_presentation_mode_ignores_trigger_words(self):
        ctrl = _make_controller()
        assert ctrl.mode == Mode.PRESENTATION
        ctrl.command_parser.parse.return_value = Command(CommandType.UNKNOWN)
        ctrl.transcriber.transcribe.return_value = "amazing grace"

        ctrl._handle_segment(_audio())

        ctrl.pro_controller.next_slide.assert_not_called()
