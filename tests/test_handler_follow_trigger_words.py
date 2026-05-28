"""
Unit tests for FollowTriggerWordsHandler.
No network, audio hardware, or Whisper model required.
"""

from collections import deque
from unittest.mock import MagicMock, patch

from propresenter_speech.command_parser import Command, CommandType
from propresenter_speech.handlers.follow_trigger_words import FollowTriggerWordsHandler
from propresenter_speech.predictors import TranscriptionResult


def _make_handler(**kwargs) -> FollowTriggerWordsHandler:
    follower = MagicMock()
    follower.has_triggers = True
    follower.trigger_words = ["grace"]
    follower.matches.return_value = False
    follower.refresh_after_advance.return_value = False
    follower.validate.return_value = (True, "")

    defaults = {
        "pro_controller": MagicMock(),
        "command_parser": MagicMock(),
        "slide_follower": follower,
        "verbose": False,
    }
    defaults.update(kwargs)
    h = FollowTriggerWordsHandler(**defaults)
    h.pro_controller.next_slide.return_value = True
    return h


def _buf() -> deque:
    return deque(maxlen=200)


def _result(text: str) -> TranscriptionResult:
    return TranscriptionResult(text=text, word_buffer=_buf())


class TestTriggerMatching:
    def test_match_calls_next_slide(self):
        h = _make_handler()
        h.slide_follower.matches.return_value = True
        h.command_parser.parse.return_value = Command(CommandType.UNKNOWN)
        h.on_prediction(_result("amazing grace"))
        h.pro_controller.next_slide.assert_called_once()

    def test_match_calls_refresh_after_advance(self):
        h = _make_handler()
        h.slide_follower.matches.return_value = True
        h.command_parser.parse.return_value = Command(CommandType.UNKNOWN)
        h.on_prediction(_result("amazing grace"))
        h.slide_follower.refresh_after_advance.assert_called_once()

    def test_no_match_does_not_advance(self):
        h = _make_handler()
        h.slide_follower.matches.return_value = False
        h.command_parser.parse.return_value = Command(CommandType.UNKNOWN)
        h.on_prediction(_result("random words"))
        h.pro_controller.next_slide.assert_not_called()

    def test_match_prints_follow_indicator(self, capsys):
        h = _make_handler()
        h.slide_follower.matches.return_value = True
        h.command_parser.parse.return_value = Command(CommandType.UNKNOWN)
        h.on_prediction(_result("grace"))
        assert "follow-trigger-words" in capsys.readouterr().out.lower()

    def test_no_triggers_calls_refresh_before_matching(self):
        h = _make_handler()
        h.slide_follower.has_triggers = False
        h.slide_follower.matches.return_value = False
        h.command_parser.parse.return_value = Command(CommandType.UNKNOWN)
        h.on_prediction(_result("some words"))
        h.slide_follower.refresh.assert_called_once()

    def test_cooldown_prevents_immediate_re_advance(self):
        h = _make_handler()
        h.slide_follower.matches.return_value = True
        h.command_parser.parse.return_value = Command(CommandType.UNKNOWN)

        # First call advances and records timestamp
        h.on_prediction(_result("grace"))
        h.pro_controller.next_slide.reset_mock()

        # Second call within cooldown window — should not advance
        h.on_prediction(_result("grace"))
        h.pro_controller.next_slide.assert_not_called()

    def test_advance_after_cooldown_expires(self):
        h = _make_handler()
        h.slide_follower.matches.return_value = True
        h.command_parser.parse.return_value = Command(CommandType.UNKNOWN)

        h.on_prediction(_result("grace"))
        h.pro_controller.next_slide.reset_mock()

        # Expire the cooldown
        h._last_advance = 0.0
        h.slide_follower.has_triggers = True
        h.slide_follower.refresh_after_advance.return_value = False

        h.on_prediction(_result("grace"))
        h.pro_controller.next_slide.assert_called_once()


class TestExplicitCommands:
    def test_next_slide_command(self):
        h = _make_handler()
        h.command_parser.parse.return_value = Command(CommandType.NEXT_SLIDE)
        h.on_prediction(_result("next slide"))
        h.pro_controller.next_slide.assert_called_once()

    def test_explicit_command_refreshes_follower(self):
        h = _make_handler()
        h.command_parser.parse.return_value = Command(CommandType.PREVIOUS_SLIDE)
        h.pro_controller.previous_slide.return_value = True
        h.on_prediction(_result("previous slide"))
        h.slide_follower.refresh_after_advance.assert_called_once_with(delta=-1)

    def test_go_to_slide_command(self):
        h = _make_handler()
        h.command_parser.parse.return_value = Command(CommandType.GO_TO_SLIDE, slide_number=3)
        h.pro_controller.go_to_slide.return_value = True
        h.on_prediction(_result("go to slide three"))
        h.pro_controller.go_to_slide.assert_called_once_with(3)
        h.slide_follower.refresh_to_slide.assert_called_once_with(2)


class TestOnStartup:
    def test_exits_when_validation_fails(self):
        h = _make_handler()
        h.slide_follower.validate.return_value = (False, "no presentation active")
        import pytest
        with pytest.raises(SystemExit):
            h.on_startup()

    def test_calls_refresh_on_success(self):
        h = _make_handler()
        h.on_startup()
        h.slide_follower.refresh.assert_called_once()
