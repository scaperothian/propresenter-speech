"""
Unit tests for FollowHandler.
No network, audio hardware, or Whisper model required.
"""

from collections import deque
from unittest.mock import MagicMock, patch

from propresenter_speech.command_parser import Command, CommandType
from propresenter_speech.handlers.follow import FollowHandler


def _make_handler(**kwargs) -> FollowHandler:
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
    h = FollowHandler(**defaults)
    h.pro_controller.next_slide.return_value = True
    return h


def _buf() -> deque:
    return deque(maxlen=200)


class TestTriggerMatching:
    def test_match_calls_next_slide(self):
        h = _make_handler()
        h.slide_follower.matches.return_value = True
        h.command_parser.parse.return_value = Command(CommandType.UNKNOWN)
        h.on_transcription("amazing grace", _buf())
        h.pro_controller.next_slide.assert_called_once()

    def test_match_calls_refresh_after_advance(self):
        h = _make_handler()
        h.slide_follower.matches.return_value = True
        h.command_parser.parse.return_value = Command(CommandType.UNKNOWN)
        h.on_transcription("amazing grace", _buf())
        h.slide_follower.refresh_after_advance.assert_called_once()

    def test_no_match_does_not_advance(self):
        h = _make_handler()
        h.slide_follower.matches.return_value = False
        h.command_parser.parse.return_value = Command(CommandType.UNKNOWN)
        h.on_transcription("random words", _buf())
        h.pro_controller.next_slide.assert_not_called()

    def test_match_prints_follow_indicator(self, capsys):
        h = _make_handler()
        h.slide_follower.matches.return_value = True
        h.command_parser.parse.return_value = Command(CommandType.UNKNOWN)
        h.on_transcription("grace", _buf())
        assert "follow" in capsys.readouterr().out.lower()

    def test_no_triggers_calls_refresh_before_matching(self):
        h = _make_handler()
        h.slide_follower.has_triggers = False
        h.slide_follower.matches.return_value = False
        h.command_parser.parse.return_value = Command(CommandType.UNKNOWN)
        h.on_transcription("some words", _buf())
        h.slide_follower.refresh.assert_called_once()

    def test_cooldown_prevents_immediate_re_advance(self):
        h = _make_handler()
        h.slide_follower.matches.return_value = True
        h.command_parser.parse.return_value = Command(CommandType.UNKNOWN)

        # First call advances and records timestamp
        h.on_transcription("grace", _buf())
        h.pro_controller.next_slide.reset_mock()

        # Second call within cooldown window — should not advance
        h.on_transcription("grace", _buf())
        h.pro_controller.next_slide.assert_not_called()

    def test_advance_after_cooldown_expires(self):
        h = _make_handler()
        h.slide_follower.matches.return_value = True
        h.command_parser.parse.return_value = Command(CommandType.UNKNOWN)

        h.on_transcription("grace", _buf())
        h.pro_controller.next_slide.reset_mock()

        # Expire the cooldown
        h._last_advance = 0.0
        h.slide_follower.has_triggers = True
        h.slide_follower.refresh_after_advance.return_value = False

        h.on_transcription("grace", _buf())
        h.pro_controller.next_slide.assert_called_once()


class TestExplicitCommands:
    def test_next_slide_command(self):
        h = _make_handler()
        h.command_parser.parse.return_value = Command(CommandType.NEXT_SLIDE)
        h.on_transcription("next slide", _buf())
        h.pro_controller.next_slide.assert_called_once()

    def test_explicit_command_refreshes_follower(self):
        h = _make_handler()
        h.command_parser.parse.return_value = Command(CommandType.PREVIOUS_SLIDE)
        h.pro_controller.previous_slide.return_value = True
        h.on_transcription("previous slide", _buf())
        h.slide_follower.refresh.assert_called_once()

    def test_go_to_slide_command(self):
        h = _make_handler()
        h.command_parser.parse.return_value = Command(CommandType.GO_TO_SLIDE, slide_number=3)
        h.pro_controller.go_to_slide.return_value = True
        h.on_transcription("go to slide three", _buf())
        h.pro_controller.go_to_slide.assert_called_once_with(3)
        h.slide_follower.refresh.assert_called_once()


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
