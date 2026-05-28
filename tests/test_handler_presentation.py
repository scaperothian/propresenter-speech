"""
Unit tests for PresentationHandler.
No network, audio hardware, or Whisper model required.
"""

import pytest
from collections import deque
from unittest.mock import MagicMock

from propresenter_speech.command_parser import Command, CommandType
from propresenter_speech.handlers.presentation import PresentationHandler
from propresenter_speech.predictors import TranscriptionResult


def _make_handler(**kwargs) -> PresentationHandler:
    defaults = {
        "pro_controller": MagicMock(),
        "command_parser": MagicMock(),
        "verbose": False,
    }
    defaults.update(kwargs)
    return PresentationHandler(**defaults)


def _buf() -> deque:
    return deque(maxlen=200)


def _result(text: str) -> TranscriptionResult:
    return TranscriptionResult(text=text, word_buffer=_buf())


class TestOnPrediction:
    def test_next_slide(self):
        h = _make_handler()
        h.command_parser.parse.return_value = Command(CommandType.NEXT_SLIDE)
        h.pro_controller.next_slide.return_value = True
        h.on_prediction(_result("next slide"))
        h.pro_controller.next_slide.assert_called_once()

    def test_previous_slide(self):
        h = _make_handler()
        h.command_parser.parse.return_value = Command(CommandType.PREVIOUS_SLIDE)
        h.pro_controller.previous_slide.return_value = True
        h.on_prediction(_result("previous slide"))
        h.pro_controller.previous_slide.assert_called_once()

    def test_go_to_slide(self):
        h = _make_handler()
        h.command_parser.parse.return_value = Command(CommandType.GO_TO_SLIDE, slide_number=5)
        h.pro_controller.go_to_slide.return_value = True
        h.on_prediction(_result("go to slide five"))
        h.pro_controller.go_to_slide.assert_called_once_with(5)

    def test_unknown_command_does_not_call_pro_controller(self):
        h = _make_handler()
        h.command_parser.parse.return_value = Command(CommandType.UNKNOWN)
        h.on_prediction(_result("random noise"))
        h.pro_controller.next_slide.assert_not_called()
        h.pro_controller.previous_slide.assert_not_called()
        h.pro_controller.go_to_slide.assert_not_called()


class TestExecuteOutput:
    def test_next_slide_success(self, capsys):
        h = _make_handler()
        h.pro_controller.next_slide.return_value = True
        h._execute(Command(CommandType.NEXT_SLIDE))
        assert "Next slide" in capsys.readouterr().out

    def test_next_slide_failure(self, capsys):
        h = _make_handler()
        h.pro_controller.next_slide.return_value = False
        h._execute(Command(CommandType.NEXT_SLIDE))
        assert "Failed" in capsys.readouterr().out

    def test_previous_slide_success(self, capsys):
        h = _make_handler()
        h.pro_controller.previous_slide.return_value = True
        h._execute(Command(CommandType.PREVIOUS_SLIDE))
        assert "Previous slide" in capsys.readouterr().out

    def test_go_to_slide_shows_number(self, capsys):
        h = _make_handler()
        h.pro_controller.go_to_slide.return_value = True
        h._execute(Command(CommandType.GO_TO_SLIDE, slide_number=7))
        assert "7" in capsys.readouterr().out

    def test_go_to_slide_failure(self, capsys):
        h = _make_handler()
        h.pro_controller.go_to_slide.return_value = False
        h._execute(Command(CommandType.GO_TO_SLIDE, slide_number=7))
        assert "Failed" in capsys.readouterr().out
