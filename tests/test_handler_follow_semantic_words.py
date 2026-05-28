"""
Unit tests for FollowSemanticWordsHandler.
No network, audio hardware, or Whisper model required.
"""

import pytest
from collections import deque
from unittest.mock import MagicMock

from propresenter_speech.handlers.follow_semantic_words import FollowSemanticWordsHandler
from propresenter_speech.predictors import TranscriptionResult


def _make_handler(**kwargs) -> FollowSemanticWordsHandler:
    defaults = {
        "pro_controller": MagicMock(),
        "slide_embedder": MagicMock(),
        "context_words": 3,
        "similarity_threshold": 0.4,
        "min_margin": 0.15,
        "verbose": False,
    }
    defaults.update(kwargs)
    h = FollowSemanticWordsHandler(**defaults)
    h.pro_controller.go_to_slide.return_value = True
    return h


def _buf(*words: str) -> deque:
    d: deque = deque(maxlen=200)
    d.extend(words)
    return d


def _result(text: str, *words: str) -> TranscriptionResult:
    return TranscriptionResult(text=text, word_buffer=_buf(*words))


class TestOnPrediction:
    def test_cues_slide_when_confidence_above_threshold(self):
        h = _make_handler()
        h.slide_embedder.find_slide_with_margin.return_value = (0, 0.85, 0.30)
        h.on_prediction(_result("ignored", "allegiance", "to", "the", "flag"))
        h.pro_controller.go_to_slide.assert_called_once_with(1)

    def test_no_cue_when_confidence_and_margin_both_low(self):
        h = _make_handler(similarity_threshold=0.7, min_margin=0.2)
        h.slide_embedder.find_slide_with_margin.return_value = (2, 0.5, 0.05)
        h.on_prediction(_result("ignored", "some", "spoken", "words"))
        h.pro_controller.go_to_slide.assert_not_called()

    def test_cues_when_below_threshold_but_margin_sufficient(self):
        h = _make_handler(similarity_threshold=0.4, min_margin=0.15)
        h.slide_embedder.find_slide_with_margin.return_value = (1, 0.35, 0.20)
        h.on_prediction(_result("ignored", "everywhere", "that", "mary"))
        h.pro_controller.go_to_slide.assert_called_once_with(2)

    def test_no_cue_when_same_slide_already_active(self):
        h = _make_handler()
        h._current_slide_idx = 1
        h.slide_embedder.find_slide_with_margin.return_value = (1, 0.9, 0.40)
        h.on_prediction(_result("ignored", "united", "states", "america"))
        h.pro_controller.go_to_slide.assert_not_called()

    def test_cues_new_slide_when_match_changes(self):
        h = _make_handler()
        h._current_slide_idx = 0
        h.slide_embedder.find_slide_with_margin.return_value = (1, 0.88, 0.30)
        h.on_prediction(_result("ignored", "united", "states", "america"))
        h.pro_controller.go_to_slide.assert_called_once_with(2)
        assert h._current_slide_idx == 1

    def test_no_cue_when_fewer_than_two_context_words(self):
        h = _make_handler(context_words=3)
        h.slide_embedder.find_slide_with_margin.return_value = (0, 0.9, 0.5)
        h.on_prediction(_result("ignored", "flag"))
        h.pro_controller.go_to_slide.assert_not_called()

    def test_no_cue_when_slide_idx_negative(self):
        h = _make_handler()
        h.slide_embedder.find_slide_with_margin.return_value = (-1, 0.0, 0.0)
        h.on_prediction(_result("ignored", "some", "spoken", "words"))
        h.pro_controller.go_to_slide.assert_not_called()

    def test_failed_go_to_slide_does_not_update_current_idx(self, capsys):
        h = _make_handler()
        h.pro_controller.go_to_slide.return_value = False
        h.slide_embedder.find_slide_with_margin.return_value = (0, 0.9, 0.5)
        h.on_prediction(_result("ignored", "allegiance", "flag", "republic"))
        assert h._current_slide_idx is None
        assert "Failed" in capsys.readouterr().out

    def test_verbose_prints_query_info(self, capsys):
        h = _make_handler(verbose=True)
        h.slide_embedder.find_slide_with_margin.return_value = (0, 0.85, 0.30)
        h.on_prediction(_result("ignored", "allegiance", "to", "flag"))
        out = capsys.readouterr().out
        assert "query" in out
