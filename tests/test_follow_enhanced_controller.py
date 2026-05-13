"""
Unit tests for FollowEnhancedController.

All external I/O (sounddevice, Whisper, ProPresenter, fastembed) is mocked.
Tests exercise _transcribe_and_match directly to avoid threading complexity.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from propresenter_speech.follow_enhanced_controller import FollowEnhancedController


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_controller(**kwargs) -> FollowEnhancedController:
    defaults = {
        "transcriber": MagicMock(),
        "pro_controller": MagicMock(),
        "slide_embedder": MagicMock(),
        "context_words": 3,
        "similarity_threshold": 0.4,
        "min_margin": 0.15,
        "verbose": False,
    }
    defaults.update(kwargs)
    ctrl = FollowEnhancedController(**defaults)
    ctrl.pro_controller.go_to_slide.return_value = True
    return ctrl


def _audio() -> np.ndarray:
    return np.zeros(16_000, dtype=np.float32)


# ---------------------------------------------------------------------------
# _transcribe_and_match
# ---------------------------------------------------------------------------

class TestTranscribeAndMatch:
    def test_cues_slide_when_confidence_above_threshold(self):
        ctrl = _make_controller()
        ctrl.transcriber.transcribe.return_value = "allegiance to the flag"
        ctrl.slide_embedder.find_slide_with_margin.return_value = (0, 0.85, 0.30)

        ctrl._transcribe_and_match(_audio())

        ctrl.pro_controller.go_to_slide.assert_called_once_with(1)

    def test_no_cue_when_confidence_and_margin_both_low(self):
        ctrl = _make_controller(similarity_threshold=0.7, min_margin=0.2)
        ctrl.transcriber.transcribe.return_value = "some spoken words"
        ctrl.slide_embedder.find_slide_with_margin.return_value = (2, 0.5, 0.05)

        ctrl._transcribe_and_match(_audio())

        ctrl.pro_controller.go_to_slide.assert_not_called()

    def test_cues_when_below_threshold_but_margin_sufficient(self):
        # Score 0.35 < threshold 0.4, but margin 0.20 >= min_margin 0.15 → trigger
        ctrl = _make_controller(similarity_threshold=0.4, min_margin=0.15)
        ctrl.transcriber.transcribe.return_value = "everywhere that mary"
        ctrl.slide_embedder.find_slide_with_margin.return_value = (1, 0.35, 0.20)

        ctrl._transcribe_and_match(_audio())

        ctrl.pro_controller.go_to_slide.assert_called_once_with(2)

    def test_no_cue_when_same_slide_already_active(self):
        ctrl = _make_controller()
        ctrl._current_slide_idx = 1
        ctrl.transcriber.transcribe.return_value = "united states of america"
        ctrl.slide_embedder.find_slide_with_margin.return_value = (1, 0.9, 0.40)

        ctrl._transcribe_and_match(_audio())

        ctrl.pro_controller.go_to_slide.assert_not_called()

    def test_cues_new_slide_when_match_changes(self):
        ctrl = _make_controller()
        ctrl._current_slide_idx = 0
        ctrl.transcriber.transcribe.return_value = "united states of america"
        ctrl.slide_embedder.find_slide_with_margin.return_value = (1, 0.88, 0.30)

        ctrl._transcribe_and_match(_audio())

        ctrl.pro_controller.go_to_slide.assert_called_once_with(2)
        assert ctrl._current_slide_idx == 1

    def test_empty_transcription_skips_match(self):
        ctrl = _make_controller()
        ctrl.transcriber.transcribe.return_value = ""

        ctrl._transcribe_and_match(_audio())

        ctrl.slide_embedder.find_slide_with_margin.assert_not_called()
        ctrl.pro_controller.go_to_slide.assert_not_called()

    def test_whisper_busy_flag_cleared_after_call(self):
        ctrl = _make_controller()
        ctrl.transcriber.transcribe.return_value = "some text"
        ctrl.slide_embedder.find_slide_with_margin.return_value = (0, 0.3, 0.05)

        ctrl._transcribe_and_match(_audio())

        assert not ctrl._whisper_busy

    def test_whisper_busy_flag_cleared_even_on_exception(self):
        ctrl = _make_controller()
        ctrl.transcriber.transcribe.side_effect = RuntimeError("whisper failed")

        with pytest.raises(RuntimeError):
            ctrl._transcribe_and_match(_audio())

        assert not ctrl._whisper_busy

    def test_word_buffer_accumulates_across_calls(self):
        ctrl = _make_controller(context_words=3)
        ctrl.slide_embedder.find_slide_with_margin.return_value = (-1, 0.0, 0.0)

        ctrl.transcriber.transcribe.return_value = "pledge allegiance"
        ctrl._transcribe_and_match(_audio())

        ctrl.transcriber.transcribe.return_value = "to the flag"
        ctrl._transcribe_and_match(_audio())

        # find_slide_with_margin should have been called with the last 3 words
        last_call_arg = ctrl.slide_embedder.find_slide_with_margin.call_args_list[-1][0][0]
        assert last_call_arg == "to the flag"

    def test_no_cue_when_fewer_than_two_context_words(self):
        ctrl = _make_controller(context_words=3)
        ctrl.transcriber.transcribe.return_value = "flag"
        ctrl.slide_embedder.find_slide_with_margin.return_value = (0, 0.9, 0.5)

        ctrl._transcribe_and_match(_audio())

        ctrl.pro_controller.go_to_slide.assert_not_called()

    def test_verbose_prints_heard_and_query(self, capsys):
        ctrl = _make_controller(verbose=True)
        ctrl.transcriber.transcribe.return_value = "allegiance to the flag"
        ctrl.slide_embedder.find_slide_with_margin.return_value = (0, 0.85, 0.30)

        ctrl._transcribe_and_match(_audio())

        out = capsys.readouterr().out
        assert "heard" in out
        assert "query" in out

    def test_failed_go_to_slide_prints_error(self, capsys):
        ctrl = _make_controller()
        ctrl.pro_controller.go_to_slide.return_value = False
        ctrl.transcriber.transcribe.return_value = "allegiance flag"
        ctrl.slide_embedder.find_slide_with_margin.return_value = (0, 0.9, 0.5)

        ctrl._transcribe_and_match(_audio())

        assert "Failed" in capsys.readouterr().out
        assert ctrl._current_slide_idx is None  # not updated on failure
