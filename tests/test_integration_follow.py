"""
Integration tests for follow mode.

TestFollowModeTriggerOrder — pure unit tests, no audio or Whisper required.
TestFollowModeAudio — end-to-end tests using the real WAV file and Whisper.
  Skipped automatically when the audio file is absent (e.g. in CI).

Run with:
  poetry run pytest tests/test_integration_follow.py -v
"""

import threading
from pathlib import Path

import numpy as np
import pytest

from propresenter_speech.audio_capture import AudioFileCapture
from propresenter_speech.slide_follower import SlideFollower
from propresenter_speech.transcriber import Transcriber

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

AUDIO_FILE = Path(__file__).parent.parent / "audio" / "pledge_of_allegiance.wav"

PLEDGE_SLIDES = [
    {"text": "I pledge allegiance to the flag"},
    {"text": "Of the United States of America"},
    {"text": "And to the republic for which it stands"},
    {"text": "One Nation, Under God, Indivisible"},
    {"text": "With Liberty and Justice for All"},
]

PLEDGE_DETAILS = {"presentation": {"groups": [{"slides": PLEDGE_SLIDES}]}}

# Expected last word of each slide (trigger_word_count=1)
EXPECTED_TRIGGERS = ["flag", "america", "stands", "indivisible", "all"]


class FakeProPresenter:
    """Minimal ProPresenter stub simulating the 5-slide pledge presentation."""

    _UUID = "pledge-test-uuid"

    def __init__(self):
        self._slide_index = 0
        self.advance_count = 0
        self.advance_log: list[int] = []  # slide indices at time of each advance

    def get_active_presentation_uuid(self) -> str:
        return self._UUID

    def get_presentation_details(self, uuid: str) -> dict:
        return PLEDGE_DETAILS

    def get_slide_index(self) -> int:
        return self._slide_index

    def find_slides(self, details) -> list:
        return PLEDGE_SLIDES

    def next_slide(self) -> bool:
        if self._slide_index < len(PLEDGE_SLIDES) - 1:
            self.advance_log.append(self._slide_index)
            self._slide_index += 1
            self.advance_count += 1
            return True
        return False

    def get_status(self):
        return None


# ---------------------------------------------------------------------------
# Pure unit: trigger word sequence via refresh_after_advance
# ---------------------------------------------------------------------------

class TestFollowModeTriggerOrder:
    """Verifies that refresh_after_advance navigates the slide list correctly."""

    def test_initial_trigger_is_first_slide_last_word(self):
        fake_pro = FakeProPresenter()
        follower = SlideFollower(fake_pro, trigger_word_count=1)
        ok, reason = follower.validate()
        assert ok, reason
        follower.refresh()
        assert follower.trigger_words == ["flag"]

    def test_sequential_advances_yield_correct_triggers(self):
        fake_pro = FakeProPresenter()
        follower = SlideFollower(fake_pro, trigger_word_count=1)
        ok, _ = follower.validate()
        assert ok
        follower.refresh()

        for i, expected in enumerate(EXPECTED_TRIGGERS):
            assert follower.trigger_words == [expected], (
                f"Slide {i}: expected trigger {expected!r}, got {follower.trigger_words}"
            )
            if i < len(EXPECTED_TRIGGERS) - 1:
                fake_pro.next_slide()
                follower.refresh_after_advance()

    def test_refresh_after_advance_past_end_clears_triggers(self):
        fake_pro = FakeProPresenter()
        follower = SlideFollower(fake_pro, trigger_word_count=1)
        ok, _ = follower.validate()
        assert ok
        follower.refresh()

        # Advance through all slides
        for _ in range(len(PLEDGE_SLIDES) - 1):
            fake_pro.next_slide()
            follower.refresh_after_advance()

        # One more should exhaust the presentation
        result = follower.refresh_after_advance()
        assert result is False
        assert follower.trigger_words == []
        assert not follower.has_triggers

    def test_two_word_trigger_uses_last_two_words(self):
        fake_pro = FakeProPresenter()
        follower = SlideFollower(fake_pro, trigger_word_count=2)
        ok, _ = follower.validate()
        assert ok
        follower.refresh()
        # "I pledge allegiance to the flag" → last 2 words: ["the", "flag"]
        assert follower.trigger_words == ["the", "flag"]

    def test_no_api_call_for_slide_index_after_advance(self):
        """refresh_after_advance must not call get_slide_index (race-condition guard)."""
        fake_pro = FakeProPresenter()
        import unittest.mock as mock
        fake_pro.get_slide_index = mock.MagicMock(return_value=0)

        follower = SlideFollower(fake_pro, trigger_word_count=1)
        follower.validate()
        follower.refresh()
        call_count_after_refresh = fake_pro.get_slide_index.call_count

        fake_pro.next_slide()
        follower.refresh_after_advance()

        # get_slide_index should NOT have been called again
        assert fake_pro.get_slide_index.call_count == call_count_after_refresh


# ---------------------------------------------------------------------------
# End-to-end: real audio + real Whisper
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not AUDIO_FILE.exists(),
    reason=f"Audio file not found: {AUDIO_FILE}",
)
class TestFollowModeAudio:
    """
    Drives the real VAD + Whisper pipeline against the pledge WAV file and
    verifies that follow mode advances at least one slide.

    Uses the 'tiny' Whisper model for speed; accuracy is sufficient to detect
    "flag" which ends the first slide.
    """

    @pytest.fixture(scope="class")
    def transcriber(self):
        t = Transcriber("tiny")
        t.load()
        return t

    @pytest.fixture(scope="class")
    def audio_segments(self) -> list[np.ndarray]:
        segments: list[np.ndarray] = []
        capture = AudioFileCapture(file_path=str(AUDIO_FILE))
        capture.start(lambda seg: segments.append(seg))
        capture._thread.join(timeout=30)
        return segments

    def _run_follow(self, transcriber: Transcriber, segments: list[np.ndarray]) -> FakeProPresenter:
        fake_pro = FakeProPresenter()
        follower = SlideFollower(fake_pro, trigger_word_count=1)
        ok, reason = follower.validate()
        assert ok, reason
        follower.refresh()

        for segment in segments:
            text = transcriber.transcribe(segment)
            if not text.strip():
                continue
            while follower.has_triggers and follower.matches(text):
                if not fake_pro.next_slide():
                    break
                if not follower.refresh_after_advance():
                    break

        return fake_pro

    def test_advances_at_least_one_slide(self, transcriber, audio_segments):
        fake_pro = self._run_follow(transcriber, audio_segments)
        assert fake_pro.advance_count >= 1, (
            "Expected at least one slide advance from pledge audio"
        )

    def test_first_advance_triggered_from_slide_zero(self, transcriber, audio_segments):
        fake_pro = self._run_follow(transcriber, audio_segments)
        assert fake_pro.advance_log, "No advances recorded"
        assert fake_pro.advance_log[0] == 0, (
            f"First advance should be from slide 0, got {fake_pro.advance_log[0]}"
        )
