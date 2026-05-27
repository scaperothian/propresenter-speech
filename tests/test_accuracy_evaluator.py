"""
Unit tests for accuracy_evaluator.py.
No real audio, Whisper model, or sentence-transformers required.
"""

import json
import math
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from speech_accuracy.evaluator import (
    AccuracyEvaluator,
    GroundTruthSlide,
    InferenceEvent,
    ground_truth_at,
    load_ground_truth,
    print_summary,
)
from propresenter_speech.audio_pipeline import SAMPLE_RATE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gt(*args) -> list[GroundTruthSlide]:
    """Build GroundTruthSlide list from (start, stop, text) triples."""
    return [
        GroundTruthSlide(idx=i, text=t, start_sec=a, stop_sec=b)
        for i, (a, b, t) in enumerate(args)
    ]


def _make_evaluator(ground_truth, window_seconds=2.0, poll_interval=1.0, context_words=3):
    transcriber = MagicMock()
    transcriber.transcribe.return_value = ""
    embedder = MagicMock()
    embedder.find_slide_with_margin.return_value = (-1, 0.0, 0.0)
    return AccuracyEvaluator(
        transcriber=transcriber,
        embedder=embedder,
        ground_truth=ground_truth,
        context_words=context_words,
        window_seconds=window_seconds,
        poll_interval=poll_interval,
    ), transcriber, embedder


def _silent_audio(seconds: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    return np.zeros(int(seconds * sr), dtype=np.float32)


# ---------------------------------------------------------------------------
# ground_truth_at
# ---------------------------------------------------------------------------

class TestGroundTruthAt:
    def test_returns_correct_slide_at_start(self):
        slides = _gt((0.0, 5.0, "slide A"), (5.0, 10.0, "slide B"))
        assert ground_truth_at(slides, 0.0, 10.0).idx == 0

    def test_returns_correct_slide_mid_range(self):
        slides = _gt((0.0, 5.0, "slide A"), (5.0, 10.0, "slide B"))
        assert ground_truth_at(slides, 7.0, 10.0).idx == 1

    def test_returns_none_before_first_slide(self):
        slides = _gt((2.0, 5.0, "slide A"))
        assert ground_truth_at(slides, 1.0, 10.0) is None

    def test_stop_exclusive(self):
        slides = _gt((0.0, 5.0, "slide A"), (5.0, 10.0, "slide B"))
        assert ground_truth_at(slides, 5.0, 10.0).idx == 1

    def test_last_slide_uses_audio_duration_as_stop(self):
        # stop_sec = 0.0 (sentinel) → evaluator uses audio_duration
        slides = [GroundTruthSlide(idx=0, text="last", start_sec=3.0, stop_sec=0.0)]
        assert ground_truth_at(slides, 4.0, 10.0).idx == 0
        assert ground_truth_at(slides, 10.1, 10.0) is None


# ---------------------------------------------------------------------------
# load_ground_truth
# ---------------------------------------------------------------------------

_START_STOP_JSON = {
    "presentation": {
        "id": {"uuid": "abc", "name": "Test Song", "audio": "/tmp/test.wav"},
        "groups": [
            {
                "name": "",
                "color": None,
                "slides": [
                    {"enabled": True, "text": "verse one",
                     "start time": 0.0, "stop time": 5.0, "notes": "", "label": ""},
                    {"enabled": True, "text": "verse two",
                     "start time": 5.0, "stop time": 10.0, "notes": "", "label": ""},
                    {"enabled": False, "text": "disabled",
                     "start time": 10.0, "stop time": 15.0, "notes": "", "label": ""},
                ],
            }
        ],
        "has_timeline": False,
        "presentation_path": "/tmp/test.pro",
        "destination": "presentation",
    }
}

_TRIGGER_TIME_JSON = {
    "presentation": {
        "id": {"uuid": "def", "name": "Pledge", "audio": "/tmp/pledge.wav"},
        "groups": [
            {
                "name": "",
                "color": None,
                "slides": [
                    {"enabled": True, "text": "I pledge allegiance",
                     "trigger time": 0.0, "notes": "", "label": ""},
                    {"enabled": True, "text": "to the flag",
                     "trigger time": 3.0, "notes": "", "label": ""},
                    {"enabled": True, "text": "of the United States",
                     "trigger time": 6.0, "notes": "", "label": ""},
                ],
            }
        ],
        "has_timeline": False,
        "presentation_path": "/tmp/pledge.pro",
        "destination": "presentation",
    }
}


_REPEATED_SLIDE_JSON = {
    "presentation": {
        "id": {"uuid": "ghi", "name": "Song With Chorus", "audio": "/tmp/chorus.wav"},
        "groups": [
            {
                "name": "",
                "color": None,
                "slides": [
                    {"enabled": True, "text": "verse one",
                     "trigger time": [0.0], "notes": "", "label": ""},
                    {"enabled": True, "text": "chorus text",
                     "trigger time": [10.0, 30.0, 50.0], "notes": "", "label": ""},
                    {"enabled": True, "text": "verse two",
                     "trigger time": [20.0], "notes": "", "label": ""},
                ],
            }
        ],
        "has_timeline": False,
        "presentation_path": "/tmp/chorus.pro",
        "destination": "presentation",
    }
}


class TestLoadGroundTruth:
    def _write(self, data: dict) -> Path:
        p = Path(tempfile.mktemp(suffix=".json"))
        p.write_text(json.dumps(data))
        return p

    def test_start_stop_format_legacy_scalars(self):
        p = self._write(_START_STOP_JSON)
        slides, audio_path, name = load_ground_truth(p)
        assert name == "Test Song"
        assert audio_path == "/tmp/test.wav"
        assert len(slides) == 2  # enabled=False excluded
        assert slides[0].start_sec == 0.0
        assert slides[0].stop_sec == 5.0
        assert slides[1].start_sec == 5.0

    def test_trigger_time_format_legacy_scalar_derives_stop(self):
        p = self._write(_TRIGGER_TIME_JSON)
        slides, _, _ = load_ground_truth(p)
        assert len(slides) == 3
        assert slides[0].stop_sec == 3.0
        assert slides[1].stop_sec == 6.0
        assert slides[2].stop_sec == 0.0  # sentinel

    def test_trigger_time_list_expands_to_one_entry_per_occurrence(self):
        p = self._write(_REPEATED_SLIDE_JSON)
        slides, _, _ = load_ground_truth(p)
        # verse one (t=0), chorus (t=10), verse two (t=20), chorus (t=30), chorus (t=50)
        assert len(slides) == 5

    def test_repeated_slide_entries_sorted_chronologically(self):
        p = self._write(_REPEATED_SLIDE_JSON)
        slides, _, _ = load_ground_truth(p)
        starts = [s.start_sec for s in slides]
        assert starts == sorted(starts)

    def test_repeated_slide_stop_derived_from_next_entry(self):
        p = self._write(_REPEATED_SLIDE_JSON)
        slides, _, _ = load_ground_truth(p)
        # t=0 (verse1) stop → 10.0 (first chorus start)
        assert slides[0].stop_sec == 10.0
        # t=10 (chorus) stop → 20.0 (verse2 start)
        assert slides[1].stop_sec == 20.0
        # t=20 (verse2) stop → 30.0 (second chorus start)
        assert slides[2].stop_sec == 30.0

    def test_repeated_slide_same_idx_different_start(self):
        p = self._write(_REPEATED_SLIDE_JSON)
        slides, _, _ = load_ground_truth(p)
        chorus_entries = [s for s in slides if s.text == "chorus text"]
        assert len(chorus_entries) == 3
        # All share the same ProPresenter slide index
        assert len({s.idx for s in chorus_entries}) == 1
        # But each has a distinct start time
        assert len({s.start_sec for s in chorus_entries}) == 3

    def test_disabled_slides_excluded(self):
        p = self._write(_START_STOP_JSON)
        slides, _, _ = load_ground_truth(p)
        texts = [s.text for s in slides]
        assert "disabled" not in texts


# ---------------------------------------------------------------------------
# AccuracyEvaluator
# ---------------------------------------------------------------------------

class TestAccuracyEvaluator:
    def _run(self, ground_truth, audio_seconds=6.0, window=2.0, poll=1.0,
             transcribe_returns="", embedder_returns=None):
        evaluator, transcriber, embedder = _make_evaluator(
            ground_truth, window_seconds=window, poll_interval=poll
        )
        transcriber.transcribe.return_value = transcribe_returns
        if embedder_returns is not None:
            embedder.find_slide_with_margin.return_value = embedder_returns

        audio = _silent_audio(audio_seconds)
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            import soundfile as sf
            sf.write(f.name, audio, SAMPLE_RATE)
            result = evaluator.evaluate(f.name, "Test", "tiny")
        return result

    def test_correct_step_count(self):
        gt = _gt((0.0, 6.0, "hello world"))
        # window=2s, poll=1s, audio=6s → 6 sliding windows (t=1,2,3,4,5,6)
        result = self._run(gt, audio_seconds=6.0, window=2.0, poll=1.0,
                           transcribe_returns="hello world something")
        assert result.total_steps == 6

    def test_all_correct_when_embedder_always_right(self):
        gt = _gt((0.0, 6.0, "hello world"))
        result = self._run(gt, audio_seconds=6.0,
                           transcribe_returns="hello world something",
                           embedder_returns=(0, 0.9, 0.5))
        # last step lands at t=6.0 which is outside [0,6) so gt_idx=-1 there;
        # count only steps where gt is active
        active = [e for e in result.events if e.gt_slide_idx >= 0]
        assert all(e.is_correct for e in active)
        assert len(active) > 0

    def test_all_wrong_when_embedder_returns_wrong_slide(self):
        gt = _gt((0.0, 6.0, "slide a"), (6.0, 12.0, "slide b"))
        result = self._run(gt, audio_seconds=6.0,
                           transcribe_returns="hello world something",
                           embedder_returns=(1, 0.9, 0.5))
        # slide 0 is active for t in [0,6); embedder always returns slide 1 → all wrong
        active = [e for e in result.events if e.gt_slide_idx == 0]
        assert all(not e.is_correct for e in active)

    def test_missed_slides_when_never_predicted(self):
        gt = _gt((0.0, 3.0, "slide a"), (3.0, 6.0, "slide b"))
        # embedder always returns slide 0
        result = self._run(gt, audio_seconds=6.0,
                           transcribe_returns="hello world something",
                           embedder_returns=(0, 0.9, 0.5))
        missed = result.missed_slides
        assert any(s.slide_idx == 1 for s in missed)

    def test_detection_latency_is_positive(self):
        gt = _gt((0.0, 6.0, "hello world"))
        result = self._run(gt, audio_seconds=6.0, window=2.0, poll=1.0,
                           transcribe_returns="hello world something",
                           embedder_returns=(0, 0.9, 0.5))
        assert result.per_slide[0].detection_latency_sec >= 0.0

    def test_no_correct_when_no_text_transcribed(self):
        gt = _gt((0.0, 6.0, "hello world"))
        result = self._run(gt, audio_seconds=6.0, transcribe_returns="",
                           embedder_returns=(-1, 0.0, 0.0))
        assert result.correct_steps == 0

    def test_events_logged_per_step(self):
        gt = _gt((0.0, 6.0, "hello world"))
        result = self._run(gt, audio_seconds=6.0)
        assert len(result.events) == result.total_steps
        for e in result.events:
            assert isinstance(e, InferenceEvent)

    def test_inference_accuracy_fraction(self):
        gt = _gt((0.0, 6.0, "hello world"))
        result = self._run(gt, audio_seconds=6.0,
                           transcribe_returns="hello world something",
                           embedder_returns=(0, 0.9, 0.5))
        assert 0.0 <= result.inference_accuracy <= 1.0
        assert result.total_steps > 0
        assert math.isclose(
            result.inference_accuracy,
            result.correct_steps / result.total_steps,
            abs_tol=1e-4,
        )


# ---------------------------------------------------------------------------
# print_summary (smoke test — no crash)
# ---------------------------------------------------------------------------

class TestPrintSummary:
    def test_does_not_raise(self, capsys):
        from speech_accuracy.evaluator import SlideMetrics, EvaluationResult
        sm = SlideMetrics(slide_idx=0, slide_text="hello\nworld", total_steps=10,
                          correct_steps=7, detection_latency_sec=1.5)
        sm2 = SlideMetrics(slide_idx=1, slide_text="missed slide", total_steps=5,
                           correct_steps=0, detection_latency_sec=None)
        r = EvaluationResult(
            presentation_name="Test",
            audio_path="/tmp/test.wav",
            total_steps=15,
            correct_steps=7,
            inference_accuracy=0.467,
            context_words=5,
            window_seconds=2.0,
            poll_interval=0.2,
            model_name="tiny",
            per_slide=[sm, sm2],
            events=[],
        )
        print_summary(r)
        out = capsys.readouterr().out
        assert "Test" in out
        assert "MISSED" in out
