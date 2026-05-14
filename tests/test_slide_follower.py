"""
Unit tests for SlideFollower, extract_words, and extract_text_from_response.
No network I/O — ProPresenterController is fully mocked.
"""

import pytest
from unittest.mock import MagicMock

from propresenter_speech.slide_follower import (
    SlideFollower,
    extract_text_from_response,
    extract_words,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_pro():
    return MagicMock()


@pytest.fixture
def follower(mock_pro):
    return SlideFollower(mock_pro, trigger_word_count=1, trigger_index=-1)


# ---------------------------------------------------------------------------
# extract_words
# ---------------------------------------------------------------------------

class TestExtractWords:
    def test_simple_sentence(self):
        assert extract_words("Hello world") == ["hello", "world"]

    def test_strips_punctuation(self):
        assert extract_words("Hello, world!") == ["hello", "world"]

    def test_strips_rtf_control_words(self):
        result = extract_words(r"\par Hello \b0 world")
        assert "hello" in result
        assert "world" in result

    def test_strips_html_tags(self):
        result = extract_words("<b>Hello</b> <i>world</i>")
        assert result == ["hello", "world"]

    def test_lowercases_all_words(self):
        assert extract_words("NEXT SLIDE") == ["next", "slide"]

    def test_empty_string(self):
        assert extract_words("") == []

    def test_whitespace_only(self):
        assert extract_words("   ") == []

    def test_numbers_preserved(self):
        result = extract_words("slide 5")
        assert "5" in result

    def test_multiline_text(self):
        result = extract_words("Amazing\ngrace\nhow sweet")
        assert result == ["amazing", "grace", "how", "sweet"]


# ---------------------------------------------------------------------------
# extract_text_from_response
# ---------------------------------------------------------------------------

class TestExtractTextFromResponse:
    def test_plain_string(self):
        assert extract_text_from_response("hello world") == "hello world"

    def test_dict_with_text_key(self):
        assert extract_text_from_response({"text": "Amazing grace"}) == "Amazing grace"

    def test_dict_with_plain_text_key(self):
        assert extract_text_from_response({"plainText": "Amazing grace"}) == "Amazing grace"

    def test_dict_with_body_key(self):
        assert extract_text_from_response({"body": "Verse text"}) == "Verse text"

    def test_nested_slides_list(self):
        data = {"slides": [{"text": "Amazing"}, {"text": "Grace"}]}
        result = extract_text_from_response(data)
        assert result == "Amazing"  # returns first match

    def test_deeply_nested(self):
        data = {"groups": [{"slides": [{"elements": [{"text": "deep text"}]}]}]}
        assert extract_text_from_response(data) == "deep text"

    def test_list_input(self):
        result = extract_text_from_response([{"text": "first"}, {"text": "second"}])
        assert result == "first"

    def test_empty_dict(self):
        assert extract_text_from_response({}) == ""

    def test_empty_list(self):
        assert extract_text_from_response([]) == ""

    def test_ignores_empty_string_values(self):
        data = {"text": "", "body": "real content"}
        assert extract_text_from_response(data) == "real content"

    def test_non_string_input(self):
        assert extract_text_from_response(42) == ""

    def test_none_input(self):
        assert extract_text_from_response(None) == ""


# ---------------------------------------------------------------------------
# SlideFollower.refresh
# ---------------------------------------------------------------------------

def _mock_slide_text(mock_pro, text: str, slide_index: int = 0) -> None:
    """Configure mock_pro to return a presentation with a single slide containing text."""
    slides = [{"text": text}]
    mock_pro.get_active_presentation_uuid.return_value = "test-uuid"
    mock_pro.get_presentation_details.return_value = {"slides": slides}
    mock_pro.find_slides.return_value = slides
    mock_pro.get_slide_index.return_value = slide_index


class TestSlideFollowerRefresh:
    def test_refresh_returns_true_on_success(self, follower, mock_pro):
        _mock_slide_text(mock_pro, "Amazing grace")
        assert follower.refresh() is True

    def test_refresh_sets_trigger_words(self, follower, mock_pro):
        _mock_slide_text(mock_pro, "Amazing grace")
        follower.refresh()
        assert follower.trigger_words == ["grace"]

    def test_refresh_uses_last_n_words(self, mock_pro):
        f = SlideFollower(mock_pro, trigger_word_count=2, trigger_index=-1)
        _mock_slide_text(mock_pro, "Amazing grace how sweet")
        f.refresh()
        assert f.trigger_words == ["how", "sweet"]

    def test_trigger_index_second_to_last(self, mock_pro):
        f = SlideFollower(mock_pro, trigger_word_count=1, trigger_index=-2)
        _mock_slide_text(mock_pro, "Amazing grace how sweet")
        f.refresh()
        assert f.trigger_words == ["how"]

    def test_trigger_index_with_multiple_words(self, mock_pro):
        f = SlideFollower(mock_pro, trigger_word_count=2, trigger_index=-2)
        _mock_slide_text(mock_pro, "Amazing grace how sweet")
        f.refresh()
        assert f.trigger_words == ["grace", "how"]

    def test_trigger_index_default_preserves_existing_behavior(self, mock_pro):
        f = SlideFollower(mock_pro, trigger_word_count=2, trigger_index=-1)
        _mock_slide_text(mock_pro, "Amazing grace how sweet")
        f.refresh()
        assert f.trigger_words == ["how", "sweet"]

    def test_refresh_falls_back_to_status(self, follower, mock_pro):
        mock_pro.get_active_presentation_uuid.return_value = None
        mock_pro.get_status.return_value = {"text": "Fallback text"}
        assert follower.refresh() is True
        assert follower.trigger_words == ["text"]

    def test_refresh_returns_false_when_no_text(self, follower, mock_pro):
        mock_pro.get_active_presentation_uuid.return_value = None
        mock_pro.get_status.return_value = None
        assert follower.refresh() is False

    def test_refresh_clears_triggers_when_no_text(self, follower, mock_pro):
        _mock_slide_text(mock_pro, "first slide")
        follower.refresh()
        mock_pro.get_active_presentation_uuid.return_value = None
        mock_pro.get_status.return_value = None
        follower.refresh()
        assert follower.trigger_words == []

    def test_has_triggers_false_before_refresh(self, follower):
        assert not follower.has_triggers

    def test_has_triggers_true_after_successful_refresh(self, follower, mock_pro):
        _mock_slide_text(mock_pro, "some text")
        follower.refresh()
        assert follower.has_triggers


# ---------------------------------------------------------------------------
# SlideFollower.matches
# ---------------------------------------------------------------------------

class TestSlideFollowerMatches:
    def test_matches_when_trigger_word_present(self, follower, mock_pro):
        _mock_slide_text(mock_pro, "Amazing grace")
        follower.refresh()
        assert follower.matches("I said amazing grace how sweet")

    def test_no_match_when_trigger_absent(self, follower, mock_pro):
        _mock_slide_text(mock_pro, "Amazing grace")
        follower.refresh()
        assert not follower.matches("something completely different")

    def test_no_match_when_no_triggers(self, follower):
        assert not follower.matches("any text at all")

    def test_matching_is_case_insensitive(self, follower, mock_pro):
        _mock_slide_text(mock_pro, "Amazing Grace")
        follower.refresh()
        assert follower.matches("I said GRACE")

    def test_all_trigger_words_must_match(self, mock_pro):
        f = SlideFollower(mock_pro, trigger_word_count=2, trigger_index=-1)
        _mock_slide_text(mock_pro, "how sweet the sound")
        f.refresh()
        assert f.matches("the sound")
        assert not f.matches("the noise")

    def test_empty_transcript(self, follower, mock_pro):
        _mock_slide_text(mock_pro, "Amazing grace")
        follower.refresh()
        assert not follower.matches("")
