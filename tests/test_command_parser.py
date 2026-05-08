"""
Unit tests for CommandParser.

All tests are pure Python — no I/O, no external services.
"""

import pytest
from propresenter_speech.command_parser import Command, CommandParser, CommandType


@pytest.fixture
def parser() -> CommandParser:
    return CommandParser()


# ---------------------------------------------------------------------------
# Next-slide commands
# ---------------------------------------------------------------------------

class TestNextSlide:
    def test_next_slide_literal(self, parser):
        assert parser.parse("next slide").type == CommandType.NEXT_SLIDE

    def test_next_slide_uppercase(self, parser):
        assert parser.parse("NEXT SLIDE").type == CommandType.NEXT_SLIDE

    def test_next_slide_mixed_case(self, parser):
        assert parser.parse("Next Slide").type == CommandType.NEXT_SLIDE

    def test_advance_slide(self, parser):
        assert parser.parse("advance slide").type == CommandType.NEXT_SLIDE

    def test_go_forward(self, parser):
        assert parser.parse("go forward").type == CommandType.NEXT_SLIDE

    def test_go_to_next(self, parser):
        assert parser.parse("go to next").type == CommandType.NEXT_SLIDE

    def test_go_to_next_slide(self, parser):
        assert parser.parse("go to next slide").type == CommandType.NEXT_SLIDE

    def test_forward_slide(self, parser):
        assert parser.parse("forward slide").type == CommandType.NEXT_SLIDE

    def test_slide_forward(self, parser):
        assert parser.parse("slide forward").type == CommandType.NEXT_SLIDE


# ---------------------------------------------------------------------------
# Previous-slide commands
# ---------------------------------------------------------------------------

class TestPreviousSlide:
    def test_previous_slide_literal(self, parser):
        assert parser.parse("previous slide").type == CommandType.PREVIOUS_SLIDE

    def test_prev_slide(self, parser):
        assert parser.parse("prev slide").type == CommandType.PREVIOUS_SLIDE

    def test_go_back(self, parser):
        assert parser.parse("go back").type == CommandType.PREVIOUS_SLIDE

    def test_go_to_previous(self, parser):
        assert parser.parse("go to previous").type == CommandType.PREVIOUS_SLIDE

    def test_go_to_previous_slide(self, parser):
        assert parser.parse("go to previous slide").type == CommandType.PREVIOUS_SLIDE

    def test_last_slide(self, parser):
        assert parser.parse("last slide").type == CommandType.PREVIOUS_SLIDE

    def test_prior_slide(self, parser):
        assert parser.parse("prior slide").type == CommandType.PREVIOUS_SLIDE

    def test_back_slide(self, parser):
        assert parser.parse("back slide").type == CommandType.PREVIOUS_SLIDE

    def test_slide_back(self, parser):
        assert parser.parse("slide back").type == CommandType.PREVIOUS_SLIDE

    def test_previous_slide_uppercase(self, parser):
        assert parser.parse("PREVIOUS SLIDE").type == CommandType.PREVIOUS_SLIDE


# ---------------------------------------------------------------------------
# Go-to-slide commands
# ---------------------------------------------------------------------------

class TestGoToSlide:
    def test_go_to_slide_integer(self, parser):
        cmd = parser.parse("go to slide 5")
        assert cmd.type == CommandType.GO_TO_SLIDE
        assert cmd.slide_number == 5

    def test_go_to_slide_word_number(self, parser):
        cmd = parser.parse("go to slide five")
        assert cmd.type == CommandType.GO_TO_SLIDE
        assert cmd.slide_number == 5

    def test_go_to_slide_double_digit(self, parser):
        cmd = parser.parse("go to slide 12")
        assert cmd.type == CommandType.GO_TO_SLIDE
        assert cmd.slide_number == 12

    def test_go_to_slide_teen_word(self, parser):
        cmd = parser.parse("go to slide fifteen")
        assert cmd.type == CommandType.GO_TO_SLIDE
        assert cmd.slide_number == 15

    def test_slide_number_keyword(self, parser):
        cmd = parser.parse("slide number 3")
        assert cmd.type == CommandType.GO_TO_SLIDE
        assert cmd.slide_number == 3

    def test_slide_number_word(self, parser):
        cmd = parser.parse("slide number three")
        assert cmd.type == CommandType.GO_TO_SLIDE
        assert cmd.slide_number == 3

    def test_slide_bare_integer(self, parser):
        cmd = parser.parse("slide 7")
        assert cmd.type == CommandType.GO_TO_SLIDE
        assert cmd.slide_number == 7

    def test_jump_to_slide(self, parser):
        cmd = parser.parse("jump to slide 10")
        assert cmd.type == CommandType.GO_TO_SLIDE
        assert cmd.slide_number == 10

    def test_go_to_slide_first(self, parser):
        cmd = parser.parse("go to slide one")
        assert cmd.type == CommandType.GO_TO_SLIDE
        assert cmd.slide_number == 1

    def test_go_to_slide_twenty(self, parser):
        cmd = parser.parse("go to slide twenty")
        assert cmd.type == CommandType.GO_TO_SLIDE
        assert cmd.slide_number == 20


# ---------------------------------------------------------------------------
# Unknown / unrecognised input
# ---------------------------------------------------------------------------

class TestUnknownCommand:
    def test_empty_string(self, parser):
        assert parser.parse("").type == CommandType.UNKNOWN

    def test_gibberish(self, parser):
        assert parser.parse("blah blah blah").type == CommandType.UNKNOWN

    def test_partial_match_no_number(self, parser):
        # "slide" alone with no valid number should not trigger GO_TO_SLIDE
        # (word2number will fail on a random word)
        result = parser.parse("slide blorg")
        assert result.type == CommandType.UNKNOWN

    def test_unrelated_sentence(self, parser):
        assert parser.parse("the weather is nice today").type == CommandType.UNKNOWN

    def test_whitespace_only(self, parser):
        assert parser.parse("   ").type == CommandType.UNKNOWN


# ---------------------------------------------------------------------------
# Command dataclass
# ---------------------------------------------------------------------------

class TestCommandRepr:
    def test_repr_go_to_slide(self, parser):
        cmd = Command(type=CommandType.GO_TO_SLIDE, slide_number=3)
        assert "3" in repr(cmd)

    def test_repr_next_slide(self):
        cmd = Command(type=CommandType.NEXT_SLIDE)
        assert "next_slide" in repr(cmd)

    def test_slide_number_defaults_to_none(self):
        cmd = Command(type=CommandType.NEXT_SLIDE)
        assert cmd.slide_number is None
