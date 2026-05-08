"""
Parse transcribed speech text into ProPresenter slide commands.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from word2number import w2n


class CommandType(Enum):
    NEXT_SLIDE = "next_slide"
    PREVIOUS_SLIDE = "previous_slide"
    GO_TO_SLIDE = "go_to_slide"
    UNKNOWN = "unknown"


@dataclass
class Command:
    type: CommandType
    slide_number: Optional[int] = field(default=None)

    def __repr__(self) -> str:
        if self.type == CommandType.GO_TO_SLIDE:
            return f"Command(GO_TO_SLIDE, slide={self.slide_number})"
        return f"Command({self.type.value})"


# Patterns are matched in order; place more specific patterns first.
_NEXT_PATTERNS = [
    r"\bnext\s+slide\b",
    r"\bgo\s+(to\s+)?next\b",
    r"\badvance\s+slide\b",
    r"\bslide\s+forward\b",
    r"\bgo\s+forward\b",
    r"\bforward\s+slide\b",
]

_PREVIOUS_PATTERNS = [
    r"\bprevious\s+slide\b",
    r"\bprev\s+slide\b",
    r"\bgo\s+(to\s+)?previous\b",
    r"\bprior\s+slide\b",
    r"\blast\s+slide\b",
    r"\bback\s+slide\b",
    r"\bgo\s+back\b",
    r"\bslide\s+back\b",
]

# Group 1 captures the slide identifier (digit string or word like "five").
_GO_TO_PATTERNS = [
    r"\bgo\s+to\s+slide\s+([\w\s-]+?)(?:\s*$|\s+(?:please|now)\b)",
    r"\bjump\s+to\s+slide\s+([\w\s-]+?)(?:\s*$|\s+(?:please|now)\b)",
    r"\bslide\s+number\s+([\w\s-]+?)(?:\s*$|\s+(?:please|now)\b)",
    r"\bgo\s+to\s+slide\s+(\w+)",
    r"\bjump\s+to\s+slide\s+(\w+)",
    r"\bslide\s+number\s+(\w+)",
    r"\bslide\s+(\w+)",
]


class CommandParser:
    """Converts transcribed speech into structured slide commands."""

    def parse(self, text: str) -> Command:
        normalized = text.lower().strip()

        for pattern in _NEXT_PATTERNS:
            if re.search(pattern, normalized):
                return Command(type=CommandType.NEXT_SLIDE)

        for pattern in _PREVIOUS_PATTERNS:
            if re.search(pattern, normalized):
                return Command(type=CommandType.PREVIOUS_SLIDE)

        for pattern in _GO_TO_PATTERNS:
            match = re.search(pattern, normalized)
            if match:
                candidate = match.group(1).strip()
                num = _parse_number(candidate)
                if num is not None and num > 0:
                    return Command(type=CommandType.GO_TO_SLIDE, slide_number=num)

        return Command(type=CommandType.UNKNOWN)


def _parse_number(text: str) -> Optional[int]:
    text = text.strip()

    try:
        return int(text)
    except ValueError:
        pass

    try:
        result = w2n.word_to_num(text)
        if isinstance(result, int):
            return result
        return int(result)
    except (ValueError, TypeError):
        pass

    return None
