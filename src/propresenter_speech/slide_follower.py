"""
Follow-mode slide tracking.

Fetches the text of the currently active ProPresenter slide, extracts the
last N words as trigger words, then detects when the speaker has reached
that point in their text so the next slide can be cued automatically.

ProPresenter API notes:
- v1/presentation/active  — returns presentation metadata and may contain
  nested slide text in various shapes depending on PP7 version.
- v1/status/slide         — returns current slide position; sometimes
  includes text fields depending on how PP is configured.

Because the exact response shape varies across ProPresenter versions, text
extraction walks the response recursively looking for known field names.
If slide text cannot be retrieved the follower degrades gracefully: follow
mode keeps listening for explicit commands while warning the user.
"""

import logging
import re
from typing import Optional

from propresenter_slides.main import ProPresenterController

logger = logging.getLogger(__name__)

# ProPresenter response keys that typically carry plain slide text.
_TEXT_KEYS = ("text", "plainText", "plain_text", "content", "body", "notes", "lyrics", "label")
# Keys whose values are containers worth recursing into.
_CONTAINER_KEYS = ("slides", "slide", "slideGroups", "groups", "group", "elements", "element", "items")


class SlideFollower:
    """
    Tracks trigger words derived from the active slide's text.

    Usage::

        follower = SlideFollower(pro_controller, trigger_word_count=2)
        follower.refresh()          # fetch current slide text
        print(follower.trigger_words)

        if follower.matches(transcript):
            pro_controller.next_slide()
            follower.refresh()      # update for the new slide
    """

    def __init__(
        self,
        pro_controller: ProPresenterController,
        trigger_word_count: int = 1,
    ):
        self._controller = pro_controller
        self._trigger_word_count = trigger_word_count
        self._trigger_words: list[str] = []

    @property
    def trigger_words(self) -> list[str]:
        return list(self._trigger_words)

    @property
    def has_triggers(self) -> bool:
        return bool(self._trigger_words)

    def refresh(self) -> bool:
        """
        Fetch the active slide's text and update trigger words.

        Returns True when trigger words were successfully updated.
        """
        text = self._fetch_slide_text()
        if not text:
            logger.warning("Could not retrieve slide text from ProPresenter.")
            self._trigger_words = []
            return False

        words = extract_words(text)
        if not words:
            logger.warning("Slide text yielded no words after normalisation.")
            self._trigger_words = []
            return False

        self._trigger_words = words[-self._trigger_word_count:]
        logger.info("Trigger words updated: %s", self._trigger_words)
        return True

    def matches(self, transcript: str) -> bool:
        """
        Return True if *all* trigger words appear in the transcript.

        Uses simple substring matching so word order is not enforced, which
        is more robust to ASR insertions and reorderings.
        """
        if not self._trigger_words:
            return False
        normalised = transcript.lower()
        return all(word in normalised for word in self._trigger_words)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_slide_text(self) -> str:
        """Try multiple ProPresenter endpoints to retrieve current slide text."""
        data = self._controller.get_active_presentation()
        if data:
            text = extract_text_from_response(data)
            if text:
                return text

        status = self._controller.get_status()
        if status:
            text = extract_text_from_response(status)
            if text:
                return text

        return ""


# ------------------------------------------------------------------
# Pure helpers (module-level so they are independently testable)
# ------------------------------------------------------------------

def extract_words(text: str) -> list[str]:
    """
    Normalise *text* and return a list of lowercase alphabetic words.

    Strips RTF control sequences, HTML tags, and punctuation so that the
    result is suitable for substring matching against ASR transcripts.
    """
    # Remove RTF control words (e.g. \par, \b0, \cf1)
    text = re.sub(r"\\[a-zA-Z]+\d*\s?", " ", text)
    # Remove HTML / XML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Lower-case, keep only letters, digits, and whitespace
    text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return [w for w in text.split() if w]


def extract_text_from_response(data: object) -> str:
    """
    Recursively search a ProPresenter API response dict/list for text content.

    Returns the first non-empty string found under a known text key, or an
    empty string if nothing is found.
    """
    if isinstance(data, str):
        stripped = data.strip()
        return stripped if stripped else ""

    if isinstance(data, list):
        for item in data:
            result = extract_text_from_response(item)
            if result:
                return result
        return ""

    if not isinstance(data, dict):
        return ""

    for key in _TEXT_KEYS:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    for key in _CONTAINER_KEYS:
        value = data.get(key)
        if value:
            result = extract_text_from_response(value)
            if result:
                return result

    return ""
