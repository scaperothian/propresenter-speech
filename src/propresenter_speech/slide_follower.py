"""
Follow-mode slide tracking.

Fetches the text of the current slide from the ProPresenter HTTP API and
extracts the last N words as trigger words so the controller can auto-advance
when the speaker reaches them.

Slide-text retrieval flow (per refresh):
1. get_active_presentation_uuid()  — resolves the UUID of the active presentation.
2. get_presentation_details(uuid)  — fetches full details; result is cached for the
   lifetime of the class and re-fetched only when the UUID changes (new presentation).
3. get_slide_index()               — returns the zero-based index of the current slide
   via GET /v1/presentation/slide_index?chunked=false.
4. find_slides(details)[index]["text"]  — reads the exact slide text directly.

Falls back to GET /v1/status/slide + recursive text search when any step above fails.

API reference: http://<propresenter-ip>:1025/v1/doc/index.html#
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
        self._presentation_uuid: str | None = None
        self._presentation_details: dict | None = None
        self._slide_index: int | None = None

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

    def validate(self) -> tuple[bool, str]:
        """
        Confirm that presentation details are fetchable with slides containing text.

        Caches the UUID and details so the first refresh() call reuses them.
        Returns (True, "") on success, or (False, reason) on failure.
        """
        uuid = self._controller.get_active_presentation_uuid()
        if not uuid:
            return False, "Could not retrieve active presentation UUID from ProPresenter."

        details = self._controller.get_presentation_details(uuid)
        if not details:
            return False, f"Could not retrieve presentation details for UUID {uuid}."

        slides = self._controller.find_slides(details)
        if not slides:
            return False, "Presentation details contain no slides."

        if not any("text" in slide for slide in slides if isinstance(slide, dict)):
            return False, "No slides in the presentation contain a 'text' field."

        self._presentation_uuid = uuid
        self._presentation_details = details
        return True, ""

    def _fetch_slide_text(self) -> str:
        """
        Return the text of the currently active slide.

        Keeps presentation details cached; only re-fetches when the active
        presentation UUID changes.  Falls back to status endpoint on any failure.
        """
        uuid = self._controller.get_active_presentation_uuid()
        logger.debug("Active UUID: %s", uuid)
        if not uuid:
            logger.debug("No active UUID — falling back to status endpoint")
            return self._status_fallback()

        if uuid != self._presentation_uuid or self._presentation_details is None:
            logger.debug("UUID changed or no cached details — fetching presentation %s", uuid)
            details = self._controller.get_presentation_details(uuid)
            if not details:
                logger.debug("get_presentation_details returned nothing — falling back to status endpoint")
                return self._status_fallback()
            self._presentation_uuid = uuid
            self._presentation_details = details

        slide_index = self._controller.get_slide_index()
        logger.debug("Slide index: %s", slide_index)
        if slide_index is None:
            logger.debug("get_slide_index returned None — falling back to status endpoint")
            return self._status_fallback()

        self._slide_index = slide_index
        slides = self._controller.find_slides(self._presentation_details)
        logger.debug("Slides found: %d, using index: %d", len(slides), slide_index)
        if slide_index < len(slides):
            text = self._text_at_index(slide_index)
            if text:
                return text
            logger.debug("Slide text empty or non-string — falling back to status endpoint")
        else:
            logger.debug("Slide index %d out of range (slides: %d) — falling back to status endpoint", slide_index, len(slides))

        return self._status_fallback()

    def _text_at_index(self, index: int) -> str:
        if self._presentation_details is None:
            return ""
        slides = self._controller.find_slides(self._presentation_details)
        if index < 0 or index >= len(slides):
            return ""
        slide = slides[index]
        text = slide.get("text", "") if isinstance(slide, dict) else ""
        logger.debug("Slide text at %d (type=%s): %r", index, type(text).__name__, str(text)[:80])
        return text.strip() if isinstance(text, str) else ""

    def refresh_after_advance(self, delta: int = 1) -> bool:
        """
        Update trigger words for the slide *delta* positions ahead of the cached index.

        Avoids calling ``get_slide_index()`` again (which may still return the
        pre-advance value due to API propagation delay).  Falls back to a full
        ``refresh()`` when no cached index is available.

        Returns True if trigger words were updated, False when the end of the
        presentation has been reached (trigger words are cleared).
        """
        if self._slide_index is None or self._presentation_details is None:
            return self.refresh()

        next_index = self._slide_index + delta
        slides = self._controller.find_slides(self._presentation_details)
        if next_index >= len(slides):
            self._trigger_words = []
            logger.info("End of presentation reached — no more trigger words.")
            return False

        text = self._text_at_index(next_index)
        if not text:
            return self.refresh()

        words = extract_words(text)
        if not words:
            return self.refresh()

        self._slide_index = next_index
        self._trigger_words = words[-self._trigger_word_count:]
        logger.info("Trigger words updated: %s", self._trigger_words)
        return True

    def _status_fallback(self) -> str:
        status = self._controller.get_status()
        if status:
            return extract_text_from_response(status)
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
