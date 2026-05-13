import logging
from collections import deque

from propresenter_slides.main import ProPresenterController

from ..command_parser import Command, CommandParser, CommandType

logger = logging.getLogger(__name__)


class PresentationHandler:
    """Responds to explicit voice commands only."""

    def __init__(
        self,
        pro_controller: ProPresenterController,
        command_parser: CommandParser,
        verbose: bool = False,
    ):
        self.pro_controller = pro_controller
        self.command_parser = command_parser
        self.verbose = verbose

    def on_startup(self) -> None:
        pass

    def startup_description(self) -> str:
        return "Listening for voice commands. Say 'next slide', 'previous slide', or 'go to slide N'."

    def on_transcription(self, text: str, word_buffer: deque) -> None:
        command = self.command_parser.parse(text)
        if command.type != CommandType.UNKNOWN:
            self._execute(command)

    def _execute(self, command: Command) -> None:
        if command.type == CommandType.NEXT_SLIDE:
            ok = self.pro_controller.next_slide()
            print("→ Next slide" if ok else "✗ Failed: next slide")

        elif command.type == CommandType.PREVIOUS_SLIDE:
            ok = self.pro_controller.previous_slide()
            print("← Previous slide" if ok else "✗ Failed: previous slide")

        elif command.type == CommandType.GO_TO_SLIDE:
            n = command.slide_number
            ok = self.pro_controller.go_to_slide(n)
            print(f"→ Slide {n}" if ok else f"✗ Failed: go to slide {n}")

        else:
            logger.debug("Unhandled command type: %s", command.type)
