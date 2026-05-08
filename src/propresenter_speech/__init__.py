"""
propresenter-speech — voice-controlled slide advancement for ProPresenter using Whisper ASR.
"""

__version__ = "0.1.0"

from .command_parser import Command, CommandParser, CommandType
from .transcriber import Transcriber

__all__ = ["Command", "CommandParser", "CommandType", "Transcriber"]
