from enum import Enum


class Mode(Enum):
    PRESENTATION = "presentation"
    FOLLOW_TRIGGER_WORDS = "follow-trigger-words"
    FOLLOW_SEMANTIC_WORDS = "follow-semantic-words"
    FOLLOW_SEMANTIC_AUDIO = "follow-semantic-audio"
