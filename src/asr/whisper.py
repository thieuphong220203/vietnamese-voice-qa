from __future__ import annotations

from src.audio.types import AudioData

from .base import SpeechRecognizer


class WhisperRecognizer(SpeechRecognizer):
    """Reserved adapter for a future Whisper integration."""

    def __init__(self, **_: object) -> None:
        raise NotImplementedError("Whisper is a phase-2 integration; use name: mock")

    def transcribe(self, audio: AudioData) -> str:
        raise NotImplementedError
