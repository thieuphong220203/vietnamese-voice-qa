from __future__ import annotations

from src.audio.types import AudioData

from .base import SpeechRecognizer


class PhoWhisperRecognizer(SpeechRecognizer):
    """Reserved adapter for a future PhoWhisper integration."""

    def __init__(self, **_: object) -> None:
        raise NotImplementedError("PhoWhisper is a phase-2 integration; use name: mock")

    def transcribe(self, audio: AudioData) -> str:
        raise NotImplementedError
