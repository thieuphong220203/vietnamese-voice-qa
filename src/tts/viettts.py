from __future__ import annotations

from src.audio.types import AudioData

from .base import SpeechSynthesizer


class VietTTSSynthesizer(SpeechSynthesizer):
    """Reserved adapter for a future Vietnamese TTS integration."""

    def __init__(self, **_: object) -> None:
        raise NotImplementedError("VietTTS is a phase-2 integration; use name: mock")

    def synthesize(self, text: str) -> AudioData:
        raise NotImplementedError
