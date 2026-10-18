from __future__ import annotations

from src.audio.types import AudioData

from .base import SpeechEnhancer


class DeepFilterNetEnhancer(SpeechEnhancer):
    """Reserved adapter for a future DeepFilterNet integration."""

    def __init__(self, **_: object) -> None:
        raise NotImplementedError("DeepFilterNet is a phase-2 integration; use name: mock")

    def enhance(self, audio: AudioData) -> AudioData:
        raise NotImplementedError
