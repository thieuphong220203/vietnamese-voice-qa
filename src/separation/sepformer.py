from __future__ import annotations

from src.audio.types import AudioData

from .base import SpeechSeparator


class SepFormerSeparator(SpeechSeparator):
    """Reserved adapter for a future SepFormer integration."""

    def __init__(self, **_: object) -> None:
        raise NotImplementedError("SepFormer is a phase-2 integration; use name: mock")

    def separate(self, audio: AudioData) -> tuple[AudioData, ...]:
        raise NotImplementedError
