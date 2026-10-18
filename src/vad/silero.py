from __future__ import annotations

from src.audio.types import AudioData, SpeechSegment

from .base import VoiceActivityDetector


class SileroVAD(VoiceActivityDetector):
    """Reserved adapter for a future Silero VAD integration."""

    def __init__(self, **_: object) -> None:
        raise NotImplementedError("Silero VAD is a phase-2 integration; use name: mock")

    def detect(self, audio: AudioData) -> tuple[SpeechSegment, ...]:
        raise NotImplementedError
