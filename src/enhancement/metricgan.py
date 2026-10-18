from __future__ import annotations

from src.audio.types import AudioData

from .base import SpeechEnhancer


class MetricGANEnhancer(SpeechEnhancer):
    """Reserved adapter for a future MetricGAN integration."""

    def __init__(self, **_: object) -> None:
        raise NotImplementedError("MetricGAN is a phase-2 integration; use name: mock")

    def enhance(self, audio: AudioData) -> AudioData:
        raise NotImplementedError
