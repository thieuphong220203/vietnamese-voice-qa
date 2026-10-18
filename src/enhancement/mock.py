from __future__ import annotations

from src.audio.types import AudioData

from .base import SpeechEnhancer


class MockEnhancer(SpeechEnhancer):
    """Pass audio through unchanged while marking the mock stage."""

    def enhance(self, audio: AudioData) -> AudioData:
        return audio.with_samples(
            audio.samples,
            metadata={**audio.metadata, "mock_enhanced": True},
        )
