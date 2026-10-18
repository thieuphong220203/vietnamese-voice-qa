from __future__ import annotations

from src.audio.types import AudioData

from .base import SpeechSynthesizer


class MockTTS(SpeechSynthesizer):
    """Create silent audio carrying the synthesized text as metadata."""

    def __init__(
        self,
        sample_rate: int = 16_000,
        seconds_per_character: float = 0.02,
        minimum_duration_seconds: float = 0.1,
    ) -> None:
        if seconds_per_character < 0 or minimum_duration_seconds < 0:
            raise ValueError("mock TTS durations cannot be negative")
        self.sample_rate = sample_rate
        self.seconds_per_character = seconds_per_character
        self.minimum_duration_seconds = minimum_duration_seconds

    def synthesize(self, text: str) -> AudioData:
        duration = max(self.minimum_duration_seconds, len(text) * self.seconds_per_character)
        return AudioData.silence(
            duration,
            self.sample_rate,
            metadata={"mock_tts": True, "text": text},
        )
