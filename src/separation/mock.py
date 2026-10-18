from __future__ import annotations

from src.audio.types import AudioData

from .base import SpeechSeparator


class MockSeparator(SpeechSeparator):
    """Copy the input into a configurable number of fake sources."""

    def __init__(self, source_count: int = 1) -> None:
        if source_count < 1:
            raise ValueError("source_count must be at least 1")
        self.source_count = source_count

    def separate(self, audio: AudioData) -> tuple[AudioData, ...]:
        return tuple(
            audio.with_samples(
                audio.samples,
                metadata={**audio.metadata, "mock_source_index": index},
            )
            for index in range(self.source_count)
        )
