from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

from src.audio.types import AudioData


class SpeechSynthesizer(ABC):
    @abstractmethod
    def synthesize(self, text: str) -> AudioData:
        """Convert response text into the shared audio representation."""

    def stream_synthesize(self, text: str) -> Iterator[AudioData]:
        """Yield audio pieces; batch-only implementations yield one piece."""
        yield self.synthesize(text)
