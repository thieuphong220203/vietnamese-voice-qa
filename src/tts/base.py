from __future__ import annotations

from abc import ABC, abstractmethod

from src.audio.types import AudioData


class SpeechSynthesizer(ABC):
    @abstractmethod
    def synthesize(self, text: str) -> AudioData:
        """Convert response text into the shared audio representation."""
