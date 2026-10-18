from __future__ import annotations

from abc import ABC, abstractmethod

from src.audio.types import AudioData


class SpeechEnhancer(ABC):
    @abstractmethod
    def enhance(self, audio: AudioData) -> AudioData:
        """Return an enhanced signal in the shared audio representation."""
