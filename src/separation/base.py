from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from src.audio.types import AudioData


class SpeechSeparator(ABC):
    @abstractmethod
    def separate(self, audio: AudioData) -> Sequence[AudioData]:
        """Split an audio signal into zero or more speaker/source signals."""
