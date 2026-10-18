from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from src.audio.types import AudioData, SpeechSegment


class VoiceActivityDetector(ABC):
    @abstractmethod
    def detect(self, audio: AudioData) -> Sequence[SpeechSegment]:
        """Return speech regions without changing the pipeline audio format."""
