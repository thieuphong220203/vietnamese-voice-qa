from __future__ import annotations

from abc import ABC, abstractmethod

from src.audio.types import AudioData

#speech to text
class SpeechRecognizer(ABC):
    @abstractmethod
    def transcribe(self, audio: AudioData) -> str:
        """Convert one speech signal into text."""
