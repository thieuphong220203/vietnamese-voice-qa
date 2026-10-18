from __future__ import annotations

from src.audio.types import AudioData

from .base import SpeechRecognizer


class MockASR(SpeechRecognizer):
    """Return deterministic configured text for every non-empty signal."""

    def __init__(self, transcript: str = "Xin chào, đây là câu hỏi thử nghiệm.", **_) -> None:
        self.transcript = transcript

    def transcribe(self, audio: AudioData) -> str:
        return self.transcript if audio.frame_count else ""
