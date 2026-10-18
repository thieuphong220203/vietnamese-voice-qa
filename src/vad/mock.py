from __future__ import annotations

from src.audio.types import AudioData, SpeechSegment

from .base import VoiceActivityDetector


class MockVAD(VoiceActivityDetector):
    """Treat the complete input as one speech segment."""

    def __init__(self, emit_speech: bool = True) -> None:
        self.emit_speech = emit_speech

    def detect(self, audio: AudioData) -> tuple[SpeechSegment, ...]:
        if not self.emit_speech or audio.frame_count == 0:
            return ()
        return (SpeechSegment(audio, 0.0, audio.duration_seconds),)
