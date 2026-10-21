"""Adapter for VieNeu's Vietnamese streaming speech synthesizer."""

from __future__ import annotations

from collections.abc import Iterator
from itertools import chain

import numpy as np

from src.audio.types import AudioData

from .base import SpeechSynthesizer


class VieNeuSynthesizer(SpeechSynthesizer):
    """Load VieNeu once and expose its audio chunks through the shared interface."""

    def __init__(
        self,
        voice: str = "Hải Đăng",
        backend: str = "onnx",
        precision: str = "fp32",
    ) -> None:
        if backend != "onnx":
            raise ValueError("VieNeu TTS currently supports backend='onnx' here")
        if not voice.strip():
            raise ValueError("VieNeu voice cannot be empty")

        try:
            from vieneu import Vieneu
        except ImportError as error:
            raise RuntimeError(
                "VieNeu is missing. Install project requirements first."
            ) from error

        self.voice = voice
        self.engine = Vieneu(backend=backend, precision=precision)
        self.sample_rate = int(self.engine.sample_rate)

    def stream_synthesize(self, text: str) -> Iterator[AudioData]:
        if not text.strip():
            return

        for chunk in self.engine.infer_stream(text, voice=self.voice):
            samples = np.asarray(chunk, dtype=np.float32)
            if samples.ndim != 1:
                raise ValueError(f"VieNeu returned audio with shape {samples.shape}")
            if samples.size:
                yield AudioData.from_iterable(samples, sample_rate=self.sample_rate)

    def synthesize(self, text: str) -> AudioData:
        chunks = self.stream_synthesize(text)
        return AudioData.from_iterable(
            chain.from_iterable(chunk.samples for chunk in chunks),
            sample_rate=self.sample_rate,
        )
