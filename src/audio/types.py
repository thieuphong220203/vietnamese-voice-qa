"""Shared, model-agnostic data types used by the audio pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


@dataclass(frozen=True, slots=True)
class AudioData:
    """An interleaved floating-point waveform and its basic format metadata."""

    samples: tuple[float, ...]
    sample_rate: int
    channels: int = 1
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "samples", tuple(float(value) for value in self.samples))
        object.__setattr__(self, "metadata", dict(self.metadata))
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be greater than zero")
        if self.channels <= 0:
            raise ValueError("channels must be greater than zero")
        if len(self.samples) % self.channels != 0:
            raise ValueError("interleaved sample count must be divisible by channels")

    @classmethod
    def from_iterable(
        cls,
        samples: Iterable[float],
        sample_rate: int,
        channels: int = 1,
        metadata: Mapping[str, Any] | None = None,
    ) -> "AudioData":
        return cls(tuple(samples), sample_rate, channels, metadata or {})

    @classmethod
    def silence(
        cls,
        duration_seconds: float,
        sample_rate: int = 16_000,
        channels: int = 1,
        metadata: Mapping[str, Any] | None = None,
    ) -> "AudioData":
        if duration_seconds < 0:
            raise ValueError("duration_seconds cannot be negative")
        frame_count = round(duration_seconds * sample_rate)
        return cls((0.0,) * (frame_count * channels), sample_rate, channels, metadata or {})

    @property
    def frame_count(self) -> int:
        return len(self.samples) // self.channels

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / self.sample_rate

    def with_samples(
        self,
        samples: Iterable[float],
        *,
        sample_rate: int | None = None,
        channels: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "AudioData":
        return AudioData.from_iterable(
            samples,
            sample_rate or self.sample_rate,
            channels or self.channels,
            self.metadata if metadata is None else metadata,
        )


@dataclass(frozen=True, slots=True)
class SpeechSegment:
    """A speech interval and the corresponding waveform."""

    audio: AudioData
    start_seconds: float
    end_seconds: float

    def __post_init__(self) -> None:
        if self.start_seconds < 0 or self.end_seconds < self.start_seconds:
            raise ValueError("invalid speech segment boundaries")
