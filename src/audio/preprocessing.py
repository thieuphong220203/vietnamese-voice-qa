"""Reusable audio transforms that are not tied to a particular model."""

from __future__ import annotations

from .types import AudioData


def normalize_peak(audio: AudioData, target_peak: float = 0.95) -> AudioData:
    if not 0 < target_peak <= 1:
        raise ValueError("target_peak must be in (0, 1]")
    peak = max((abs(sample) for sample in audio.samples), default=0.0)
    if peak == 0:
        return audio
    scale = target_peak / peak
    return audio.with_samples(sample * scale for sample in audio.samples)


def to_mono(audio: AudioData) -> AudioData:
    if audio.channels == 1:
        return audio
    mono = (
        sum(audio.samples[index : index + audio.channels]) / audio.channels
        for index in range(0, len(audio.samples), audio.channels)
    )
    return audio.with_samples(mono, channels=1)
