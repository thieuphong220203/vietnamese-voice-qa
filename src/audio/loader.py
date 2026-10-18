"""Small WAV loader/writer used by demos and tests.

Model wrappers remain responsible for any model-specific tensor conversion.
"""

from __future__ import annotations

import wave
from pathlib import Path

from .types import AudioData


def load_wav(path: str | Path) -> AudioData:
    source = Path(path)
    with wave.open(str(source), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()
        raw = wav_file.readframes(wav_file.getnframes())

    if sample_width == 1:
        samples = tuple((value - 128) / 128.0 for value in raw)
    elif sample_width in (2, 4):
        scale = float(1 << (sample_width * 8 - 1))
        samples = tuple(
            int.from_bytes(raw[index : index + sample_width], "little", signed=True) / scale
            for index in range(0, len(raw), sample_width)
        )
    else:
        raise ValueError(f"unsupported PCM sample width: {sample_width} bytes")

    return AudioData(samples, sample_rate, channels, {"source": str(source)})


def save_wav(audio: AudioData, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pcm = b"".join(
        round(max(-1.0, min(1.0, sample)) * 32767).to_bytes(2, "little", signed=True)
        for sample in audio.samples
    )
    with wave.open(str(destination), "wb") as wav_file:
        wav_file.setnchannels(audio.channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(audio.sample_rate)
        wav_file.writeframes(pcm)
    return destination
