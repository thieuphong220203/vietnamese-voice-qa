"""Đọc/ghi WAV bằng numpy (chỉ dùng thư viện chuẩn ``wave``, không cần soundfile).

Dùng chung được cho cả phần chuẩn bị dữ liệu separation và enhancement.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Trả về (tín hiệu mono float32 trong [-1, 1], sample rate)."""

    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_rate = wav_file.getframerate()
        width = wav_file.getsampwidth()
        raw = wav_file.readframes(wav_file.getnframes())

    if width == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 3:
        bytes_ = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        ints = (
            bytes_[:, 0].astype(np.int32)
            | (bytes_[:, 1].astype(np.int32) << 8)
            | (bytes_[:, 2].astype(np.int32) << 16)
        )
        ints = np.where(ints >= 1 << 23, ints - (1 << 24), ints)
        data = ints.astype(np.float32) / float(1 << 23)
    elif width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / float(1 << 31)
    else:
        raise ValueError(f"không hỗ trợ PCM {width} byte: {path}")

    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data.astype(np.float32), sample_rate


def write_wav(path: str | Path, signal: np.ndarray, sample_rate: int) -> Path:
    """Ghi mono PCM 16-bit."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(np.asarray(signal, dtype=np.float64), -1.0, 1.0) * 32767.0).round()
    with wave.open(str(destination), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.astype("<i2").tobytes())
    return destination


def resample_linear(signal: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Resample đơn giản (nội suy tuyến tính). Dữ liệu VIVOS đã là 16 kHz nên hiếm khi cần."""

    if source_rate == target_rate:
        return np.asarray(signal, dtype=np.float32)
    try:
        from scipy.signal import resample_poly

        divisor = np.gcd(source_rate, target_rate)
        return resample_poly(signal, target_rate // divisor, source_rate // divisor).astype(np.float32)
    except ImportError:
        target_len = round(len(signal) * target_rate / source_rate)
        old_x = np.arange(len(signal)) / source_rate
        new_x = np.arange(target_len) / target_rate
        return np.interp(new_x, old_x, signal).astype(np.float32)


def rms(signal: np.ndarray) -> float:
    signal = np.asarray(signal, dtype=np.float64)
    return float(np.sqrt(np.mean(signal**2))) if signal.size else 0.0


def scale_to_dbfs(signal: np.ndarray, target_dbfs: float) -> np.ndarray:
    """Chỉnh âm lượng để RMS bằng ``target_dbfs`` (ví dụ -25 dBFS)."""

    current = rms(signal)
    if current == 0:
        return np.asarray(signal, dtype=np.float32)
    return (signal * (10 ** (target_dbfs / 20) / current)).astype(np.float32)
