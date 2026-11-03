"""Chọn nguồn (giọng) nào sẽ được đưa sang ASR sau khi tách.

Pipeline chỉ cần MỘT câu hỏi cho QA. Nếu đưa toàn bộ nguồn sang ASR rồi nối
transcript lại, câu hỏi của người dùng sẽ bị trộn với lời của người nói nền.
Quy tắc mặc định: người hỏi thường nói gần micro nhất nên nguồn có năng lượng
(RMS) lớn nhất được coi là giọng chính.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

SELECTION_MODES = ("primary", "all")


def rms(signal: np.ndarray) -> float:
    signal = np.asarray(signal, dtype=np.float64)
    if signal.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(signal**2)))


def primary_index(sources: Sequence[np.ndarray]) -> int:
    """Index của nguồn có RMS lớn nhất."""

    if not sources:
        raise ValueError("cần ít nhất một nguồn để chọn")
    return int(np.argmax([rms(source) for source in sources]))


def rank_by_energy(sources: Sequence[np.ndarray]) -> list[int]:
    """Thứ tự các nguồn từ to nhất đến nhỏ nhất."""

    return sorted(range(len(sources)), key=lambda i: rms(sources[i]), reverse=True)
