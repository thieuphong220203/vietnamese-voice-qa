"""Chỉ số đánh giá tách giọng (numpy, không cần torch).

SI-SNR (scale-invariant signal-to-noise ratio, dB): càng cao càng tốt.
SI-SNRi: mức cải thiện so với chính mixture (SI-SNR(est) - SI-SNR(mix)).
Vì model không biết nguồn nào là "s1", ta ghép ước lượng với đáp án theo
hoán vị cho điểm cao nhất (PIT - permutation invariant).
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import permutations

import numpy as np

EPS = 1e-8


def si_snr(estimate: np.ndarray, reference: np.ndarray) -> float:
    estimate = np.asarray(estimate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    length = min(len(estimate), len(reference))
    estimate, reference = estimate[:length], reference[:length]
    estimate = estimate - estimate.mean()
    reference = reference - reference.mean()

    projection = (np.dot(estimate, reference) / (np.dot(reference, reference) + EPS)) * reference
    noise = estimate - projection
    return float(10 * np.log10((np.dot(projection, projection) + EPS) / (np.dot(noise, noise) + EPS)))


def best_permutation(
    estimates: Sequence[np.ndarray], references: Sequence[np.ndarray]
) -> tuple[tuple[int, ...], list[float]]:
    """Trả về (perm, điểm SI-SNR theo từng reference); ``estimates[perm[i]]`` ứng với ``references[i]``."""

    if len(estimates) < len(references):
        raise ValueError("số nguồn ước lượng ít hơn số nguồn tham chiếu")
    scores = np.array([[si_snr(est, ref) for est in estimates] for ref in references])
    best_perm: tuple[int, ...] = ()
    best_total = -np.inf
    for perm in permutations(range(len(estimates)), len(references)):
        total = sum(scores[ref_idx, est_idx] for ref_idx, est_idx in enumerate(perm))
        if total > best_total:
            best_total, best_perm = total, perm
    return best_perm, [float(scores[i, j]) for i, j in enumerate(best_perm)]


def si_snr_improvement(
    estimates: Sequence[np.ndarray], references: Sequence[np.ndarray], mixture: np.ndarray
) -> dict[str, object]:
    perm, est_scores = best_permutation(estimates, references)
    mix_scores = [si_snr(mixture, ref) for ref in references]
    improvements = [est - mix for est, mix in zip(est_scores, mix_scores)]
    return {
        "perm": perm,
        "si_snr": float(np.mean(est_scores)),
        "si_snr_mix": float(np.mean(mix_scores)),
        "si_snri": float(np.mean(improvements)),
        "si_snr_per_source": est_scores,
    }
