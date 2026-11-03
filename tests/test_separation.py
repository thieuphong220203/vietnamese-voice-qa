from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.audio.types import AudioData
from src.data.prepare_separation import make_mixture, prepare
from src.data.wav_utils import read_wav, write_wav
from src.separation.factory import create_separator
from src.separation.metrics import best_permutation, si_snr, si_snr_improvement
from src.separation.selection import primary_index, rank_by_energy

SR = 16_000


def tone(freq: float, seconds: float = 1.0, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# ------------------------------------------------------------------ metrics
def test_si_snr_is_scale_invariant_and_high_for_perfect_estimate() -> None:
    ref = tone(220)
    assert si_snr(ref, ref) > 60
    noisy = ref + 0.05 * tone(1234)
    assert si_snr(3.0 * noisy, ref) == pytest.approx(si_snr(noisy, ref), abs=1e-3)


def test_pit_finds_swapped_sources() -> None:
    a, b = tone(220), tone(570)
    perm, scores = best_permutation([b, a], [a, b])
    assert perm == (1, 0)
    assert min(scores) > 60


def test_si_snr_improvement_positive_for_good_separation() -> None:
    a, b = tone(220), tone(570, amp=0.3)
    mix = a + b
    result = si_snr_improvement([a, b], [a, b], mix)
    assert result["si_snri"] > 20
    assert result["si_snr_mix"] < result["si_snr"]


# ------------------------------------------------------------------ selection
def test_primary_is_loudest_source() -> None:
    quiet, loud = tone(220, amp=0.1), tone(330, amp=0.6)
    assert primary_index([quiet, loud]) == 1
    assert rank_by_energy([quiet, loud]) == [1, 0]


# ------------------------------------------------------------------ data prep
def test_make_mixture_keeps_s1_louder_and_sums() -> None:
    rng = __import__("random").Random(0)
    mix, s1, s2 = make_mixture(tone(220, 1.0), tone(330, 0.5), 4.0, mode="max", segment_samples=None, rng=rng)
    assert len(mix) == len(s1) == len(s2) == SR
    assert np.allclose(mix, s1 + s2, atol=1e-6)
    gain = 20 * np.log10(np.sqrt(np.mean(s1**2)) / np.sqrt(np.mean(s2[: SR // 2] ** 2) * 0.5))
    assert gain == pytest.approx(4.0, abs=0.5)
    assert np.max(np.abs(mix)) <= 0.9 + 1e-6


def make_fake_vivos(root: Path) -> None:
    freqs = iter(range(150, 2000, 37))
    for split, speakers in (("train", ["SPK01", "SPK02", "SPK03", "SPK04"]), ("test", ["DEV01", "DEV02"])):
        prompts = []
        for spk in speakers:
            for n in range(3):
                utt = f"{spk}_R{n:03d}"
                write_wav(root / split / "waves" / spk / f"{utt}.wav", tone(next(freqs), 1.2 + 0.3 * n), SR)
                prompts.append(f"{utt} XIN CHÀO {spk} {n}")
        (root / split / "prompts.txt").write_text("\n".join(prompts), encoding="utf-8")


def test_prepare_creates_speaker_disjoint_splits(tmp_path: Path) -> None:
    make_fake_vivos(tmp_path / "vivos")
    out = tmp_path / "sep"
    paths = prepare(tmp_path / "vivos", out, n_train=6, n_valid=3, n_test=4, n_valid_speakers=2,
                    segment_seconds=1.0, seed=7)

    rows = {split: list(csv.DictReader(path.open(encoding="utf-8"))) for split, path in paths.items()}
    speakers = {split: {r["s1_speaker"] for r in rs} | {r["s2_speaker"] for r in rs} for split, rs in rows.items()}
    assert not speakers["train"] & speakers["valid"]
    assert speakers["test"] <= {"DEV01", "DEV02"}

    for split, split_rows in rows.items():
        for row in split_rows:
            assert row["s1_speaker"] != row["s2_speaker"]
            assert row["s1_text"].startswith("XIN CHÀO")
            mix, rate = read_wav(out / row["mix"])
            s1, _ = read_wav(out / row["s1"])
            s2, _ = read_wav(out / row["s2"])
            assert rate == SR
            assert np.allclose(mix, s1 + s2, atol=2e-4)
            if split != "test":
                assert len(mix) == SR  # cắt đúng segment 1 s


# ------------------------------------------------------------------ SepFormer wrapper (cần torch)
class FakeSepformer:
    """Giả lập SepformerSeparation: nguồn 0 nhỏ, nguồn 1 to."""

    hparams = SimpleNamespace(sample_rate=16_000, num_spks=2)

    def separate_batch(self, batch):
        import torch

        return torch.stack([batch * 0.2, batch * 0.8], dim=-1)


def test_sepformer_wrapper_returns_loudest_source_first() -> None:
    pytest.importorskip("torch")
    audio = AudioData.from_iterable(tone(300, 0.5).tolist(), SR)

    primary = create_separator({"name": "sepformer", "params": {"model": FakeSepformer(), "device": "cpu"}})
    (only,) = primary.separate(audio)
    assert only.metadata["source_index"] == 1
    assert only.sample_rate == SR and only.frame_count == audio.frame_count

    everything = create_separator(
        {"name": "sepformer", "params": {"model": FakeSepformer(), "device": "cpu", "output": "all"}}
    )
    sources = everything.separate(audio)
    assert [s.metadata["source_index"] for s in sources] == [1, 0]


def test_sepformer_wrapper_resamples_to_model_rate() -> None:
    pytest.importorskip("torchaudio")
    model = FakeSepformer()
    model.hparams = SimpleNamespace(sample_rate=8_000, num_spks=2)
    separator = create_separator({"name": "sepformer", "params": {"model": model, "device": "cpu", "output": "all"}})
    sources = separator.separate_array(tone(300, 0.5), SR)
    assert len(sources) == 2 and all(len(s) == SR // 2 for s in sources)


def test_pit_loss_is_permutation_invariant() -> None:
    torch = pytest.importorskip("torch")
    from src.separation.train import pit_si_snr_loss

    a, b = torch.from_numpy(tone(220)), torch.from_numpy(tone(570))
    targets = torch.stack([a, b], dim=-1).unsqueeze(0)
    swapped = torch.stack([b, a], dim=-1).unsqueeze(0)
    assert float(pit_si_snr_loss(swapped, targets)) < -50
    assert float(pit_si_snr_loss(swapped, targets)) == pytest.approx(float(pit_si_snr_loss(targets, targets)))


def test_invalid_output_mode_rejected() -> None:
    with pytest.raises(ValueError):
        create_separator({"name": "sepformer", "params": {"model": FakeSepformer(), "output": "nope"}})
