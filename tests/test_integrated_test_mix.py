from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import pytest

from scripts.make_integrated_test_mix import (
    SAMPLE_RATE,
    generate_dataset,
    level_difference_db,
    make_integrated_mixture,
    signal_rms,
    snr_db,
)
from src.data.wav_utils import read_wav, write_wav


def tone(frequency: float, seconds: float = 0.25, amplitude: float = 0.4) -> np.ndarray:
    time = np.arange(round(seconds * SAMPLE_RATE), dtype=np.float32) / SAMPLE_RATE
    return (amplitude * np.sin(2.0 * np.pi * frequency * time)).astype(np.float32)


def test_integrated_mixture_hits_requested_gain_and_snr_without_clipping() -> None:
    result = make_integrated_mixture(
        tone(220, 0.25),
        tone(410, 0.18),
        np.random.default_rng(7).standard_normal(SAMPLE_RATE).astype(np.float32),
        gain_db=3.25,
        requested_snr_db=5.0,
        rng=random.Random(9),
    )

    assert result.actual_gain_db == pytest.approx(3.25, abs=1e-3)
    assert result.actual_snr_db == pytest.approx(5.0, abs=1e-3)
    assert level_difference_db(result.target, result.interferer) == pytest.approx(3.25, abs=1e-3)
    assert snr_db(result.target + result.interferer, result.noise) == pytest.approx(5.0, abs=1e-3)
    assert np.allclose(result.mix, result.target + result.interferer + result.noise, atol=1e-6)
    assert max(float(np.max(np.abs(signal))) for signal in (result.mix, result.target, result.interferer, result.noise)) <= 0.95 + 1e-6


def _make_fake_fleurs(root: Path) -> None:
    audio_dir = root / "audio"
    rows = []
    for speaker_index, frequency in enumerate((180, 260, 340, 420)):
        for utterance_index in range(2):
            path = audio_dir / f"spk{speaker_index}_{utterance_index}.wav"
            write_wav(path, tone(frequency + utterance_index * 10, 0.15 + utterance_index * 0.03), SAMPLE_RATE)
            rows.append(
                {
                    "id": f"utt_{speaker_index}_{utterance_index}",
                    "audio_path": f"audio/{path.name}",
                    "transcription": f"câu thử nghiệm {speaker_index} {utterance_index}",
                    "speaker_id": f"spk{speaker_index}",
                }
            )
    root.mkdir(parents=True, exist_ok=True)
    with (root / "metadata.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def test_generate_dataset_balances_snr_validates_audio_and_refuses_overwrite(tmp_path: Path) -> None:
    fleurs_root = tmp_path / "fleurs"
    noise_root = tmp_path / "demand"
    output_dir = tmp_path / "integrated"
    _make_fake_fleurs(fleurs_root)
    write_wav(noise_root / "PCAFE_ch01.wav", np.random.default_rng(1).standard_normal(SAMPLE_RATE), SAMPLE_RATE)
    write_wav(noise_root / "TCAR_ch01.wav", np.random.default_rng(2).standard_normal(SAMPLE_RATE), SAMPLE_RATE)

    metadata = generate_dataset(
        fleurs_root=fleurs_root,
        noise_root=noise_root,
        output_dir=output_dir,
        count=60,
        snr_levels=(0.0, 5.0, 10.0),
        seed=42,
    )
    with metadata.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 60
    assert {level: sum(float(row["snr_db"]) == level for row in rows) for level in (0.0, 5.0, 10.0)} == {
        0.0: 20,
        5.0: 20,
        10.0: 20,
    }
    assert all(row["target_speaker"] != row["interferer_speaker"] for row in rows)
    for row in rows:
        mix, rate = read_wav(output_dir / row["mix_path"])
        assert rate == SAMPLE_RATE and mix.size and np.all(np.isfinite(mix))
        assert float(np.max(np.abs(mix))) < 0.999

    with pytest.raises(FileExistsError, match="không ghi đè"):
        generate_dataset(
            fleurs_root=fleurs_root,
            noise_root=noise_root,
            output_dir=output_dir,
            count=60,
            snr_levels=(0.0, 5.0, 10.0),
            seed=42,
        )


def test_signal_rms_rejects_no_information_only_through_mixer() -> None:
    assert signal_rms(np.zeros(10, dtype=np.float32)) == 0.0
    with pytest.raises(ValueError, match="im lặng"):
        make_integrated_mixture(
            np.zeros(10, dtype=np.float32),
            np.ones(10, dtype=np.float32),
            np.ones(10, dtype=np.float32),
            gain_db=1.0,
            requested_snr_db=0.0,
            rng=random.Random(0),
        )

