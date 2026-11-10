from __future__ import annotations

import numpy as np

from scripts.download_fleurs_samples import METADATA_FIELDS, SAMPLE_RATE, normalize_gender, prepare_audio


def test_prepare_audio_converts_stereo_8k_to_mono_16k_float32() -> None:
    time = np.arange(800, dtype=np.float32) / 8_000
    stereo = np.stack(
        [
            np.sin(2 * np.pi * 220 * time),
            np.sin(2 * np.pi * 330 * time),
        ],
        axis=1,
    )

    converted = prepare_audio(stereo, 8_000)

    assert converted.dtype == np.float32
    assert converted.ndim == 1
    assert len(converted) == SAMPLE_RATE // 10
    assert METADATA_FIELDS == (
        "id",
        "audio_path",
        "transcription",
        "speaker_id",
        "gender",
        "speaker_id_source",
        "role",
        "split",
    )


def test_normalize_gender_handles_numeric_fleurs_labels() -> None:
    assert normalize_gender(0) == "male"
    assert normalize_gender(1) == "female"
    assert normalize_gender("Female") == "female"

