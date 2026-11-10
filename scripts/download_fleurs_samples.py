from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.wav_utils import resample_linear  # noqa: E402

SAMPLE_RATE = 16_000
METADATA_FIELDS = (
    "id",
    "audio_path",
    "transcription",
    "speaker_id",
    "gender",
    "speaker_id_source",
    "role",
    "split",
)


# Ensure UTF-8 output encoding for console
if sys.stdout:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def prepare_audio(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """Chuyển waveform FLEURS về mono, 16 kHz, float32."""

    signal = np.asarray(samples, dtype=np.float32)
    if signal.ndim == 2:
        # soundfile trả về frames x channels.
        signal = signal.mean(axis=1)
    if signal.ndim != 1 or not signal.size:
        raise ValueError(f"Audio must be non-empty mono/stereo, got shape={signal.shape}")
    if not np.all(np.isfinite(signal)):
        raise ValueError("Audio contains NaN or infinity")
    return resample_linear(signal, sample_rate, SAMPLE_RATE).astype(np.float32, copy=False)


def normalize_gender(value: object, feature: object | None = None) -> str:
    """Đổi nhãn ClassLabel của Hugging Face thành chuỗi ổn định."""

    if isinstance(value, (int, np.integer)) and feature is not None:
        int2str = getattr(feature, "int2str", None)
        if callable(int2str):
            value = int2str(int(value))
    gender = str(value).strip().lower()
    aliases = {"0": "male", "1": "female", "2": "other"}
    return aliases.get(gender, gender)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Vietnamese audio samples from FLEURS."
    )
    parser.add_argument(
        "--count",
        type=int,
        default=20,
        help="Number of target samples to download.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "validation", "test"],
        default="test",
        help="Dataset split.",
    )
    parser.add_argument(
        "--interferer-count",
        type=int,
        default=100,
        help="Number of female FLEURS-train samples used only as interferers.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/samples/fleurs_vi"),
        help="Directory used to save audio and metadata.",
    )
    return parser.parse_args()


def download_samples(
    count: int,
    split: str,
    output_dir: Path,
    interferer_count: int = 100,
) -> None:
    if count <= 0:
        raise ValueError("count must be greater than zero")
    if interferer_count <= 0:
        raise ValueError("interferer_count must be greater than zero")

    metadata_path = output_dir / "metadata.csv"
    if metadata_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing metadata: {metadata_path}")

    print("Connecting to FLEURS dataset...")
    metadata_rows: list[dict[str, str]] = []

    def download_role(
        *, role: str, source_split: str, requested_count: int, required_gender: str | None
    ) -> None:
        dataset = load_dataset(
            "google/fleurs",
            "vi_vn",
            split=source_split,
            streaming=True,
        )
        gender_feature = dataset.features.get("gender")
        dataset = dataset.cast_column("audio", Audio(decode=False))
        audio_dir = output_dir / role / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        accepted = 0

        for source_index, sample in enumerate(dataset):
            gender = normalize_gender(sample.get("gender", ""), gender_feature)
            if required_gender is not None and gender != required_gender:
                continue

            audio_info = sample["audio"]
            if audio_info.get("bytes"):
                samples, sample_rate = sf.read(io.BytesIO(audio_info["bytes"]))
            elif audio_info.get("path"):
                samples, sample_rate = sf.read(audio_info["path"])
            else:
                raise ValueError(f"Failed to read {source_split} sample {source_index}")

            samples = prepare_audio(samples, sample_rate)
            speaker_id = sample.get("speaker_id")
            filename = f"{role}_{accepted:04d}.wav"
            audio_path = audio_dir / filename
            sf.write(audio_path, samples, SAMPLE_RATE, subtype="PCM_16")

            metadata_rows.append(
                {
                    "id": f"{role}_{sample.get('id', source_index)}",
                    "audio_path": (Path(role) / "audio" / filename).as_posix(),
                    "transcription": sample["transcription"].strip(),
                    "speaker_id": "" if speaker_id is None else str(speaker_id),
                    "gender": gender,
                    "speaker_id_source": (
                        "fleurs" if speaker_id is not None else "unavailable_in_fleurs_v2"
                    ),
                    "role": role,
                    "split": source_split,
                }
            )
            accepted += 1
            print(f"[{role} {accepted}/{requested_count}] Saved {audio_path}")
            if accepted == requested_count:
                break

        if accepted != requested_count:
            condition = f" with gender={required_gender}" if required_gender else ""
            raise RuntimeError(
                f"Requested {requested_count} samples from {source_split}{condition}, "
                f"but only found {accepted}"
            )

    print(
        "FLEURS v2 không có speaker_id: target lấy từ test; interferer lấy từ "
        "train và lọc female để bảo đảm khác người."
    )
    download_role(role="target", source_split=split, requested_count=count, required_gender=None)
    download_role(
        role="interferer",
        source_split="train",
        requested_count=interferer_count,
        required_gender="female",
    )

    with metadata_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as metadata_file:
        writer = csv.DictWriter(
            metadata_file,
            fieldnames=METADATA_FIELDS,
        )
        writer.writeheader()
        writer.writerows(metadata_rows)

    print()
    print(f"Completed: {count} target + {interferer_count} interferer samples")
    print(f"Data directory: {output_dir}")
    print(f"Metadata file: {metadata_path}")


def main() -> None:
    args = parse_args()

    download_samples(
        count=args.count,
        split=args.split,
        output_dir=args.output_dir,
        interferer_count=args.interferer_count,
    )


if __name__ == "__main__":
    main()

#python scripts\download_fleurs_samples.py --count 5
