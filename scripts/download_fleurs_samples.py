from __future__ import annotations

import argparse
import csv
from itertools import islice
from pathlib import Path
import io
import sys
import soundfile as sf
from datasets import Audio, load_dataset


# Ensure UTF-8 output encoding for console
if sys.stdout:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Vietnamese audio samples from FLEURS."
    )
    parser.add_argument(
        "--count",
        type=int,
        default=20,
        help="Number of samples to download.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "validation", "test"],
        default="test",
        help="Dataset split.",
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
) -> None:
    if count <= 0:
        raise ValueError("count must be greater than zero")

    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / "metadata.csv"

    print("Connecting to FLEURS dataset...")

    dataset = load_dataset(
        "google/fleurs",
        "vi_vn",
        split=split,
        streaming=True,
    )
    dataset = dataset.cast_column("audio", Audio(decode=False))

    metadata_rows: list[dict[str, str]] = []

    for index, sample in enumerate(islice(dataset, count)):
        audio_info = sample["audio"]
        if "bytes" in audio_info and audio_info["bytes"]:
            samples, sample_rate = sf.read(io.BytesIO(audio_info["bytes"]))
        elif "path" in audio_info and audio_info["path"]:
            samples, sample_rate = sf.read(audio_info["path"])
        else:
            raise ValueError(f"Failed to read audio from sample {index}")

        transcription = sample["transcription"].strip()

        filename = f"sample_{index:03d}.wav"
        audio_path = audio_dir / filename

        sf.write(
            audio_path,
            samples,
            sample_rate,
            subtype="PCM_16",
        )

        relative_audio_path = Path("audio") / filename

        metadata_rows.append(
            {
                "audio_path": relative_audio_path.as_posix(),
                "transcription": transcription,
            }
        )

        print(f"[{index + 1}/{count}] Saved {audio_path}")
        print(f"    Transcript: {transcription}")

    with metadata_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as metadata_file:
        writer = csv.DictWriter(
            metadata_file,
            fieldnames=["audio_path", "transcription"],
        )
        writer.writeheader()
        writer.writerows(metadata_rows)

    print()
    print(f"Completed: {len(metadata_rows)} samples")
    print(f"Audio directory: {audio_dir}")
    print(f"Metadata file: {metadata_path}")


def main() -> None:
    args = parse_args()

    download_samples(
        count=args.count,
        split=args.split,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

#python scripts\download_fleurs_samples.py --count 5