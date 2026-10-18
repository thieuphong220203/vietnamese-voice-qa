import argparse
import csv
import json
import shutil
import tarfile
import wave
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract and explore the VIVOS dataset.")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--extract-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=6)
    return parser.parse_args()


def extract_archive(archive: Path, extract_dir: Path) -> None:
    marker = extract_dir / ".extraction_complete"
    if marker.exists():
        print(f"Dataset already extracted: {extract_dir}")
        return

    extract_dir.mkdir(parents=True, exist_ok=True)
    root = extract_dir.resolve()

    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            destination = (extract_dir / member.name).resolve()
            if root not in destination.parents and destination != root:
                raise ValueError(f"Unsafe path in archive: {member.name}")
        tar.extractall(extract_dir)

    marker.touch()
    print(f"Extracted dataset: {extract_dir}")


def infer_split(path: Path) -> str:
    lowered = [part.lower() for part in path.parts]
    for split in ("train", "test", "dev", "valid", "validation"):
        if split in lowered:
            return split
    return "unknown"


def read_wav_info(path: Path) -> dict:
    with wave.open(str(path), "rb") as wav_file:
        frames = wav_file.getnframes()
        sample_rate = wav_file.getframerate()
        return {
            "path": str(path),
            "split": infer_split(path),
            "speaker": path.parent.name,
            "sample_rate": sample_rate,
            "channels": wav_file.getnchannels(),
            "sample_width_bytes": wav_file.getsampwidth(),
            "frames": frames,
            "duration_seconds": frames / sample_rate if sample_rate else 0.0,
        }


def load_transcripts(root: Path) -> dict[str, str]:
    transcripts: dict[str, str] = {}
    candidates = [
        path
        for path in root.rglob("*")
        if path.is_file() and "prompt" in path.name.lower()
    ]

    for path in candidates:
        for encoding in ("utf-8", "utf-8-sig"):
            try:
                lines = path.read_text(encoding=encoding).splitlines()
                break
            except UnicodeDecodeError:
                lines = []
        for line in lines:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                transcripts[parts[0]] = parts[1]
    return transcripts


def save_metadata(rows: list[dict], transcripts: dict[str, str], path: Path) -> None:
    fieldnames = [
        "path",
        "split",
        "speaker",
        "sample_rate",
        "channels",
        "sample_width_bytes",
        "frames",
        "duration_seconds",
        "transcript",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "transcript": transcripts.get(Path(row["path"]).stem, "")})


def save_plots(rows: list[dict], sample_paths: list[Path], output_dir: Path) -> None:
    durations = [row["duration_seconds"] for row in rows]
    plt.figure(figsize=(9, 5))
    plt.hist(durations, bins=50, color="#2878B5", edgecolor="white")
    plt.xlabel("Duration (seconds)")
    plt.ylabel("Number of audio files")
    plt.title("VIVOS utterance duration distribution")
    plt.tight_layout()
    plt.savefig(output_dir / "duration_histogram.png", dpi=150)
    plt.close()

    figure, axes = plt.subplots(len(sample_paths), 1, figsize=(12, 2.5 * len(sample_paths)))
    axes = np.atleast_1d(axes)
    for axis, path in zip(axes, sample_paths):
        with wave.open(str(path), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            sample_width = wav_file.getsampwidth()
            channels = wav_file.getnchannels()
            frames = wav_file.readframes(wav_file.getnframes())

        dtype = {1: np.uint8, 2: np.int16, 4: np.int32}.get(sample_width)
        if dtype is None:
            axis.text(0.5, 0.5, f"Unsupported sample width: {sample_width}", ha="center")
            continue
        samples = np.frombuffer(frames, dtype=dtype)
        if channels > 1:
            samples = samples.reshape(-1, channels)[:, 0]
        seconds = np.arange(len(samples)) / sample_rate
        axis.plot(seconds, samples, linewidth=0.5)
        axis.set_title(path.name)
        axis.set_xlabel("Seconds")
    figure.tight_layout()
    figure.savefig(output_dir / "sample_waveforms.png", dpi=150)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    extract_archive(args.archive, args.extract_dir)

    wav_paths = sorted(args.extract_dir.rglob("*.wav"))
    if not wav_paths:
        raise FileNotFoundError(f"No WAV files found under {args.extract_dir}")

    rows = []
    unreadable = []
    for path in wav_paths:
        try:
            rows.append(read_wav_info(path))
        except (wave.Error, EOFError) as error:
            unreadable.append({"path": str(path), "error": str(error)})

    transcripts = load_transcripts(args.extract_dir)
    durations = np.array([row["duration_seconds"] for row in rows], dtype=float)
    sample_rates = Counter(row["sample_rate"] for row in rows)
    channels = Counter(row["channels"] for row in rows)
    splits = Counter(row["split"] for row in rows)
    speakers = {row["speaker"] for row in rows}

    summary = {
        "archive": str(args.archive),
        "audio_files": len(rows),
        "unreadable_audio_files": len(unreadable),
        "speakers": len(speakers),
        "transcripts": len(transcripts),
        "total_hours": round(float(durations.sum() / 3600), 3),
        "duration_seconds": {
            "minimum": round(float(durations.min()), 3),
            "median": round(float(np.median(durations)), 3),
            "mean": round(float(durations.mean()), 3),
            "maximum": round(float(durations.max()), 3),
        },
        "files_by_split": dict(splits),
        "sample_rates": {str(key): value for key, value in sample_rates.items()},
        "channels": {str(key): value for key, value in channels.items()},
    }

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "unreadable_files.json").write_text(
        json.dumps(unreadable, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_metadata(rows, transcripts, args.output_dir / "metadata.csv")

    sample_paths = [Path(row["path"]) for row in rows[: max(1, args.num_samples)]]
    samples_dir = args.output_dir / "audio_samples"
    samples_dir.mkdir(exist_ok=True)
    for path in sample_paths:
        shutil.copy2(path, samples_dir / path.name)
    save_plots(rows, sample_paths, args.output_dir)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Reports saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
