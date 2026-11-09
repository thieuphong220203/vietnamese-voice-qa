"""Tạo tập kiểm thử tích hợp FLEURS vi-VN: hai người nói chồng lấn + DEMAND noise.

Mỗi hỗn hợp chứa một người nói mục tiêu, một người nói gây nhiễu khác speaker và
một đoạn tạp âm môi trường.  Đầu ra dùng chung cho thí nghiệm ablation C0--C3
(không tiền xử lý, SepFormer, DTLN, và SepFormer + DTLN).

``--fleurs-root`` chấp nhận một trong các dạng sau:

* FLEURS test được lưu bởi ``datasets.Dataset.save_to_disk``;
* thư mục chứa parquet của split test;
* ``metadata.csv``/``test.csv``/``test.tsv`` có cột đường dẫn audio,
  transcript và ``speaker_id``.

Script cố ý không tải hoặc thay thế DEMAND bằng MUSAN/noise mô phỏng.  Mọi file
noise phải tồn tại sẵn trong ``--noise-root`` để nguồn kiểm thử được kiểm soát.
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import random
import shutil
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.wav_utils import read_wav, resample_linear, write_wav  # noqa: E402

SAMPLE_RATE = 16_000
MAX_PEAK = 0.95
TARGET_DBFS = -25.0
EPSILON = 1e-12

CSV_FIELDS = (
    "id",
    "mix_path",
    "target_path",
    "interferer_path",
    "noise_path",
    "transcript",
    "target_speaker",
    "interferer_speaker",
    "snr_db",
    "gain_db",
    "actual_snr_db",
    "actual_gain_db",
    "noise_type",
    "duration",
)


@dataclass(frozen=True)
class FleursUtterance:
    utterance_id: str
    speaker: str
    transcript: str
    samples: np.ndarray


@dataclass(frozen=True)
class NoiseRecording:
    path: Path
    noise_type: str


@dataclass(frozen=True)
class IntegratedMix:
    mix: np.ndarray
    target: np.ndarray
    interferer: np.ndarray
    noise: np.ndarray
    actual_snr_db: float
    actual_gain_db: float
    scale_factor: float


def signal_rms(signal: np.ndarray) -> float:
    """RMS toàn đoạn, gồm cả phần đệm im lặng."""

    array = np.asarray(signal, dtype=np.float64)
    return float(np.sqrt(np.mean(array**2))) if array.size else 0.0


def level_difference_db(louder: np.ndarray, quieter: np.ndarray) -> float:
    return 20.0 * math.log10((signal_rms(louder) + EPSILON) / (signal_rms(quieter) + EPSILON))


def snr_db(speech: np.ndarray, noise: np.ndarray) -> float:
    return 20.0 * math.log10((signal_rms(speech) + EPSILON) / (signal_rms(noise) + EPSILON))


def _mono_float32(samples: Any) -> np.ndarray:
    array = np.asarray(samples, dtype=np.float32)
    if array.ndim == 2:
        # soundfile: frames x channels; một số decoder: channels x frames.
        axis = 0 if array.shape[0] <= 8 < array.shape[1] else 1
        array = array.mean(axis=axis)
    if array.ndim != 1:
        raise ValueError(f"audio phải là mono/2-D, nhận shape={array.shape}")
    if not array.size or not np.all(np.isfinite(array)):
        raise ValueError("audio rỗng hoặc chứa NaN/Inf")
    return array.astype(np.float32, copy=False)


def _decode_audio(value: Any, fallback_path: str | Path | None = None) -> tuple[np.ndarray, int]:
    """Giải mã audio từ Hugging Face row, bytes, ndarray hoặc đường dẫn."""

    if isinstance(value, Mapping):
        if value.get("array") is not None:
            return _mono_float32(value["array"]), int(value.get("sampling_rate", SAMPLE_RATE))
        if value.get("bytes") is not None:
            import soundfile as sf

            samples, rate = sf.read(io.BytesIO(value["bytes"]), dtype="float32", always_2d=False)
            return _mono_float32(samples), int(rate)
        if value.get("path"):
            fallback_path = value["path"]
    elif isinstance(value, np.ndarray):
        return _mono_float32(value), SAMPLE_RATE
    elif hasattr(value, "get_all_samples"):
        decoded = value.get_all_samples()
        samples = getattr(decoded, "data", getattr(decoded, "samples", None))
        rate = getattr(decoded, "sample_rate", SAMPLE_RATE)
        if hasattr(samples, "detach"):
            samples = samples.detach().cpu().numpy()
        return _mono_float32(samples), int(rate)
    elif isinstance(value, (str, Path)):
        fallback_path = value

    if fallback_path is None:
        raise ValueError("không tìm thấy dữ liệu/đường dẫn audio trong FLEURS row")
    path = Path(fallback_path)
    try:
        samples, rate = read_wav(path)
    except Exception:
        import soundfile as sf

        samples, rate = sf.read(path, dtype="float32", always_2d=False)
    return _mono_float32(samples), int(rate)


def _row_to_utterance(row: Mapping[str, Any], index: int, base_dir: Path | None = None) -> FleursUtterance:
    speaker = row.get("speaker_id", row.get("speaker", row.get("speaker_name")))
    if speaker is None or str(speaker).strip() == "":
        raise ValueError(
            "FLEURS metadata thiếu speaker_id; không thể chứng minh hai người nói khác nhau. "
            "Hãy dùng split test chính thức từ google/fleurs."
        )
    transcript = row.get("transcription", row.get("transcript", row.get("text", "")))
    if not str(transcript).strip():
        raise ValueError(f"FLEURS row {index} thiếu transcript")

    audio_value = row.get("audio")
    path_value = row.get("audio_path", row.get("path", row.get("file")))
    if path_value and base_dir and not Path(str(path_value)).is_absolute():
        first = base_dir / str(path_value)
        path_value = first if first.exists() else PROJECT_ROOT / str(path_value)
    samples, rate = _decode_audio(audio_value, path_value)
    samples = resample_linear(samples, rate, SAMPLE_RATE)
    utterance_id = str(row.get("id", Path(str(path_value)).stem if path_value else index))
    return FleursUtterance(utterance_id, str(speaker), str(transcript).strip(), samples)


def _utterances_from_dataset(dataset: Any) -> list[FleursUtterance]:
    column_names = getattr(dataset, "column_names", None)
    if isinstance(column_names, Mapping):
        if "test" not in dataset:
            raise ValueError("DatasetDict FLEURS không có split test")
        dataset = dataset["test"]

    if "speaker_id" not in dataset.column_names:
        raise ValueError("FLEURS dataset thiếu cột speaker_id")
    if "audio" in dataset.column_names:
        try:
            from datasets import Audio

            dataset = dataset.cast_column("audio", Audio(decode=False))
        except Exception:
            pass
    return [_row_to_utterance(dataset[index], index) for index in range(len(dataset))]


def _load_tabular_metadata(metadata_path: Path) -> list[FleursUtterance]:
    delimiter = "\t" if metadata_path.suffix.lower() == ".tsv" else ","
    with metadata_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter=delimiter))
    if not rows:
        raise ValueError(f"metadata rỗng: {metadata_path}")
    return [_row_to_utterance(row, index, metadata_path.parent) for index, row in enumerate(rows)]


def load_fleurs_test(fleurs_root: Path) -> list[FleursUtterance]:
    """Nạp split test FLEURS và bắt buộc giữ speaker_id thật."""

    root = fleurs_root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"không tìm thấy --fleurs-root: {root}")

    saved_dataset_markers = (root / "state.json", root / "dataset_dict.json")
    if any(marker.exists() for marker in saved_dataset_markers):
        from datasets import load_from_disk

        utterances = _utterances_from_dataset(load_from_disk(str(root)))
    else:
        metadata_candidates = [
            root / "test.csv",
            root / "test.tsv",
            root / "metadata.csv",
            root / "metadata.tsv",
        ]
        metadata = next((path for path in metadata_candidates if path.exists()), None)
        if metadata:
            utterances = _load_tabular_metadata(metadata)
        else:
            parquet_files = sorted(root.rglob("*test*.parquet"))
            if not parquet_files:
                raise FileNotFoundError(
                    f"{root} không phải Dataset.save_to_disk và không có test CSV/TSV/parquet"
                )
            from datasets import load_dataset

            dataset = load_dataset("parquet", data_files=[str(path) for path in parquet_files], split="train")
            utterances = _utterances_from_dataset(dataset)

    speakers = {utterance.speaker for utterance in utterances}
    if len(speakers) < 2:
        raise ValueError(f"FLEURS test cần ít nhất 2 speaker, chỉ tìm thấy {len(speakers)}")
    print(f"FLEURS test: {len(utterances)} câu, {len(speakers)} speaker")
    return utterances


def _noise_type(path: Path) -> str:
    upper = "_".join(part.upper() for part in (*path.parts[-3:-1], path.stem))
    known = (
        "PCAFE",
        "OOFFICE",
        "PRESTAU",
        "PRESTO",
        "STRAFFIC",
        "TCAR",
        "SPSQUARE",
        "TBUS",
        "TMETRO",
        "DWASHING",
        "DKITCHEN",
        "DLIVING",
        "NFIELD",
        "NPARK",
        "NRIVER",
    )
    environment = next((name for name in known if name in upper), path.stem.upper())
    return f"DEMAND_{environment}"


def discover_demand_noise(noise_root: Path) -> list[NoiseRecording]:
    root = noise_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"không tìm thấy --noise-root: {root}")
    paths = sorted(path for suffix in ("*.wav", "*.flac", "*.ogg") for path in root.rglob(suffix))
    paths = [path for path in paths if "musan" not in str(path).lower()]
    if not paths:
        raise FileNotFoundError(f"không có file DEMAND audio trong {root}")
    recordings = [NoiseRecording(path, _noise_type(path)) for path in paths]
    print(f"DEMAND: {len(recordings)} file, {len({item.noise_type for item in recordings})} loại noise")
    return recordings


def load_audio_16k(path: Path) -> np.ndarray:
    samples, rate = _decode_audio(path)
    return resample_linear(samples, rate, SAMPLE_RATE).astype(np.float32)


def _pad_to_length(signal: np.ndarray, length: int) -> np.ndarray:
    if len(signal) >= length:
        return signal[:length].astype(np.float32, copy=False)
    return np.pad(signal, (0, length - len(signal))).astype(np.float32)


def _scale_to_rms(signal: np.ndarray, desired_rms: float) -> np.ndarray:
    current = signal_rms(signal)
    if current <= EPSILON:
        raise ValueError("không thể scale tín hiệu im lặng")
    return (np.asarray(signal, dtype=np.float32) * (desired_rms / current)).astype(np.float32)


def _noise_segment(noise: np.ndarray, length: int, rng: random.Random) -> np.ndarray:
    if not len(noise):
        raise ValueError("noise rỗng")
    noise = np.asarray(noise, dtype=np.float32)
    if len(noise) < length:
        noise = np.tile(noise, math.ceil(length / len(noise)))
    max_start = len(noise) - length
    start = rng.randint(0, max_start) if max_start else 0
    segment = noise[start : start + length].copy()
    segment -= float(np.mean(segment))
    if signal_rms(segment) <= EPSILON:
        raise ValueError("đoạn noise được chọn là im lặng")
    return segment


def make_integrated_mixture(
    target: np.ndarray,
    interferer: np.ndarray,
    noise: np.ndarray,
    *,
    gain_db: float,
    requested_snr_db: float,
    rng: random.Random,
    max_peak: float = MAX_PEAK,
) -> IntegratedMix:
    """Trộn hai giọng và noise, rồi nhân cùng một hệ số để chống clipping."""

    if not 0.0 <= gain_db <= 5.0:
        raise ValueError("gain_db phải nằm trong [0, 5]")
    target = _mono_float32(target)
    interferer = _mono_float32(interferer)
    length = max(len(target), len(interferer))
    target = _pad_to_length(target, length)
    interferer = _pad_to_length(interferer, length)

    target_rms = 10.0 ** (TARGET_DBFS / 20.0)
    target = _scale_to_rms(target, target_rms)
    interferer = _scale_to_rms(interferer, target_rms / (10.0 ** (gain_db / 20.0)))
    speech = target + interferer

    noise = _noise_segment(_mono_float32(noise), length, rng)
    desired_noise_rms = signal_rms(speech) / (10.0 ** (requested_snr_db / 20.0))
    noise = _scale_to_rms(noise, desired_noise_rms)
    mixture = speech + noise

    peak = max(float(np.max(np.abs(item))) for item in (mixture, target, interferer, noise))
    scale = min(1.0, max_peak / max(peak, EPSILON))
    mixture = (mixture * scale).astype(np.float32)
    target = (target * scale).astype(np.float32)
    interferer = (interferer * scale).astype(np.float32)
    noise = (noise * scale).astype(np.float32)

    return IntegratedMix(
        mix=mixture,
        target=target,
        interferer=interferer,
        noise=noise,
        actual_snr_db=snr_db(target + interferer, noise),
        actual_gain_db=level_difference_db(target, interferer),
        scale_factor=scale,
    )


def _safe_manifest_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    resolved_root = root.resolve()
    if path != resolved_root and resolved_root not in path.parents:
        raise ValueError(f"đường dẫn metadata không an toàn: {relative}")
    return path


def validate_dataset(
    output_dir: Path,
    *,
    expected_count: int,
    expected_snr_levels: Sequence[float],
    tolerance_db: float = 0.15,
) -> list[dict[str, str]]:
    """Đọc lại WAV đã ghi và xác nhận cân bằng, mức trộn và tính toàn vẹn."""

    metadata_path = output_dir / "metadata.csv"
    with metadata_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != expected_count:
        raise AssertionError(f"cần {expected_count} mẫu, metadata có {len(rows)}")

    per_level = expected_count // len(expected_snr_levels)
    observed_levels = Counter(round(float(row["snr_db"]), 6) for row in rows)
    expected_levels = Counter({round(float(level), 6): per_level for level in expected_snr_levels})
    if observed_levels != expected_levels:
        raise AssertionError(f"phân bố SNR sai: {dict(observed_levels)} != {dict(expected_levels)}")

    for row in rows:
        if row["target_speaker"] == row["interferer_speaker"]:
            raise AssertionError(f"{row['id']}: target và interferer cùng speaker")
        if not row["transcript"].strip():
            raise AssertionError(f"{row['id']}: transcript rỗng")

        signals: dict[str, np.ndarray] = {}
        rates = set()
        for field in ("mix_path", "target_path", "interferer_path", "noise_path"):
            path = _safe_manifest_path(output_dir, row[field])
            signal, rate = read_wav(path)
            if not signal.size or not np.all(np.isfinite(signal)):
                raise AssertionError(f"{row['id']}: {field} rỗng hoặc chứa NaN/Inf")
            if float(np.max(np.abs(signal))) >= 0.999:
                raise AssertionError(f"{row['id']}: {field} bị clipping")
            signals[field] = signal
            rates.add(rate)
        if rates != {SAMPLE_RATE}:
            raise AssertionError(f"{row['id']}: sample rate không đồng nhất: {rates}")
        if len({len(signal) for signal in signals.values()}) != 1:
            raise AssertionError(f"{row['id']}: độ dài các nguồn không đồng nhất")

        reconstructed = signals["target_path"] + signals["interferer_path"] + signals["noise_path"]
        if not np.allclose(signals["mix_path"], reconstructed, atol=1.1e-4):
            raise AssertionError(f"{row['id']}: mix khác tổng ba nguồn tham chiếu")
        measured_gain = level_difference_db(signals["target_path"], signals["interferer_path"])
        measured_snr = snr_db(
            signals["target_path"] + signals["interferer_path"], signals["noise_path"]
        )
        if abs(measured_gain - float(row["gain_db"])) > tolerance_db:
            raise AssertionError(f"{row['id']}: gain thực tế sai ({measured_gain:.3f} dB)")
        if abs(measured_snr - float(row["snr_db"])) > tolerance_db:
            raise AssertionError(f"{row['id']}: SNR thực tế sai ({measured_snr:.3f} dB)")
    return rows


def print_summary(rows: Iterable[Mapping[str, str]]) -> None:
    counts = Counter((float(row["snr_db"]), row["noise_type"]) for row in rows)
    print("\nTóm tắt theo SNR và loại nhiễu")
    print(f"{'SNR (dB)':>8}  {'Loại nhiễu':<24}  {'Số mẫu':>7}")
    print("-" * 44)
    for (level, noise_type), count in sorted(counts.items()):
        print(f"{level:>8.1f}  {noise_type:<24}  {count:>7}")


def generate_dataset(
    *,
    fleurs_root: Path,
    noise_root: Path,
    output_dir: Path,
    count: int = 60,
    snr_levels: Sequence[float] = (0.0, 5.0, 10.0),
    seed: int = 42,
) -> Path:
    if count <= 0:
        raise ValueError("count phải > 0")
    levels = tuple(float(level) for level in snr_levels)
    if not levels or len(set(levels)) != len(levels):
        raise ValueError("snr-levels phải khác nhau và không được rỗng")
    if count % len(levels):
        raise ValueError("count phải chia hết cho số mức SNR để phân bố cân bằng")

    destination = output_dir.resolve()
    if destination.exists():
        raise FileExistsError(f"không ghi đè dữ liệu hiện có: {destination}")

    utterances = load_fleurs_test(fleurs_root)
    noise_recordings = discover_demand_noise(noise_root)
    groups: dict[str, list[FleursUtterance]] = {}
    for utterance in utterances:
        groups.setdefault(utterance.speaker, []).append(utterance)
    speakers = sorted(groups)

    rng = random.Random(seed)
    schedule = [level for level in levels for _ in range(count // len(levels))]
    rng.shuffle(schedule)
    noise_cache: dict[Path, np.ndarray] = {}

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    staging = temp_root / destination.name
    try:
        for folder in ("mix", "target", "interferer", "noise"):
            (staging / folder).mkdir(parents=True, exist_ok=True)

        rows: list[dict[str, str]] = []
        for index, requested_snr in enumerate(schedule):
            target_speaker, interferer_speaker = rng.sample(speakers, 2)
            target_utt = rng.choice(groups[target_speaker])
            interferer_utt = rng.choice(groups[interferer_speaker])
            gain = rng.uniform(0.0, 5.0)
            noise_recording = rng.choice(noise_recordings)
            if noise_recording.path not in noise_cache:
                noise_cache[noise_recording.path] = load_audio_16k(noise_recording.path)

            result = make_integrated_mixture(
                target_utt.samples,
                interferer_utt.samples,
                noise_cache[noise_recording.path],
                gain_db=gain,
                requested_snr_db=requested_snr,
                rng=rng,
            )
            sample_id = f"integrated_{index:04d}"
            relative_paths = {folder: f"{folder}/{sample_id}.wav" for folder in ("mix", "target", "interferer", "noise")}
            for folder, signal in (
                ("mix", result.mix),
                ("target", result.target),
                ("interferer", result.interferer),
                ("noise", result.noise),
            ):
                write_wav(staging / relative_paths[folder], signal, SAMPLE_RATE)

            rows.append(
                {
                    "id": sample_id,
                    "mix_path": relative_paths["mix"],
                    "target_path": relative_paths["target"],
                    "interferer_path": relative_paths["interferer"],
                    "noise_path": relative_paths["noise"],
                    "transcript": target_utt.transcript,
                    "target_speaker": target_speaker,
                    "interferer_speaker": interferer_speaker,
                    "snr_db": f"{requested_snr:.1f}",
                    "gain_db": f"{gain:.4f}",
                    "actual_snr_db": f"{result.actual_snr_db:.4f}",
                    "actual_gain_db": f"{result.actual_gain_db:.4f}",
                    "noise_type": noise_recording.noise_type,
                    "duration": f"{len(result.mix) / SAMPLE_RATE:.4f}",
                }
            )
            print(
                f"[{index + 1:02d}/{count}] {sample_id}: SNR {requested_snr:.1f}/{result.actual_snr_db:.2f} dB, "
                f"gain {gain:.2f}/{result.actual_gain_db:.2f} dB, {noise_recording.noise_type}"
            )

        metadata_path = staging / "metadata.csv"
        with metadata_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

        validated_rows = validate_dataset(
            staging,
            expected_count=count,
            expected_snr_levels=levels,
        )
        print_summary(validated_rows)
        staging.rename(destination)
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)

    print(f"\nĐã xác nhận {count} mẫu hợp lệ -> {destination}")
    print(f"Metadata: {destination / 'metadata.csv'}")
    return destination / "metadata.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--count", "-count", type=int, default=60)
    parser.add_argument("--snr-levels", type=float, nargs="+", default=(0.0, 5.0, 10.0))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fleurs-root", type=Path, default=PROJECT_ROOT / "data/raw/fleurs")
    parser.add_argument("--noise-root", type=Path, default=PROJECT_ROOT / "data/raw/demand")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/processed/integrated_test")
    return parser.parse_args()


def main() -> int:
    # Tránh lỗi cp1252 khi chạy CLI có thông báo tiếng Việt trên Windows.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    args = parse_args()
    generate_dataset(
        fleurs_root=args.fleurs_root,
        noise_root=args.noise_root,
        output_dir=args.output_dir,
        count=args.count,
        snr_levels=args.snr_levels,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

