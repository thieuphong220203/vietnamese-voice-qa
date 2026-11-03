"""Tạo dữ liệu tách giọng tiếng Việt bằng cách trộn 2 người nói từ VIVOS.

Cấu trúc VIVOS (bản gốc, giải nén từ vivos.tar.gz)::

    vivos/
      train/  waves/<SPEAKER>/<SPEAKER>_<ID>.wav   prompts.txt
      test/   waves/<SPEAKER>/<SPEAKER>_<ID>.wav   prompts.txt

``prompts.txt``: mỗi dòng ``<UTTERANCE_ID> <TRANSCRIPT>``.

Chia tập theo NGƯỜI NÓI để kết quả đánh giá trung thực:
- ``train``: người nói thuộc VIVOS/train (trừ vài người giữ lại làm valid)
- ``valid``: vài người nói giữ lại từ VIVOS/train (chọn checkpoint tốt nhất)
- ``test``: người nói thuộc VIVOS/test (không bao giờ dùng để train)

Đầu ra (``--output-dir``)::

    <split>/mix/<id>.wav  <split>/s1/<id>.wav  <split>/s2/<id>.wav
    <split>.csv   (id, mix, s1, s2, duration, gain_db, s1_speaker, s2_speaker, s1_text, s2_text)

Đường dẫn trong CSV là tương đối so với thư mục chứa CSV.
``s1`` luôn là giọng TO hơn (``gain_db`` >= 0) — đóng vai "người hỏi"
để đánh giá quy tắc chọn giọng chính của pipeline.

Ví dụ::

    python -m src.data.prepare_separation --vivos-root data/raw/vivos \\
        --output-dir data/processed/separation --n-train 4000 --n-valid 300 --n-test 300
"""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.data.wav_utils import read_wav, resample_linear, scale_to_dbfs, write_wav

SAMPLE_RATE = 16_000
CSV_FIELDS = (
    "id",
    "mix",
    "s1",
    "s2",
    "duration",
    "gain_db",
    "s1_speaker",
    "s2_speaker",
    "s1_text",
    "s2_text",
)


@dataclass(frozen=True)
class Utterance:
    path: Path
    speaker: str
    text: str


def load_prompts(prompts_path: Path) -> dict[str, str]:
    prompts: dict[str, str] = {}
    if not prompts_path.exists():
        return prompts
    for line in prompts_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split(maxsplit=1)
        if parts:
            prompts[parts[0]] = parts[1] if len(parts) > 1 else ""
    return prompts


def scan_vivos_split(split_dir: Path) -> list[Utterance]:
    """Đọc mọi file wav trong ``<split_dir>/waves/<speaker>/``."""

    waves_dir = split_dir / "waves"
    if not waves_dir.is_dir():
        raise FileNotFoundError(f"không tìm thấy {waves_dir}")
    prompts = load_prompts(split_dir / "prompts.txt")
    utterances = [
        Utterance(path=wav, speaker=wav.parent.name, text=prompts.get(wav.stem, ""))
        for wav in sorted(waves_dir.glob("*/*.wav"))
    ]
    if not utterances:
        raise FileNotFoundError(f"không có file wav nào trong {waves_dir}")
    return utterances


def group_by_speaker(utterances: list[Utterance]) -> dict[str, list[Utterance]]:
    groups: dict[str, list[Utterance]] = {}
    for utterance in utterances:
        groups.setdefault(utterance.speaker, []).append(utterance)
    return groups


def split_speakers(
    speakers: list[str], n_valid_speakers: int, rng: random.Random
) -> tuple[list[str], list[str]]:
    shuffled = sorted(speakers)
    rng.shuffle(shuffled)
    # valid cần >= 2 người nói để trộn; train cũng phải còn >= 2 người.
    n_valid = 0 if n_valid_speakers <= 0 else max(2, n_valid_speakers)
    n_valid = min(n_valid, max(0, len(shuffled) - 2))
    if n_valid < 2:
        n_valid = 0
    return sorted(shuffled[n_valid:]), sorted(shuffled[:n_valid])


def fit_length(signal: np.ndarray, length: int, rng: random.Random, random_crop: bool) -> np.ndarray:
    if len(signal) > length:
        start = rng.randint(0, len(signal) - length) if random_crop else 0
        return signal[start : start + length]
    if len(signal) < length:
        return np.pad(signal, (0, length - len(signal)))
    return signal


def make_mixture(
    s1: np.ndarray,
    s2: np.ndarray,
    gain_db: float,
    *,
    mode: str,
    segment_samples: int | None,
    rng: random.Random,
    level_dbfs: float = -25.0,
    max_peak: float = 0.9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trộn 2 giọng, s1 to hơn s2 ``gain_db`` dB. Trả về (mix, s1, s2) cùng độ dài.

    mode:
      - ``segment``: cắt/đệm cả hai về ``segment_samples`` (dùng cho train/valid).
      - ``max``: đệm giọng ngắn hơn bằng 0 tới độ dài giọng dài hơn — giữ trọn câu
        để còn tính WER trên transcript (dùng cho test).
      - ``min``: cắt cả hai về độ dài giọng ngắn hơn (kiểu WSJ0-2mix).
    """

    if mode == "segment":
        if not segment_samples:
            raise ValueError("mode=segment cần segment_samples")
        length = segment_samples
        s1 = fit_length(s1, length, rng, random_crop=True)
        s2 = fit_length(s2, length, rng, random_crop=True)
    elif mode == "max":
        length = max(len(s1), len(s2))
        s1, s2 = np.pad(s1, (0, length - len(s1))), np.pad(s2, (0, length - len(s2)))
    elif mode == "min":
        length = min(len(s1), len(s2))
        s1, s2 = s1[:length], s2[:length]
    else:
        raise ValueError(f"mode không hợp lệ: {mode}")

    # Đưa 2 giọng về cùng mức, rồi chia đều chênh lệch: s1 +g/2 dB, s2 -g/2 dB.
    s1 = scale_to_dbfs(s1, level_dbfs) * 10 ** (gain_db / 40)
    s2 = scale_to_dbfs(s2, level_dbfs) * 10 ** (-gain_db / 40)
    mix = s1 + s2

    peak = max(float(np.max(np.abs(x))) if x.size else 0.0 for x in (mix, s1, s2))
    if peak > max_peak:
        factor = max_peak / peak
        mix, s1, s2 = mix * factor, s1 * factor, s2 * factor
    return mix.astype(np.float32), s1.astype(np.float32), s2.astype(np.float32)


def load_utterance(utterance: Utterance) -> np.ndarray:
    signal, rate = read_wav(utterance.path)
    return resample_linear(signal, rate, SAMPLE_RATE)


def build_split(
    name: str,
    groups: dict[str, list[Utterance]],
    count: int,
    output_dir: Path,
    *,
    mode: str,
    segment_seconds: float,
    max_gain_db: float,
    rng: random.Random,
) -> Path:
    speakers = sorted(groups)
    if len(speakers) < 2:
        raise ValueError(f"split {name} cần ít nhất 2 người nói, chỉ có {len(speakers)}")

    segment_samples = round(segment_seconds * SAMPLE_RATE) if mode == "segment" else None
    split_dir = output_dir / name
    rows = []
    for index in range(count):
        spk1, spk2 = rng.sample(speakers, 2)
        utt1, utt2 = rng.choice(groups[spk1]), rng.choice(groups[spk2])
        gain_db = rng.uniform(0.0, max_gain_db)
        mix, s1, s2 = make_mixture(
            load_utterance(utt1),
            load_utterance(utt2),
            gain_db,
            mode=mode,
            segment_samples=segment_samples,
            rng=rng,
        )
        mix_id = f"{name}_{index:05d}"
        for folder, signal in (("mix", mix), ("s1", s1), ("s2", s2)):
            write_wav(split_dir / folder / f"{mix_id}.wav", signal, SAMPLE_RATE)
        rows.append(
            {
                "id": mix_id,
                "mix": f"{name}/mix/{mix_id}.wav",
                "s1": f"{name}/s1/{mix_id}.wav",
                "s2": f"{name}/s2/{mix_id}.wav",
                "duration": f"{len(mix) / SAMPLE_RATE:.3f}",
                "gain_db": f"{gain_db:.2f}",
                "s1_speaker": utt1.speaker,
                "s2_speaker": utt2.speaker,
                "s1_text": utt1.text,
                "s2_text": utt2.text,
            }
        )
        if (index + 1) % 500 == 0:
            print(f"  {name}: {index + 1}/{count}")

    csv_path = output_dir / f"{name}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Đã tạo {count} mẫu {name} ({len(speakers)} người nói) -> {csv_path}")
    return csv_path


def prepare(
    vivos_root: Path,
    output_dir: Path,
    *,
    n_train: int,
    n_valid: int,
    n_test: int,
    n_valid_speakers: int = 4,
    segment_seconds: float = 4.0,
    test_mode: str = "max",
    max_gain_db: float = 5.0,
    seed: int = 1234,
) -> dict[str, Path]:
    rng = random.Random(seed)
    train_groups = group_by_speaker(scan_vivos_split(vivos_root / "train"))
    train_speakers, valid_speakers = split_speakers(list(train_groups), n_valid_speakers, rng)

    outputs: dict[str, Path] = {}
    common = dict(segment_seconds=segment_seconds, max_gain_db=max_gain_db, rng=rng)
    if n_train:
        outputs["train"] = build_split(
            "train", {s: train_groups[s] for s in train_speakers}, n_train, output_dir,
            mode="segment", **common,
        )
    if n_valid and valid_speakers:
        outputs["valid"] = build_split(
            "valid", {s: train_groups[s] for s in valid_speakers}, n_valid, output_dir,
            mode="segment", **common,
        )
    if n_test:
        test_groups = group_by_speaker(scan_vivos_split(vivos_root / "test"))
        outputs["test"] = build_split(
            "test", test_groups, n_test, output_dir, mode=test_mode, **common,
        )
    return outputs


VIVOS_HF_REPO = "AILAB-VNUHCM/vivos"
VIVOS_HF_ARCHIVE = "data/vivos.tar.gz"  # file nén gốc nằm sẵn trong repo dataset
VIVOS_ZENODO_URL = "https://zenodo.org/record/7068130/files/vivos.tar.gz"


def _safe_extract(archive: Path, destination: Path) -> None:
    import tarfile

    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        root = destination.resolve()
        for member in tar.getmembers():
            target = (destination / member.name).resolve()
            if root not in target.parents and target != root:
                raise ValueError(f"đường dẫn không an toàn trong file nén: {member.name}")
            if not (member.isfile() or member.isdir()):
                raise ValueError(f"file nén chứa mục không phải file/thư mục: {member.name}")
        if hasattr(tarfile, "data_filter"):  # Python >= 3.12
            tar.extractall(destination, filter="data")
        else:
            tar.extractall(destination)


def _fetch_prompts(split_dir: Path, split: str) -> None:
    """Bổ sung prompts.txt nếu file nén không có (lấy bản .gz trong repo HF)."""

    target = split_dir / "prompts.txt"
    if target.exists():
        return
    import gzip

    from huggingface_hub import hf_hub_download

    gz_path = hf_hub_download(VIVOS_HF_REPO, f"data/prompts-{split}.txt.gz", repo_type="dataset")
    with gzip.open(gz_path, "rt", encoding="utf-8") as handle:
        target.write_text(handle.read(), encoding="utf-8")


def download_vivos(vivos_root: Path) -> Path:
    """Tải vivos.tar.gz (khoảng 1.4 GB) và giải nén thành ``vivos_root/{train,test}/waves/...``.

    Thử Hugging Face trước (cần ``huggingface_hub``), lỗi thì tải từ Zenodo.
    Không dùng ``datasets.load_dataset`` vì repo VIVOS dùng loading script,
    ``datasets`` >= 4 không còn hỗ trợ.
    """

    if (vivos_root / "train" / "waves").is_dir() and (vivos_root / "test" / "waves").is_dir():
        print(f"VIVOS đã có ở {vivos_root}, bỏ qua bước tải")
    else:
        try:
            from huggingface_hub import hf_hub_download

            archive = Path(hf_hub_download(VIVOS_HF_REPO, VIVOS_HF_ARCHIVE, repo_type="dataset"))
        except Exception as error:  # noqa: BLE001 — chuyển sang nguồn dự phòng
            import urllib.request

            print(f"Tải từ Hugging Face lỗi ({error}); thử Zenodo...")
            archive = vivos_root.parent / "vivos.tar.gz"
            archive.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(VIVOS_ZENODO_URL, archive)

        print(f"Giải nén {archive} ...")
        # File nén có thư mục gốc "vivos/", nên giải nén vào thư mục cha.
        _safe_extract(archive, vivos_root.parent)
        extracted = vivos_root.parent / "vivos"
        if extracted.resolve() != vivos_root.resolve():
            extracted.rename(vivos_root)

    for split in ("train", "test"):
        _fetch_prompts(vivos_root / split, split)
        count = sum(1 for _ in (vivos_root / split / "waves").glob("*/*.wav"))
        print(f"VIVOS {split}: {count} file wav")
    return vivos_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vivos-root", type=Path, default=Path("data/raw/vivos"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/separation"))
    parser.add_argument("--from-hf", "--download", dest="download", action="store_true",
                        help="tải VIVOS (Hugging Face, dự phòng Zenodo) vào --vivos-root trước")
    parser.add_argument("--n-train", type=int, default=4000)
    parser.add_argument("--n-valid", type=int, default=300)
    parser.add_argument("--n-test", type=int, default=300)
    parser.add_argument("--n-valid-speakers", type=int, default=4)
    parser.add_argument("--segment-seconds", type=float, default=4.0)
    parser.add_argument("--test-mode", choices=("max", "min", "segment"), default="max")
    parser.add_argument("--max-gain-db", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.download:
        download_vivos(args.vivos_root)
    prepare(
        args.vivos_root,
        args.output_dir,
        n_train=args.n_train,
        n_valid=args.n_valid,
        n_test=args.n_test,
        n_valid_speakers=args.n_valid_speakers,
        segment_seconds=args.segment_seconds,
        test_mode=args.test_mode,
        max_gain_db=args.max_gain_db,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
