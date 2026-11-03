"""Tạo vài mẫu trộn 2 giọng NHỎ để chạy thử separation trên máy cá nhân (không cần VIVOS).

Mặc định ghép từng cặp file FLEURS đã tải bằng ``scripts/download_fleurs_samples.py``
(sample_000 + sample_001, sample_002 + sample_003, ...). Kết quả có cùng định dạng
CSV với ``src.data.prepare_separation`` nên dùng thẳng được cho ``src.separation.evaluate``.

Ví dụ::

    python scripts/download_fleurs_samples.py --count 6
    python scripts/make_test_mix.py --pairs 3
    python -m src.separation.evaluate --csv data/processed/separation_demo/test.csv --tag demo

    # Hoặc trộn 2 file bất kỳ (ví dụ 2 bản ghi âm của 2 người):
    python scripts/make_test_mix.py --s1 nguoi_a.wav --s2 nguoi_b.wav
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.prepare_separation import CSV_FIELDS, SAMPLE_RATE, make_mixture  # noqa: E402
from src.data.wav_utils import read_wav, resample_linear, write_wav  # noqa: E402


def load(path: Path):
    signal, rate = read_wav(path)
    return resample_linear(signal, rate, SAMPLE_RATE)


def fleurs_pairs(metadata: Path, pairs: int) -> list[tuple[Path, str, Path, str]]:
    if not metadata.exists():
        raise SystemExit(f"Không thấy {metadata}. Chạy trước: python scripts/download_fleurs_samples.py --count 6")
    with metadata.open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    base = metadata.parent
    available = [r for r in rows if (base / r["audio_path"]).exists()]
    if len(available) < 2:
        raise SystemExit(
            f"Cần ít nhất 2 file audio trong {base}. Chạy trước: python scripts/download_fleurs_samples.py --count 6"
        )
    result = []
    for i in range(0, min(pairs * 2, len(available) - 1), 2):
        a, b = available[i], available[i + 1]
        result.append((base / a["audio_path"], a["transcription"], base / b["audio_path"], b["transcription"]))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metadata", type=Path, default=PROJECT_ROOT / "data/samples/fleurs_vi/metadata.csv")
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--s1", type=Path, help="file giọng chính (sẽ to hơn)")
    parser.add_argument("--s2", type=Path, help="file giọng nền")
    parser.add_argument("--s1-text", default="")
    parser.add_argument("--s2-text", default="")
    parser.add_argument("--gain-db", type=float, default=3.0, help="s1 to hơn s2 bao nhiêu dB")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/processed/separation_demo")
    args = parser.parse_args()

    if args.s1 and args.s2:
        pairs = [(args.s1, args.s1_text, args.s2, args.s2_text)]
    elif args.s1 or args.s2:
        raise SystemExit("Cần cả --s1 và --s2")
    else:
        pairs = fleurs_pairs(args.metadata, args.pairs)

    rng = random.Random(0)
    rows = []
    for index, (path1, text1, path2, text2) in enumerate(pairs):
        mix, s1, s2 = make_mixture(load(path1), load(path2), args.gain_db, mode="max", segment_samples=None, rng=rng)
        mix_id = f"demo_{index:03d}"
        for folder, signal in (("mix", mix), ("s1", s1), ("s2", s2)):
            write_wav(args.output_dir / "test" / folder / f"{mix_id}.wav", signal, SAMPLE_RATE)
        rows.append({
            "id": mix_id,
            "mix": f"test/mix/{mix_id}.wav",
            "s1": f"test/s1/{mix_id}.wav",
            "s2": f"test/s2/{mix_id}.wav",
            "duration": f"{len(mix) / SAMPLE_RATE:.3f}",
            "gain_db": f"{args.gain_db:.2f}",
            "s1_speaker": path1.stem,
            "s2_speaker": path2.stem,
            "s1_text": text1,
            "s2_text": text2,
        })
        print(f"{mix_id}: {path1.name} (to) + {path2.name} -> {args.output_dir / 'test/mix' / (mix_id + '.wav')}")

    csv_path = args.output_dir / "test.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Đã tạo {len(rows)} mẫu -> {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
