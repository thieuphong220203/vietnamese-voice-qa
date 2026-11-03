"""Đánh giá tách giọng trên tập trộn do ``src.data.prepare_separation`` tạo ra.

Chỉ số:
- SI-SNR / SI-SNRi (dB, PIT) — chất lượng tách.
- primary_correct — quy tắc "giọng to nhất" có chọn đúng s1 (người hỏi) không.
- (``--asr``) WER/CER của PhoWhisper trên s1_text khi đưa vào:
    mix      : audio trộn, không tách (baseline "không có separation")
    primary  : giọng chính sau khi tách (đúng như pipeline)
    clean    : s1 sạch gốc (cận trên)

Ví dụ::

    # Baseline (model tiếng Anh gốc)
    python -m src.separation.evaluate --csv data/processed/separation/test.csv --checkpoint none --tag baseline --asr
    # Sau fine-tune
    python -m src.separation.evaluate --csv data/processed/separation/test.csv \\
        --checkpoint checkpoints/separation/finetuned/best.pt --tag finetuned --asr
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.audio.types import AudioData
from src.data.wav_utils import read_wav, write_wav
from src.separation.factory import create_separator
from src.separation.metrics import si_snr_improvement
from src.separation.selection import primary_index


def load_config(path: Path, checkpoint: str | None) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    params = dict(config.get("params", {}))
    if config.get("name", "sepformer") == "sepformer":
        params["output"] = "all"  # cần mọi nguồn để tính PIT
    if checkpoint and checkpoint.lower() == "none":
        params.pop("checkpoint", None)  # ép dùng model gốc (baseline)
    elif checkpoint:
        params["checkpoint"] = checkpoint
    return {"name": config.get("name", "sepformer"), "params": params}


def separate(separator: Any, mixture: np.ndarray, sample_rate: int) -> list[np.ndarray]:
    if hasattr(separator, "separate_array"):
        return separator.separate_array(mixture, sample_rate)
    sources = separator.separate(AudioData.from_iterable(mixture.tolist(), sample_rate))
    return [np.asarray(source.samples, dtype=np.float32) for source in sources]


def to_audio(signal: np.ndarray, sample_rate: int) -> AudioData:
    peak = float(np.max(np.abs(signal))) if signal.size else 0.0
    if peak > 0:
        signal = signal * (0.9 / peak)
    return AudioData.from_iterable(signal.astype(np.float32).tolist(), sample_rate)


def load_rows(csv_path: Path, limit: int | None) -> list[Mapping[str, str]]:
    with csv_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows[:limit] if limit else rows


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_rows(args.csv, args.limit)
    base_dir = args.csv.parent
    separator = create_separator(load_config(args.config, args.checkpoint))

    asr = None
    if args.asr:
        from src.asr.evaluate import calculate_character_errors, calculate_word_errors
        from src.asr.factory import create_asr

        asr_config = yaml.safe_load(args.asr_config.read_text(encoding="utf-8")) or {}
        asr = create_asr(asr_config)

    audio_dir = args.output.parent.parent / "audio" / f"separation_{args.tag}"
    results = []
    totals: dict[str, list[int]] = {}
    started = time.perf_counter()

    for index, row in enumerate(rows):
        mixture, rate = read_wav(base_dir / row["mix"])
        s1, _ = read_wav(base_dir / row["s1"])
        s2, _ = read_wav(base_dir / row["s2"])

        t0 = time.perf_counter()
        estimates = separate(separator, mixture, rate)
        elapsed = time.perf_counter() - t0

        scores = si_snr_improvement(estimates, [s1, s2], mixture)
        perm = scores["perm"]
        chosen = primary_index(estimates)
        result: dict[str, Any] = {
            "id": row["id"],
            "duration": len(mixture) / rate,
            "gain_db": float(row.get("gain_db", 0) or 0),
            "si_snr": scores["si_snr"],
            "si_snr_mix": scores["si_snr_mix"],
            "si_snri": scores["si_snri"],
            "primary_correct": int(chosen == perm[0]),
            "rtf": elapsed / (len(mixture) / rate),
        }

        if index < args.save_audio:
            for i, estimate in enumerate(estimates):
                write_wav(audio_dir / f"{row['id']}_est{i + 1}.wav", to_audio(estimate, rate).samples, rate)

        if asr is not None:
            reference = row.get("s1_text", "")
            for name, signal in (("mix", mixture), ("primary", estimates[chosen]), ("clean", s1)):
                hypothesis = asr.transcribe(to_audio(np.asarray(signal), rate))
                word_err, word_len = calculate_word_errors(reference, hypothesis)
                char_err, char_len = calculate_character_errors(reference, hypothesis)
                for key, value in ((f"{name}_werr", word_err), (f"{name}_wlen", word_len),
                                   (f"{name}_cerr", char_err), (f"{name}_clen", char_len)):
                    totals.setdefault(key, []).append(value)
                result[f"{name}_text"] = hypothesis
                result[f"{name}_wer"] = word_err / word_len if word_len else 0.0
            result["reference"] = reference

        results.append(result)
        print(
            f"[{index + 1}/{len(rows)}] {row['id']} SI-SNRi={result['si_snri']:.2f} dB "
            f"primary_ok={result['primary_correct']}"
            + (f" WER mix/primary={result['mix_wer']:.2f}/{result['primary_wer']:.2f}" if asr else "")
        )

    summary: dict[str, Any] = {
        "tag": args.tag,
        "csv": str(args.csv),
        "checkpoint": getattr(separator, "checkpoint", None),
        "samples": len(results),
        "si_snr": float(np.mean([r["si_snr"] for r in results])) if results else None,
        "si_snr_mix": float(np.mean([r["si_snr_mix"] for r in results])) if results else None,
        "si_snri": float(np.mean([r["si_snri"] for r in results])) if results else None,
        "primary_accuracy": float(np.mean([r["primary_correct"] for r in results])) if results else None,
        "rtf": float(np.mean([r["rtf"] for r in results])) if results else None,
        "elapsed_seconds": time.perf_counter() - started,
    }
    if asr is not None:
        for name in ("mix", "primary", "clean"):
            summary[f"wer_{name}"] = sum(totals[f"{name}_werr"]) / max(1, sum(totals[f"{name}_wlen"]))
            summary[f"cer_{name}"] = sum(totals[f"{name}_cerr"]) / max(1, sum(totals[f"{name}_clen"]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if results:
        fields = list(dict.fromkeys(key for result in results for key in result))
        with args.output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(results)
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n===== Tổng kết =====")
    for key, value in summary.items():
        print(f"{key:>18}: {value:.4f}" if isinstance(value, float) else f"{key:>18}: {value}")
    print(f"Chi tiết: {args.output}\nTổng kết: {summary_path}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, default=Path("data/processed/separation/test.csv"))
    parser.add_argument("--config", type=Path, default=Path("configs/separation.yaml"))
    parser.add_argument("--checkpoint", default=None, help="file .pt fine-tune; 'none' = model gốc; bỏ trống = theo config")
    parser.add_argument("--tag", default="baseline", help="tên lần chạy, dùng đặt tên file kết quả")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--asr", action="store_true", help="tính thêm WER/CER bằng PhoWhisper")
    parser.add_argument("--asr-config", type=Path, default=Path("configs/asr.yaml"))
    parser.add_argument("--save-audio", type=int, default=5, help="lưu audio tách của N mẫu đầu để nghe thử")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.output is None:
        args.output = Path("outputs/metrics") / f"separation_{args.tag}.csv"
    return args


def main() -> int:
    evaluate(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
