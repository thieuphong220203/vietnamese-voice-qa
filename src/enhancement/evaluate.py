"""Compare noisy, pretrained DTLN, and fine-tuned DTLN on paired WAV files.

The CSV manifest must contain ``noisy`` and ``clean`` columns. Paths may be
absolute or relative to the CSV file.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.data.wav_utils import read_wav, resample_linear
from src.enhancement.dtln import DTLNEnhancer


def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    length = min(len(reference), len(estimate))
    reference = np.asarray(reference[:length], dtype=np.float64)
    estimate = np.asarray(estimate[:length], dtype=np.float64)
    reference -= reference.mean()
    estimate -= estimate.mean()
    scale = np.dot(estimate, reference) / (np.dot(reference, reference) + 1e-12)
    target = scale * reference
    noise = estimate - target
    return float(10 * np.log10((np.dot(target, target) + 1e-12) / (np.dot(noise, noise) + 1e-12)))


def _resolve_audio(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _enhanced_samples(enhancer: DTLNEnhancer, samples: np.ndarray, sample_rate: int) -> np.ndarray:
    from src.audio.types import AudioData

    audio = AudioData.from_iterable(samples, sample_rate=sample_rate)
    return np.asarray(enhancer.enhance(audio).samples, dtype=np.float32)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    try:
        from pystoi import stoi
    except ImportError as error:
        raise RuntimeError("STOI dependency missing. Install it with: python -m pip install pystoi") from error

    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    params = dict(config.get("params", {}))
    fine_checkpoint = args.checkpoint or params.get("checkpoint")
    baseline_checkpoint = args.baseline_checkpoint or params.get(
        "baseline_checkpoint", "checkpoints/enhancement/dtln/DTLN_original_model.h5"
    )
    source_dir = params.get("source_dir", "checkpoints/enhancement/DTLN")
    device = args.device or params.get("device", "auto")
    sample_rate = int(params.get("sample_rate", 16_000))
    if not fine_checkpoint:
        raise ValueError("Set params.checkpoint in the config or pass --checkpoint")

    rows: list[dict[str, Any]] = []
    with args.csv.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"noisy", "clean"}.issubset(reader.fieldnames or ()):
            raise ValueError("Evaluation manifest needs CSV columns named 'noisy' and 'clean'")
        manifest = list(reader)
    if args.limit is not None:
        manifest = manifest[: args.limit]
    if not manifest:
        raise ValueError(f"No audio pairs found in {args.csv}")

    baseline = DTLNEnhancer(
        checkpoint=baseline_checkpoint, source_dir=source_dir, sample_rate=sample_rate, device=device
    )
    finetuned = DTLNEnhancer(
        checkpoint=fine_checkpoint, source_dir=source_dir, sample_rate=sample_rate, device=device
    )
    base_dir = args.csv.parent

    for index, item in enumerate(manifest, start=1):
        noisy_path = _resolve_audio(base_dir, item["noisy"])
        clean_path = _resolve_audio(base_dir, item["clean"])
        noisy, noisy_rate = read_wav(noisy_path)
        clean, clean_rate = read_wav(clean_path)
        if noisy_rate != clean_rate:
            clean = resample_linear(clean, clean_rate, noisy_rate)
        if noisy_rate != sample_rate:
            noisy = resample_linear(noisy, noisy_rate, sample_rate)
            clean = resample_linear(clean, noisy_rate, sample_rate)
            metric_rate = sample_rate
        else:
            metric_rate = noisy_rate

        outputs = {
            "noisy": noisy,
            "pretrained": _enhanced_samples(baseline, noisy, metric_rate),
            "finetuned": _enhanced_samples(finetuned, noisy, metric_rate),
        }
        sample_id = item.get("id") or noisy_path.stem
        for system, estimate in outputs.items():
            length = min(len(clean), len(estimate))
            rows.append(
                {
                    "id": sample_id,
                    "system": system,
                    "si_sdr_db": si_sdr(clean[:length], estimate[:length]),
                    "stoi": float(stoi(clean[:length], estimate[:length], metric_rate, extended=False)),
                }
            )
        print(f"[{index}/{len(manifest)}] {sample_id}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("id", "system", "si_sdr_db", "stoi"))
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[str, Any] = {
        "manifest": str(args.csv),
        "samples": len(manifest),
        "checkpoint_pretrained": str(baseline_checkpoint),
        "checkpoint_finetuned": str(fine_checkpoint),
        "metrics": {},
    }
    for system in ("noisy", "pretrained", "finetuned"):
        subset = [row for row in rows if row["system"] == system]
        summary["metrics"][system] = {
            "si_sdr_db": float(np.mean([row["si_sdr_db"] for row in subset])),
            "stoi": float(np.mean([row["stoi"] for row in subset])),
        }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["metrics"], ensure_ascii=False, indent=2))
    print(f"Details: {args.output}\nSummary: {summary_path}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True, help="CSV manifest with noisy and clean columns")
    parser.add_argument("--config", type=Path, default=Path("configs/enhancement.yaml"))
    parser.add_argument("--checkpoint", help="Fine-tuned .weights.h5 checkpoint override")
    parser.add_argument("--baseline-checkpoint", help="Original pretrained DTLN .h5 checkpoint override")
    parser.add_argument("--device", choices=("auto", "cpu", "gpu"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, default=Path("outputs/metrics/enhancement_evaluation.csv"))
    return parser.parse_args()


def main() -> int:
    evaluate(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
