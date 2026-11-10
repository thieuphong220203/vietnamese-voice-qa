"""Đánh giá ablation C0--C3 trên tập integrated test.

C0: mixture -> PhoWhisper
C1: mixture -> SepFormer fine-tuned -> PhoWhisper
C2: mixture -> DTLN fine-tuned -> PhoWhisper
C3: mixture -> SepFormer fine-tuned -> DTLN fine-tuned -> PhoWhisper

Thời gian tải model và đọc WAV không được tính. C1 và C3 dùng chung một lần chạy
SepFormer cho mỗi mẫu; latency của từng cấu hình được cộng từ thời gian từng stage.
Kết quả chi tiết được append sau từng lượt ASR để có thể tiếp tục bằng ``--resume``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.audio import AudioData  # noqa: E402
from src.data.wav_utils import read_wav  # noqa: E402
from src.separation.metrics import si_snr  # noqa: E402
from src.separation.selection import primary_index  # noqa: E402

CONFIGS = {
    "C0": (False, False, "Không tiền xử lý"),
    "C1": (True, False, "Chỉ tách tiếng nói"),
    "C2": (False, True, "Chỉ tăng cường tiếng nói"),
    "C3": (True, True, "Kết hợp hai mô-đun"),
}


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower().strip()
    text = "".join(
        character
        for character in text
        if not unicodedata.category(character).startswith("P")
    )
    return " ".join(text.split())


def edit_distance(reference: Sequence[str], prediction: Sequence[str]) -> int:
    previous = list(range(len(prediction) + 1))
    for reference_index, reference_item in enumerate(reference, start=1):
        current = [reference_index]
        for prediction_index, prediction_item in enumerate(prediction, start=1):
            current.append(
                min(
                    previous[prediction_index] + 1,
                    current[prediction_index - 1] + 1,
                    previous[prediction_index - 1]
                    + (reference_item != prediction_item),
                )
            )
        previous = current
    return previous[-1]


def calculate_word_errors(reference: str, prediction: str) -> tuple[int, int]:
    reference_items = normalize_text(reference).split()
    prediction_items = normalize_text(prediction).split()
    return edit_distance(reference_items, prediction_items), len(reference_items)


def calculate_character_errors(reference: str, prediction: str) -> tuple[int, int]:
    reference_items = list(normalize_text(reference).replace(" ", ""))
    prediction_items = list(normalize_text(prediction).replace(" ", ""))
    return edit_distance(reference_items, prediction_items), len(reference_items)

DETAIL_FIELDS = (
    "id",
    "config",
    "description",
    "sepformer_finetuned",
    "dtln_finetuned",
    "snr_db",
    "sir_db",
    "noise_type",
    "duration_s",
    "reference",
    "hypothesis",
    "normalized_reference",
    "normalized_hypothesis",
    "word_errors",
    "reference_words",
    "character_errors",
    "reference_characters",
    "sample_wer_percent",
    "sample_cer_percent",
    "separation_s",
    "enhancement_s",
    "preprocessing_s",
    "asr_s",
    "latency_s",
    "preprocessing_rtf",
    "selected_source_index",
    "oracle_target_index",
    "target_selection_correct",
)

SUMMARY_FIELDS = (
    "config",
    "description",
    "sepformer_finetuned",
    "dtln_finetuned",
    "samples",
    "phowhisper_small_wer_percent",
    "phowhisper_small_cer_percent",
    "preprocessing_rtf",
    "mean_latency_s",
    "mean_preprocessing_s",
    "mean_asr_s",
    "target_selection_accuracy_percent",
)

CONDITION_FIELDS = (
    "config",
    "snr_db",
    "sir_db",
    "samples",
    "phowhisper_small_wer_percent",
    "phowhisper_small_cer_percent",
    "preprocessing_rtf",
    "mean_latency_s",
    "target_selection_accuracy_percent",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=PROJECT_ROOT / "data/processed/integrated_test/metadata.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/metrics/integrated_ablation",
    )
    parser.add_argument(
        "--separation-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "checkpoints/separation/finetuned/best.pt",
    )
    parser.add_argument(
        "--enhancement-checkpoint",
        type=Path,
        default=(
            PROJECT_ROOT
            / "checkpoints/enhancement/finetuned/DTLN_vi_snr5_lr1e-4_run01.weights.h5"
        ),
    )
    parser.add_argument(
        "--dtln-source-dir",
        type=Path,
        default=PROJECT_ROOT / "checkpoints/enhancement/DTLN",
    )
    parser.add_argument("--sepformer-savedir", type=Path)
    parser.add_argument("--sepformer-model-id", default="speechbrain/sepformer-whamr16k")
    parser.add_argument("--asr-model-id", default="vinai/PhoWhisper-small")
    parser.add_argument("--torch-device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtln-device", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "float16", "float32", "bfloat16"), default="auto")
    parser.add_argument("--configs", nargs="+", choices=tuple(CONFIGS), default=tuple(CONFIGS))
    parser.add_argument("--limit", type=int, default=0, help="0 = toàn bộ metadata")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def _safe_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    root = root.resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"Đường dẫn không an toàn trong metadata: {relative}")
    return path


def load_metadata(path: Path, limit: int = 0) -> list[dict[str, str]]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy metadata: {path}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"id", "mix_path", "target_path", "transcript", "snr_db", "sir_db"}
    missing = required.difference(rows[0] if rows else {})
    if missing:
        raise ValueError(f"Metadata thiếu cột: {', '.join(sorted(missing))}")
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Metadata có id trùng nhau")
    selected = rows[:limit] if limit > 0 else rows
    if not selected:
        raise ValueError("Metadata không có mẫu để đánh giá")
    root = path.parent
    for row in selected:
        for field in ("mix_path", "target_path"):
            audio_path = _safe_path(root, row[field])
            if not audio_path.is_file():
                raise FileNotFoundError(f"{row['id']}: thiếu {field}: {audio_path}")
        if not row["transcript"].strip():
            raise ValueError(f"{row['id']}: transcript rỗng")
    return selected


def validate_inputs(args: argparse.Namespace) -> list[dict[str, str]]:
    rows = load_metadata(args.metadata, args.limit)
    needs_sep = any(CONFIGS[name][0] for name in args.configs)
    needs_dtln = any(CONFIGS[name][1] for name in args.configs)
    if needs_sep and not args.separation_checkpoint.is_file():
        raise FileNotFoundError(f"Thiếu checkpoint SepFormer: {args.separation_checkpoint}")
    if needs_dtln and not args.enhancement_checkpoint.is_file():
        raise FileNotFoundError(f"Thiếu checkpoint DTLN: {args.enhancement_checkpoint}")
    combinations = Counter((float(row["snr_db"]), float(row["sir_db"])) for row in rows)
    print(f"Đã kiểm tra {len(rows)} mẫu tại {args.metadata.resolve()}")
    print("Phân bố SNR x SIR:")
    for (snr, sir), count in sorted(combinations.items()):
        print(f"  SNR={snr:g} dB, SIR={sir:g} dB: {count} mẫu")
    if needs_sep:
        print(f"SepFormer checkpoint: {args.separation_checkpoint.resolve()}")
    if needs_dtln:
        print(f"DTLN checkpoint: {args.enhancement_checkpoint.resolve()}")
    return rows


def to_audio(samples: np.ndarray, sample_rate: int, metadata: Mapping[str, Any] | None = None) -> AudioData:
    signal = np.asarray(samples, dtype=np.float32).reshape(-1)
    return AudioData.from_iterable(signal.tolist(), sample_rate, 1, metadata)


def peak_normalize(samples: np.ndarray, target_peak: float = 0.9) -> np.ndarray:
    signal = np.asarray(samples, dtype=np.float32)
    peak = float(np.max(np.abs(signal))) if signal.size else 0.0
    return signal if peak <= 0 else (signal * (target_peak / peak)).astype(np.float32)


def synchronize_torch() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def timed_call(function: Any, *args: Any, **kwargs: Any) -> tuple[Any, float]:
    synchronize_torch()
    started = time.perf_counter()
    result = function(*args, **kwargs)
    synchronize_torch()
    return result, time.perf_counter() - started


def build_models(args: argparse.Namespace) -> tuple[Any, Any, Any]:
    # Tránh TensorFlow chiếm toàn bộ VRAM khi dùng chung GPU với PyTorch.
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

    from src.asr.factory import create_asr
    from src.enhancement.factory import create_enhancer
    from src.separation.factory import create_separator

    separator = None
    if any(CONFIGS[name][0] for name in args.configs):
        separator = create_separator(
            {
                "name": "sepformer",
                "params": {
                    "model_id": args.sepformer_model_id,
                    "device": args.torch_device,
                    "savedir": str(args.sepformer_savedir) if args.sepformer_savedir else None,
                    "checkpoint": str(args.separation_checkpoint),
                    "output": "all",
                    "normalize": False,
                },
            }
        )
        if getattr(separator, "checkpoint", None) is None:
            raise RuntimeError("SepFormer không nạp checkpoint fine-tuned")

    enhancer = None
    if any(CONFIGS[name][1] for name in args.configs):
        enhancer = create_enhancer(
            {
                "name": "dtln",
                "params": {
                    "checkpoint": str(args.enhancement_checkpoint),
                    "source_dir": str(args.dtln_source_dir),
                    "device": args.dtln_device,
                },
            }
        )

    asr = create_asr(
        {
            "name": "phowhisper",
            "params": {
                "model_id": args.asr_model_id,
                "device": args.torch_device,
                "dtype": args.dtype,
            },
        }
    )
    return separator, enhancer, asr


def warm_up(separator: Any, enhancer: Any, asr: Any, row: Mapping[str, str], root: Path) -> None:
    mixture, rate = read_wav(_safe_path(root, row["mix_path"]))
    audio = to_audio(mixture, rate)
    selected_audio = audio
    if separator is not None:
        estimates = separator.separate_array(mixture, rate)
        selected_audio = to_audio(peak_normalize(estimates[primary_index(estimates)]), rate)
    if enhancer is not None:
        enhancer.enhance(audio)
        if separator is not None:
            enhancer.enhance(selected_audio)
    asr.transcribe(audio)
    print("Warm-up hoàn tất; bắt đầu tính thời gian.")


def _append_detail(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DETAIL_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in DETAIL_FIELDS})
        handle.flush()
        os.fsync(handle.fileno())


def load_details(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def make_detail(
    *,
    metadata: Mapping[str, str],
    config: str,
    hypothesis: str,
    duration: float,
    separation_s: float,
    enhancement_s: float,
    asr_s: float,
    selected_index: int | None,
    oracle_index: int | None,
) -> dict[str, Any]:
    reference = metadata["transcript"]
    word_errors, reference_words = calculate_word_errors(reference, hypothesis)
    character_errors, reference_characters = calculate_character_errors(reference, hypothesis)
    sep_enabled, dtln_enabled, description = CONFIGS[config]
    preprocessing_s = separation_s + enhancement_s
    return {
        "id": metadata["id"],
        "config": config,
        "description": description,
        "sepformer_finetuned": "Bật" if sep_enabled else "Tắt",
        "dtln_finetuned": "Bật" if dtln_enabled else "Tắt",
        "snr_db": metadata.get("snr_db", ""),
        "sir_db": metadata.get("sir_db", metadata.get("gain_db", "")),
        "noise_type": metadata.get("noise_type", ""),
        "duration_s": f"{duration:.6f}",
        "reference": reference,
        "hypothesis": hypothesis,
        "normalized_reference": normalize_text(reference),
        "normalized_hypothesis": normalize_text(hypothesis),
        "word_errors": word_errors,
        "reference_words": reference_words,
        "character_errors": character_errors,
        "reference_characters": reference_characters,
        "sample_wer_percent": f"{100.0 * word_errors / max(1, reference_words):.6f}",
        "sample_cer_percent": f"{100.0 * character_errors / max(1, reference_characters):.6f}",
        "separation_s": f"{separation_s:.6f}",
        "enhancement_s": f"{enhancement_s:.6f}",
        "preprocessing_s": f"{preprocessing_s:.6f}",
        "asr_s": f"{asr_s:.6f}",
        "latency_s": f"{preprocessing_s + asr_s:.6f}",
        "preprocessing_rtf": f"{preprocessing_s / max(duration, 1e-12):.6f}",
        "selected_source_index": "" if selected_index is None else selected_index,
        "oracle_target_index": "" if oracle_index is None else oracle_index,
        "target_selection_correct": (
            "" if selected_index is None or oracle_index is None else int(selected_index == oracle_index)
        ),
    }


def summarize(details: Sequence[Mapping[str, str]], configs: Sequence[str]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for config in configs:
        rows = [row for row in details if row["config"] == config]
        if not rows:
            continue
        word_errors = sum(int(row["word_errors"]) for row in rows)
        reference_words = sum(int(row["reference_words"]) for row in rows)
        char_errors = sum(int(row["character_errors"]) for row in rows)
        reference_chars = sum(int(row["reference_characters"]) for row in rows)
        total_preprocessing = sum(float(row["preprocessing_s"]) for row in rows)
        total_duration = sum(float(row["duration_s"]) for row in rows)
        selection = [int(row["target_selection_correct"]) for row in rows if row["target_selection_correct"] != ""]
        sep_enabled, dtln_enabled, description = CONFIGS[config]
        summaries.append(
            {
                "config": config,
                "description": description,
                "sepformer_finetuned": "Bật" if sep_enabled else "Tắt",
                "dtln_finetuned": "Bật" if dtln_enabled else "Tắt",
                "samples": len(rows),
                "phowhisper_small_wer_percent": f"{100.0 * word_errors / max(1, reference_words):.4f}",
                "phowhisper_small_cer_percent": f"{100.0 * char_errors / max(1, reference_chars):.4f}",
                "preprocessing_rtf": f"{total_preprocessing / max(total_duration, 1e-12):.6f}",
                "mean_latency_s": f"{np.mean([float(row['latency_s']) for row in rows]):.6f}",
                "mean_preprocessing_s": f"{np.mean([float(row['preprocessing_s']) for row in rows]):.6f}",
                "mean_asr_s": f"{np.mean([float(row['asr_s']) for row in rows]):.6f}",
                "target_selection_accuracy_percent": (
                    f"{100.0 * sum(selection) / len(selection):.4f}" if selection else ""
                ),
            }
        )
    return summaries


def summarize_by_condition(
    details: Sequence[Mapping[str, str]], configs: Sequence[str]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    conditions = sorted(
        {(float(row["snr_db"]), float(row["sir_db"])) for row in details}
    )
    for config in configs:
        for snr, sir in conditions:
            rows = [
                row
                for row in details
                if row["config"] == config
                and float(row["snr_db"]) == snr
                and float(row["sir_db"]) == sir
            ]
            if not rows:
                continue
            summary = summarize(rows, [config])[0]
            output.append(
                {
                    "config": config,
                    "snr_db": f"{snr:g}",
                    "sir_db": f"{sir:g}",
                    **{field: summary[field] for field in CONDITION_FIELDS if field in summary},
                }
            )
    return output


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def hardware_info(separator: Any, enhancer: Any, asr: Any, args: argparse.Namespace) -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "metadata": str(args.metadata.resolve()),
        "sepformer_model_id": args.sepformer_model_id,
        "sepformer_checkpoint": str(args.separation_checkpoint.resolve()),
        "dtln_checkpoint": str(args.enhancement_checkpoint.resolve()),
        "phowhisper_model_id": args.asr_model_id,
        "torch_device": str(getattr(asr, "device", args.torch_device)),
        "dtln_device_requested": args.dtln_device,
        "timing": "model load and WAV I/O excluded; latency = preprocessing + ASR",
    }
    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        pass
    if enhancer is not None and getattr(enhancer, "_tf", None) is not None:
        tf = enhancer._tf
        info["tensorflow_version"] = tf.__version__
        info["tensorflow_gpus"] = [item.name for item in tf.config.list_physical_devices("GPU")]
    info["sepformer_loaded_checkpoint"] = getattr(separator, "checkpoint", None)
    info["dtln_loaded_checkpoint"] = str(getattr(enhancer, "loaded_checkpoint", "") or "")
    return info


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_spec(args: argparse.Namespace, sample_count: int) -> dict[str, Any]:
    return {
        "metadata": str(args.metadata.resolve()),
        "metadata_sha256": file_sha256(args.metadata),
        "samples": sample_count,
        "configs": list(args.configs),
        "sepformer_model_id": args.sepformer_model_id,
        "sepformer_checkpoint_sha256": (
            file_sha256(args.separation_checkpoint)
            if any(CONFIGS[name][0] for name in args.configs)
            else None
        ),
        "dtln_checkpoint_sha256": (
            file_sha256(args.enhancement_checkpoint)
            if any(CONFIGS[name][1] for name in args.configs)
            else None
        ),
        "asr_model_id": args.asr_model_id,
    }


def print_table(summaries: Sequence[Mapping[str, Any]]) -> None:
    print("\nBảng ablation C0-C3")
    header = f"{'Cấu hình':<4} {'WER (%)':>10} {'CER (%)':>10} {'RTF':>10} {'Latency (s)':>13}"
    print(header)
    print("-" * len(header))
    for row in summaries:
        print(
            f"{row['config']:<4} {row['phowhisper_small_wer_percent']:>10} "
            f"{row['phowhisper_small_cer_percent']:>10} {row['preprocessing_rtf']:>10} "
            f"{row['mean_latency_s']:>13}"
        )


def evaluate(args: argparse.Namespace) -> None:
    rows = validate_inputs(args)
    if args.validate_only:
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    details_path = args.output_dir / "integrated_ablation_details.csv"
    summary_path = args.output_dir / "integrated_ablation_summary.csv"
    condition_path = args.output_dir / "integrated_ablation_by_condition.csv"
    info_path = args.output_dir / "run_info.json"
    spec_path = args.output_dir / "run_spec.json"
    current_spec = run_spec(args, len(rows))
    if spec_path.is_file():
        previous_spec = json.loads(spec_path.read_text(encoding="utf-8"))
        if previous_spec != current_spec:
            raise RuntimeError(
                f"{spec_path} không khớp metadata/checkpoint hiện tại; "
                "hãy dùng output-dir mới cho lần đánh giá mới"
            )
    else:
        spec_path.write_text(
            json.dumps(current_spec, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    existing = load_details(details_path)
    if existing and not args.resume:
        raise FileExistsError(f"{details_path} đã tồn tại; dùng --resume hoặc đổi --output-dir")
    completed = {(row["id"], row["config"]) for row in existing}

    separator, enhancer, asr = build_models(args)
    warm_up(separator, enhancer, asr, rows[0], args.metadata.resolve().parent)
    root = args.metadata.resolve().parent

    for index, row in enumerate(rows, start=1):
        missing = [name for name in args.configs if (row["id"], name) not in completed]
        if not missing:
            print(f"[{index}/{len(rows)}] {row['id']}: đã hoàn tất, bỏ qua")
            continue

        mixture, rate = read_wav(_safe_path(root, row["mix_path"]))
        target, target_rate = read_wav(_safe_path(root, row["target_path"]))
        if rate != target_rate:
            raise ValueError(f"{row['id']}: mix và target khác sample rate")
        duration = len(mixture) / rate
        mixture_audio = to_audio(mixture, rate, {"id": row["id"]})

        separation_s = 0.0
        selected_index: int | None = None
        oracle_index: int | None = None
        selected_audio: AudioData | None = None
        if any(CONFIGS[name][0] for name in missing):
            estimates, separation_s = timed_call(separator.separate_array, mixture, rate)
            selected_index = primary_index(estimates)
            oracle_index = int(np.argmax([si_snr(estimate, target) for estimate in estimates]))
            selected_audio = to_audio(
                peak_normalize(estimates[selected_index]), rate, {"id": row["id"]}
            )

        enhanced_mix: AudioData | None = None
        enhancement_mix_s = 0.0
        if "C2" in missing:
            enhanced_mix, enhancement_mix_s = timed_call(enhancer.enhance, mixture_audio)

        enhanced_selected: AudioData | None = None
        enhancement_selected_s = 0.0
        if "C3" in missing:
            if selected_audio is None:
                raise RuntimeError("C3 cần đầu ra SepFormer")
            enhanced_selected, enhancement_selected_s = timed_call(enhancer.enhance, selected_audio)

        branch_audio = {
            "C0": mixture_audio,
            "C1": selected_audio,
            "C2": enhanced_mix,
            "C3": enhanced_selected,
        }
        enhancement_time = {"C0": 0.0, "C1": 0.0, "C2": enhancement_mix_s, "C3": enhancement_selected_s}

        for config in missing:
            audio = branch_audio[config]
            if audio is None:
                raise RuntimeError(f"Không tạo được audio cho {config}")
            hypothesis, asr_s = timed_call(asr.transcribe, audio)
            detail = make_detail(
                metadata=row,
                config=config,
                hypothesis=hypothesis,
                duration=duration,
                separation_s=separation_s if CONFIGS[config][0] else 0.0,
                enhancement_s=enhancement_time[config],
                asr_s=asr_s,
                selected_index=selected_index if CONFIGS[config][0] else None,
                oracle_index=oracle_index if CONFIGS[config][0] else None,
            )
            _append_detail(details_path, detail)
            completed.add((row["id"], config))
            print(
                f"[{index}/{len(rows)}] {row['id']} {config}: "
                f"WER={detail['sample_wer_percent']}% CER={detail['sample_cer_percent']}% "
                f"latency={detail['latency_s']}s"
            )

    details = load_details(details_path)
    expected = {(row["id"], name) for row in rows for name in args.configs}
    present = {(row["id"], row["config"]) for row in details}
    missing_results = expected.difference(present)
    if missing_results:
        raise RuntimeError(f"Còn thiếu {len(missing_results)} kết quả; chạy lại với --resume")

    selected_details = [row for row in details if (row["id"], row["config"]) in expected]
    summaries = summarize(selected_details, args.configs)
    condition_summaries = summarize_by_condition(selected_details, args.configs)
    write_csv(summary_path, summaries, SUMMARY_FIELDS)
    write_csv(condition_path, condition_summaries, CONDITION_FIELDS)
    info = hardware_info(separator, enhancer, asr, args)
    info["samples"] = len(rows)
    info["configurations"] = list(args.configs)
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print_table(summaries)
    print(f"\nChi tiết: {details_path}")
    print(f"Tổng hợp: {summary_path}")
    print(f"Theo SNR x SIR: {condition_path}")
    print(f"Thông tin lần chạy: {info_path}")


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    evaluate(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
