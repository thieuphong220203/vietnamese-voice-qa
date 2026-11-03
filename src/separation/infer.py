"""Tách giọng một file audio và lưu từng giọng ra file riêng.

Ví dụ::

    python -m src.separation.infer --audio path/to/mix.wav
    python -m src.separation.infer --audio mix.wav --checkpoint checkpoints/separation/finetuned/best.pt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import yaml

from src.audio.loader import load_wav, save_wav
from src.separation.factory import create_separator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/separation.yaml"))
    parser.add_argument("--checkpoint", default=None, help="file .pt; 'none' = model gốc; bỏ trống = theo config")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/audio/separation"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    params = dict(config.get("params", {}))
    if config.get("name", "sepformer") == "sepformer":
        params["output"] = "all"
    if args.checkpoint and args.checkpoint.lower() == "none":
        params.pop("checkpoint", None)
    elif args.checkpoint:
        params["checkpoint"] = args.checkpoint
    separator = create_separator({"name": config.get("name", "sepformer"), "params": params})

    audio = load_wav(args.audio)
    started = time.perf_counter()
    sources = separator.separate(audio)
    elapsed = time.perf_counter() - started

    print(f"Input: {args.audio} ({audio.duration_seconds:.2f}s, {audio.sample_rate} Hz)")
    print(f"Tách thành {len(sources)} giọng trong {elapsed:.2f}s (RTF={elapsed / max(audio.duration_seconds, 1e-9):.3f})")
    for rank, source in enumerate(sources, start=1):
        path = save_wav(source, args.output_dir / f"{args.audio.stem}_source{rank}.wav")
        label = " (giọng chính → ASR)" if rank == 1 else ""
        print(f"  #{rank}: {path}{label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
