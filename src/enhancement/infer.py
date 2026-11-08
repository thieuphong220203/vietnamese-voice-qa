"""Run DTLN speech enhancement on one WAV file.

Example::

    python -m src.enhancement.infer --audio data/samples/dtln/noisy_test.wav
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import yaml

from src.audio.loader import load_wav, save_wav
from src.enhancement.factory import create_enhancer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio", type=Path, required=True, help="Input WAV file")
    parser.add_argument("--config", type=Path, default=Path("configs/enhancement.yaml"))
    parser.add_argument("--checkpoint", type=Path, help="Override the fine-tuned .weights.h5 checkpoint")
    parser.add_argument("--device", choices=("auto", "cpu", "gpu"), help="Override the device in YAML")
    parser.add_argument("--output", type=Path, default=Path("outputs/audio/enhancement/enhanced.wav"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    params = dict(config.get("params", {}))
    if args.checkpoint:
        params["checkpoint"] = str(args.checkpoint)
    if args.device:
        params["device"] = args.device

    enhancer = create_enhancer({"name": config.get("name", "dtln"), "params": params})
    audio = load_wav(args.audio)
    started = time.perf_counter()
    enhanced = enhancer.enhance(audio)
    elapsed = time.perf_counter() - started
    output = save_wav(enhanced, args.output)

    print(f"Input: {args.audio} ({audio.duration_seconds:.2f}s, {audio.sample_rate} Hz)")
    print(f"Enhanced in {elapsed:.2f}s (RTF={elapsed / max(audio.duration_seconds, 1e-9):.3f})")
    print(f"Output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
