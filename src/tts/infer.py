"""Run TTS alone and save its streamed audio to a WAV file."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import soundfile as sf
import yaml

from .factory import create_tts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Try the configured TTS model.")
    parser.add_argument("--text", required=True, help="Vietnamese text to read.")
    parser.add_argument("--config", type=Path, default=Path("configs/tts.yaml"))
    parser.add_argument("--output", type=Path, default=Path("outputs/audio/tts_sample.wav"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.text.strip():
        raise ValueError("--text cannot be empty")

    with args.config.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError("TTS config must be a YAML mapping")

    print(f"Loading TTS from {args.config}...", flush=True)
    load_started = perf_counter()
    tts = create_tts(config)
    print(f"Model load: {perf_counter() - load_started:.3f} s", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    first_audio_at: float | None = None
    frame_count = 0
    sample_rate: int | None = None
    wav: sf.SoundFile | None = None
    try:
        for chunk in tts.stream_synthesize(args.text):
            if chunk.frame_count == 0:
                continue
            if chunk.channels != 1:
                raise ValueError("TTS output must be mono")
            if sample_rate is not None and chunk.sample_rate != sample_rate:
                raise ValueError("TTS chunk format changed during generation")
            if first_audio_at is None:
                first_audio_at = perf_counter()
                sample_rate = chunk.sample_rate
                wav = sf.SoundFile(
                    args.output, mode="w", samplerate=sample_rate, channels=1
                )
            assert wav is not None
            wav.write(np.asarray(chunk.samples, dtype=np.float32))
            frame_count += chunk.frame_count
    finally:
        if wav is not None:
            wav.close()

    elapsed = perf_counter() - started
    if not frame_count or sample_rate is None:
        raise RuntimeError("TTS generated no audio")
    duration = frame_count / sample_rate
    print(f"Audio: {args.output.resolve()}")
    print(f"Duration: {duration:.3f} s")
    print(f"First audio: {first_audio_at - started:.3f} s")
    print(f"Generation: {elapsed:.3f} s")
    print(f"RTF: {elapsed / duration:.3f}")


if __name__ == "__main__":
    main()
