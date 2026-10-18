"""Run the phase-1 pipeline with a WAV file or generated mock input."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow `python scripts/run_pipeline.py` from the repository root without
# requiring the project to be installed as a package first.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audio.loader import save_wav
from src.audio.types import AudioData
from src.pipeline.pipeline import VoiceQAPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/pipeline.yaml"))
    parser.add_argument("--input", type=Path, help="Optional PCM WAV input")
    parser.add_argument("--output", type=Path, help="Optional output WAV path")
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    pipeline = VoiceQAPipeline.from_yaml(args.config)
    input_audio = (
        pipeline.run_file(args.input)
        if args.input
        else pipeline.run(AudioData.silence(1.0, sample_rate=16_000))
    )

    if args.output:
        save_wav(input_audio.output_audio, args.output)

    summary = {
        "transcript": input_audio.transcript,
        "answer": input_audio.answer,
        "output_duration_seconds": input_audio.output_audio.duration_seconds,
        "trace": [
            {
                "stage": stage.name,
                "inputs": stage.input_count,
                "outputs": stage.output_count,
            }
            for stage in input_audio.trace
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
