"""Run an ASR model on one WAV file."""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from src.asr import create_asr
from src.audio import load_wav


def configure_console() -> None:
    """Allow the Windows terminal to print Vietnamese text."""

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe one PCM WAV file with an ASR model.",
    )

    parser.add_argument(
        "--audio",
        type=Path,
        required=True,
        help="Path to the input PCM WAV file.",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/asr.yaml"),
        help="Path to the ASR YAML configuration.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show the complete traceback when an error occurs.",
    )

    return parser.parse_args()


def load_yaml_config(path: Path) -> Mapping[str, Any]:
    """Read and validate the ASR YAML configuration."""

    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Config path is not a file: {path}"
        )

    with path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if config is None:
        raise ValueError(
            f"Config file is empty: {path}"
        )

    if not isinstance(config, Mapping):
        raise ValueError(
            "Config content must be a YAML mapping."
        )

    return extract_asr_config(config)


def extract_asr_config(
    config: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Support both asr.yaml and pipeline.yaml configuration structures."""

    # Structure of configs/asr.yaml:
    #
    # name: phowhisper
    # params: ...

    if "name" in config:
        return config

    # Optional structure:
    #
    # asr:
    #   name: phowhisper
    #   params: ...

    direct_asr_config = config.get("asr")

    if isinstance(direct_asr_config, Mapping):
        return direct_asr_config

    # Structure of configs/pipeline.yaml:
    #
    # components:
    #   asr:
    #     name: phowhisper
    #     params: ...

    components = config.get("components")

    if isinstance(components, Mapping):
        pipeline_asr_config = components.get("asr")

        if isinstance(pipeline_asr_config, Mapping):
            return pipeline_asr_config

    raise ValueError(
        "ASR configuration not found. "
        "Expected 'name', 'asr', or 'components.asr'."
    )


def validate_audio_path(path: Path) -> None:
    """Validate the input path before loading audio."""

    if not path.exists():
        raise FileNotFoundError(
            f"Audio file not found: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Audio path is not a file: {path}"
        )

    if path.suffix.lower() != ".wav":
        raise ValueError(
            "Loader currently only supports WAV files. "
            f"Received: {path.name}"
        )


def run_inference(
    audio_path: Path,
    config_path: Path,
) -> str:
    """Create the ASR component and transcribe one audio file."""

    validate_audio_path(audio_path)

    print(f"Config: {config_path.resolve()}")
    print(f"Audio:  {audio_path.resolve()}")
    print()

    asr_config = load_yaml_config(config_path)

    model_name = asr_config.get("name", "mock")
    print(f"ASR component: {model_name}")

    print("Initializing ASR...")
    model_start_time = time.perf_counter()

    asr = create_asr(asr_config)

    model_elapsed_time = time.perf_counter() - model_start_time

    print(f"ASR initialization completed: {model_elapsed_time:.2f}s")
    print()

    print("Loading audio...")
    audio = load_wav(audio_path)

    print(f"Sample rate: {audio.sample_rate} Hz")
    print(f"Channels:    {audio.channels}")
    print(f"Frames:      {audio.frame_count}")
    print(f"Duration:    {audio.duration_seconds:.2f}s")
    print()

    print("Transcribing speech...")
    inference_start_time = time.perf_counter()

    transcript = asr.transcribe(audio)

    inference_elapsed_time = (
        time.perf_counter() - inference_start_time
    )

    print()
    print("========== RESULT ==========")

    if transcript:
        print(transcript)
    else:
        print("[No speech recognized]")

    print("============================")
    print(f"Inference time: {inference_elapsed_time:.2f}s")

    return transcript


def main() -> int:
    configure_console()
    args = parse_args()

    try:
        run_inference(
            audio_path=args.audio,
            config_path=args.config,
        )
    except KeyboardInterrupt:
        print(
            "\nCancelled by user.",
            file=sys.stderr,
        )
        return 130
    except Exception as error:
        if args.debug:
            raise

        print(
            f"\nError {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        print(
            "Rerun with --debug to see the full traceback.",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
