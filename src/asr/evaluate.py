"""Evaluate an ASR model using audio files listed in metadata.csv."""

from __future__ import annotations

import argparse
import csv
import sys
import time
import unicodedata
from collections.abc import Sequence
from pathlib import Path

from src.asr.factory import create_asr
from src.asr.infer import configure_console, load_yaml_config
from src.audio.loader import load_wav


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an ASR model with a CSV metadata file.",
    )

    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/samples/fleurs_vi/metadata.csv"),
        help="CSV containing audio_path and transcription columns.",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/asr.yaml"),
        help="ASR YAML configuration.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/metrics/asr_evaluation.csv"),
        help="Where evaluation results will be saved.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of samples. Use 0 to evaluate all samples.",
    )

    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately when one sample fails.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show the complete traceback when an error occurs.",
    )

    return parser.parse_args()


def normalize_text(text: str) -> str:
    """Normalize text before calculating WER and CER."""

    text = unicodedata.normalize("NFC", text)
    text = text.lower().strip()

    # Remove punctuation marks while keeping Vietnamese diacritics.
    text = "".join(
        character
        for character in text
        if not unicodedata.category(character).startswith("P")
    )

    # Collapse multiple whitespaces into a single space.
    return " ".join(text.split())


def edit_distance(
    reference: Sequence[str],
    prediction: Sequence[str],
) -> int:
    """Calculate Levenshtein edit distance."""

    previous_row = list(range(len(prediction) + 1))

    for reference_index, reference_item in enumerate(
        reference,
        start=1,
    ):
        current_row = [reference_index]

        for prediction_index, prediction_item in enumerate(
            prediction,
            start=1,
        ):
            deletion_cost = previous_row[prediction_index] + 1
            insertion_cost = current_row[prediction_index - 1] + 1

            substitution_cost = (
                previous_row[prediction_index - 1]
                + (reference_item != prediction_item)
            )

            current_row.append(
                min(
                    deletion_cost,
                    insertion_cost,
                    substitution_cost,
                )
            )

        previous_row = current_row

    return previous_row[-1]


def calculate_word_errors(
    reference: str,
    prediction: str,
) -> tuple[int, int]:
    """Return word edit count and reference word count."""

    reference_words = normalize_text(reference).split()
    prediction_words = normalize_text(prediction).split()

    errors = edit_distance(
        reference_words,
        prediction_words,
    )

    return errors, len(reference_words)


def calculate_character_errors(
    reference: str,
    prediction: str,
) -> tuple[int, int]:
    """Return character edit count and reference character count."""

    # Remove whitespace when calculating CER.
    reference_characters = list(
        normalize_text(reference).replace(" ", "")
    )
    prediction_characters = list(
        normalize_text(prediction).replace(" ", "")
    )

    errors = edit_distance(
        reference_characters,
        prediction_characters,
    )

    return errors, len(reference_characters)


def calculate_error_rate(
    errors: int,
    reference_length: int,
) -> float:
    """Convert edit count into an error rate."""

    if reference_length == 0:
        return 0.0 if errors == 0 else 1.0

    return errors / reference_length


def load_metadata(
    metadata_path: Path,
    limit: int,
) -> list[dict[str, str]]:
    """Load and validate the metadata CSV file."""

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_path}"
        )

    if not metadata_path.is_file():
        raise ValueError(
            f"Metadata path is not a file: {metadata_path}"
        )

    if limit < 0:
        raise ValueError("--limit cannot be less than 0")

    with metadata_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as metadata_file:
        reader = csv.DictReader(metadata_file)

        required_columns = {
            "audio_path",
            "transcription",
        }

        available_columns = set(reader.fieldnames or [])
        missing_columns = required_columns - available_columns

        if missing_columns:
            missing_text = ", ".join(sorted(missing_columns))
            raise ValueError(
                f"Metadata missing required columns: {missing_text}"
            )

        rows = list(reader)

    if limit > 0:
        rows = rows[:limit]

    if not rows:
        raise ValueError("Metadata contains no samples")

    return rows


def resolve_audio_path(
    metadata_path: Path,
    audio_path_text: str,
) -> Path:
    """Resolve audio paths relative to metadata.csv."""

    audio_path = Path(audio_path_text)

    if not audio_path.is_absolute():
        audio_path = metadata_path.parent / audio_path

    return audio_path.resolve()


def save_results(
    results: list[dict[str, object]],
    output_path: Path,
) -> None:
    """Save per-sample evaluation results."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "index",
        "audio_path",
        "reference",
        "prediction",
        "normalized_reference",
        "normalized_prediction",
        "wer",
        "cer",
        "duration_seconds",
        "inference_seconds",
        "error",
    ]

    with output_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)


def evaluate(
    metadata_path: Path,
    config_path: Path,
    output_path: Path,
    limit: int,
    fail_fast: bool,
    debug: bool,
) -> None:
    """Evaluate one ASR model on all selected samples."""

    metadata_rows = load_metadata(
        metadata_path=metadata_path,
        limit=limit,
    )

    asr_config = load_yaml_config(config_path)

    print(f"Config:   {config_path.resolve()}")
    print(f"Metadata: {metadata_path.resolve()}")
    print(f"Samples:  {len(metadata_rows)}")
    print()

    print("Initializing ASR...")
    model_start_time = time.perf_counter()

    asr = create_asr(asr_config)

    model_elapsed_time = (
        time.perf_counter() - model_start_time
    )

    print(
        f"Model initialization completed: {model_elapsed_time:.2f}s"
    )
    print()

    results: list[dict[str, object]] = []

    total_word_errors = 0
    total_reference_words = 0

    total_character_errors = 0
    total_reference_characters = 0

    total_audio_duration = 0.0
    total_inference_time = 0.0

    successful_samples = 0
    failed_samples = 0

    for index, metadata_row in enumerate(
        metadata_rows,
        start=1,
    ):
        audio_path = resolve_audio_path(
            metadata_path,
            metadata_row["audio_path"],
        )

        reference = metadata_row["transcription"].strip()

        print(
            f"[{index}/{len(metadata_rows)}] "
            f"{audio_path.name}"
        )

        try:
            if not audio_path.exists():
                raise FileNotFoundError(
                    f"Audio file not found: {audio_path}"
                )

            audio = load_wav(audio_path)

            inference_start_time = time.perf_counter()
            prediction = asr.transcribe(audio)
            inference_time = (
                time.perf_counter() - inference_start_time
            )

            word_errors, reference_word_count = (
                calculate_word_errors(
                    reference,
                    prediction,
                )
            )

            character_errors, reference_character_count = (
                calculate_character_errors(
                    reference,
                    prediction,
                )
            )

            wer = calculate_error_rate(
                word_errors,
                reference_word_count,
            )

            cer = calculate_error_rate(
                character_errors,
                reference_character_count,
            )

            total_word_errors += word_errors
            total_reference_words += reference_word_count

            total_character_errors += character_errors
            total_reference_characters += (
                reference_character_count
            )

            total_audio_duration += audio.duration_seconds
            total_inference_time += inference_time
            successful_samples += 1

            results.append(
                {
                    "index": index,
                    "audio_path": str(audio_path),
                    "reference": reference,
                    "prediction": prediction,
                    "normalized_reference": normalize_text(
                        reference
                    ),
                    "normalized_prediction": normalize_text(
                        prediction
                    ),
                    "wer": f"{wer:.6f}",
                    "cer": f"{cer:.6f}",
                    "duration_seconds": (
                        f"{audio.duration_seconds:.3f}"
                    ),
                    "inference_seconds": (
                        f"{inference_time:.3f}"
                    ),
                    "error": "",
                }
            )

            print(f"  Reference:  {reference}")
            print(f"  Prediction: {prediction}")
            print(
                f"  WER: {wer * 100:.2f}% | "
                f"CER: {cer * 100:.2f}% | "
                f"Time: {inference_time:.2f}s"
            )

        except Exception as error:
            failed_samples += 1

            results.append(
                {
                    "index": index,
                    "audio_path": str(audio_path),
                    "reference": reference,
                    "prediction": "",
                    "normalized_reference": normalize_text(
                        reference
                    ),
                    "normalized_prediction": "",
                    "wer": "",
                    "cer": "",
                    "duration_seconds": "",
                    "inference_seconds": "",
                    "error": (
                        f"{type(error).__name__}: {error}"
                    ),
                }
            )

            print(
                f"  Error {type(error).__name__}: {error}",
                file=sys.stderr,
            )

            if debug or fail_fast:
                raise

        print()

    save_results(
        results=results,
        output_path=output_path,
    )

    corpus_wer = calculate_error_rate(
        total_word_errors,
        total_reference_words,
    )

    corpus_cer = calculate_error_rate(
        total_character_errors,
        total_reference_characters,
    )

    if total_audio_duration > 0:
        real_time_factor = (
            total_inference_time / total_audio_duration
        )
    else:
        real_time_factor = 0.0

    print("============ SUMMARY ============")
    print(f"Successful: {successful_samples}")
    print(f"Failed:     {failed_samples}")
    print(f"WER:        {corpus_wer * 100:.2f}%")
    print(f"CER:        {corpus_cer * 100:.2f}%")
    print(
        f"Audio:      {total_audio_duration:.2f}s"
    )
    print(
        f"Inference:  {total_inference_time:.2f}s"
    )
    print(f"RTF:        {real_time_factor:.3f}")
    print(f"Output:     {output_path.resolve()}")
    print("==================================")


def main() -> int:
    configure_console()
    args = parse_args()

    try:
        evaluate(
            metadata_path=args.metadata,
            config_path=args.config,
            output_path=args.output,
            limit=args.limit,
            fail_fast=args.fail_fast,
            debug=args.debug,
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



# python -m src.asr.evaluate --limit 5