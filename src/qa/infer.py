from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import yaml

from .factory import create_qa


def configure_stdout() -> None:
    """Allow Windows terminals to print Vietnamese text."""

    if sys.stdout:
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run question answering from the command line."
    )

    parser.add_argument(
        "--question",
        required=True,
        help="Question to send to the QA model.",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/qa.yaml"),
        help="Path to the QA YAML configuration.",
    )

    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Wait for the complete answer instead of streaming fragments.",
    )

    return parser.parse_args()


def load_config(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"QA config file does not exist: {path}"
        )

    with path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, Mapping):
        raise ValueError(
            "QA config must contain a YAML mapping."
        )

    return config


def run_streaming(
    qa: Any,
    question: str,
) -> None:
    started_at = perf_counter()
    first_fragment_at: float | None = None
    fragments: list[str] = []

    print()
    print("Answer:")
    print()

    for fragment in qa.stream_answer(question):
        if first_fragment_at is None:
            first_fragment_at = perf_counter()

        fragments.append(fragment)
        print(fragment, end="", flush=True)

    finished_at = perf_counter()
    answer = "".join(fragments).strip()

    print()
    print()

    if not answer:
        print("[Model returned an empty answer]")

    if first_fragment_at is not None:
        ttft = first_fragment_at - started_at
        print(f"TTFT: {ttft:.3f} seconds")

    total_time = finished_at - started_at
    print(f"Generation time: {total_time:.3f} seconds")


def run_non_streaming(
    qa: Any,
    question: str,
) -> None:
    started_at = perf_counter()

    answer = qa.answer(question)

    finished_at = perf_counter()

    print()
    print("Answer:")
    print()
    print(answer or "[Model returned an empty answer]")
    print()
    print(
        f"Generation time: "
        f"{finished_at - started_at:.3f} seconds"
    )


def main() -> None:
    configure_stdout()
    args = parse_args()

    try:
        config = load_config(args.config)

        print(f"Loading QA config: {args.config}")
        print(f"Question: {args.question}")
        print()

        load_started_at = perf_counter()
        qa = create_qa(config)
        load_finished_at = perf_counter()

        print(
            f"Model load time: "
            f"{load_finished_at - load_started_at:.3f} seconds"
        )

        if args.no_stream:
            run_non_streaming(
                qa=qa,
                question=args.question,
            )
        else:
            run_streaming(
                qa=qa,
                question=args.question,
            )

    except KeyboardInterrupt:
        print("\nStopped by user.", file=sys.stderr)
        raise SystemExit(130)

    except (
        FileNotFoundError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        yaml.YAMLError,
    ) as error:
        print(
            f"Error {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()