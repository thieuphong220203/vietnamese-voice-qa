"""Launch the local Gradio application."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ui import build_app
from src.pipeline import VoiceQAPipeline


def configure_console() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the Vietnamese Voice Assistant UI.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "pipeline.yaml",
        help="Path to pipeline.yaml.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Server host. Use 0.0.0.0 to expose on the local network.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Server port.",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Ask Gradio to create a temporary public share link.",
    )
    return parser.parse_args()


def main() -> int:
    configure_console()
    args = parse_args()

    try:
        config_path = args.config.resolve()
        if not config_path.is_file():
            raise FileNotFoundError(
                f"Không tìm thấy pipeline config: {config_path}"
            )

        print(f"Đang load pipeline từ: {config_path}")
        pipeline = VoiceQAPipeline.from_yaml(config_path)
        print("Pipeline đã sẵn sàng.")

        is_colab = "google.colab" in sys.modules
        should_share = args.share or is_colab

        app = build_app(pipeline)
        app.queue().launch(
            server_name=args.host,
            server_port=args.port,
            share=should_share,
            show_error=True,
        )
    except KeyboardInterrupt:
        print("\nĐã dừng ứng dụng.")
        return 130
    except Exception as error:
        print(
            f"Lỗi {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
