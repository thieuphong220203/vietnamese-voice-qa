from __future__ import annotations

from typing import Any, Mapping

from .base import QuestionAnswerer
from .mock import MockQA


def create_qa(config: Mapping[str, Any]) -> QuestionAnswerer:
    """Create a question-answering component from configuration."""

    qa_type = str(config.get("type", "mock")).strip().lower()

    if qa_type == "mock":
        return MockQA(
            answer_text=str(
                config.get(
                    "answer_text",
                    "Đây là câu trả lời thử nghiệm.",
                )
            )
        )

    if qa_type == "qwen":
        # Import ở đây để chỉ load Transformers khi thực sự dùng Qwen.
        from .qwen import QwenQA

        max_new_tokens = int(
            config.get("max_new_tokens", 128)
        )

        if max_new_tokens <= 0:
            raise ValueError(
                "qa.max_new_tokens must be greater than zero"
            )

        return QwenQA(
            model_id=str(
                config.get(
                    "model_id",
                    "Qwen/Qwen2.5-1.5B-Instruct",
                )
            ),
            device=str(config.get("device", "auto")),
            max_new_tokens=max_new_tokens,
            system_prompt=str(
                config.get(
                    "system_prompt",
                    "Bạn là một trợ lý hữu ích. "
                    "Hãy trả lời bằng tiếng Việt, "
                    "rõ ràng và ngắn gọn.",
                )
            ),
        )

    raise ValueError(
        f"Unsupported QA type: {qa_type!r}. "
        "Expected 'mock' or 'qwen'."
    )