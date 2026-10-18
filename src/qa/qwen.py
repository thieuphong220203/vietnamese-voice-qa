from __future__ import annotations

from .base import QuestionAnswerer


class QwenQuestionAnswerer(QuestionAnswerer):
    """Reserved adapter for a future Qwen integration."""

    def __init__(self, **_: object) -> None:
        raise NotImplementedError("Qwen is a phase-2 integration; use name: mock")

    def answer(self, question: str) -> str:
        raise NotImplementedError
