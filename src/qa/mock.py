from __future__ import annotations

from .base import QuestionAnswerer


class MockQA(QuestionAnswerer):
    """Return a configured answer without loading a real model."""

    def __init__(
        self,
        answer_text: str = "Đây là câu trả lời thử nghiệm.",
    ) -> None:
        self.answer_text = answer_text

    def answer(self, question: str) -> str:
        if not question.strip():
            return ""

        return self.answer_text