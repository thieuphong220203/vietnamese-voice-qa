from __future__ import annotations

from .base import QuestionAnswerer


class MockQA(QuestionAnswerer):
    """Format a deterministic response without calling an LLM."""

    def __init__(self, response_template: str = "Mock answer for: {question}") -> None:
        self.response_template = response_template

    def answer(self, question: str) -> str:
        return self.response_template.format(question=question)
