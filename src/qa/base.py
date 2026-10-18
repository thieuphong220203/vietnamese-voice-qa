from __future__ import annotations

from abc import ABC, abstractmethod


class QuestionAnswerer(ABC):
    @abstractmethod
    def answer(self, question: str) -> str:
        """Produce a text response for a transcribed question."""
