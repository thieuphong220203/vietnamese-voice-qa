from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator


class QuestionAnswerer(ABC):
    @abstractmethod
    def answer(self, question: str) -> str:
        pass

    def stream_answer(self, question: str) -> Iterator[str]:
        yield self.answer(question)
