from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from .base import QuestionAnswerer
from .mock import MockQA
from .qwen import QwenQuestionAnswerer

QA_REGISTRY: dict[str, Callable[..., QuestionAnswerer]] = {
    "mock": MockQA,
    "qwen": QwenQuestionAnswerer,
}


def create_qa(config: Mapping[str, Any]) -> QuestionAnswerer:
    name = str(config.get("name", "mock")).lower()
    params = dict(config.get("params", {}))
    try:
        constructor = QA_REGISTRY[name]
    except KeyError as error:
        raise ValueError(f"unknown QA component: {name!r}") from error
    return constructor(**params)
