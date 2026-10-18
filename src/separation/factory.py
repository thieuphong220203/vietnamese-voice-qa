from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from .base import SpeechSeparator
from .mock import MockSeparator
from .sepformer import SepFormerSeparator

SEPARATOR_REGISTRY: dict[str, Callable[..., SpeechSeparator]] = {
    "mock": MockSeparator,
    "sepformer": SepFormerSeparator,
}


def create_separator(config: Mapping[str, Any]) -> SpeechSeparator:
    name = str(config.get("name", "mock")).lower()
    params = dict(config.get("params", {}))
    try:
        constructor = SEPARATOR_REGISTRY[name]
    except KeyError as error:
        raise ValueError(f"unknown separation component: {name!r}") from error
    return constructor(**params)
