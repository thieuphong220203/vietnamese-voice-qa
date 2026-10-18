from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from .base import VoiceActivityDetector
from .mock import MockVAD
from .silero import SileroVAD

VAD_REGISTRY: dict[str, Callable[..., VoiceActivityDetector]] = {
    "mock": MockVAD,
    "silero": SileroVAD,
}


def create_vad(config: Mapping[str, Any]) -> VoiceActivityDetector:
    name = str(config.get("name", "mock")).lower()
    params = dict(config.get("params", {}))
    try:
        constructor = VAD_REGISTRY[name]
    except KeyError as error:
        raise ValueError(f"unknown VAD component: {name!r}") from error
    return constructor(**params)
