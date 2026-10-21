from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from .base import SpeechSynthesizer
from .mock import MockTTS
from .viettts import VietTTSSynthesizer

TTS_REGISTRY: dict[str, Callable[..., SpeechSynthesizer]] = {
    "mock": MockTTS,
    "viettts": VietTTSSynthesizer,
}


def create_tts(config: Mapping[str, Any]) -> SpeechSynthesizer:
    name = str(config.get("name", "mock")).lower()
    params = dict(config.get("params", {}))
    if name == "vieneu":
        # Keep the optional model dependency out of mock-only runs.
        from .vieneu import VieNeuSynthesizer

        return VieNeuSynthesizer(**params)
    try:
        constructor = TTS_REGISTRY[name]
    except KeyError as error:
        raise ValueError(f"unknown TTS component: {name!r}") from error
    return constructor(**params)
