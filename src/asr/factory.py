from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from .base import SpeechRecognizer
from .mock import MockASR
from .phowhisper import PhoWhisperRecognizer
from .whisper import WhisperRecognizer

ASR_REGISTRY: dict[str, Callable[..., SpeechRecognizer]] = {
    "mock": MockASR,
    "phowhisper": PhoWhisperRecognizer,
    "whisper": WhisperRecognizer,
}


def create_asr(config: Mapping[str, Any]) -> SpeechRecognizer:
    name = str(config.get("name", "mock")).lower()
    params = dict(config.get("params", {}))
    try:
        constructor = ASR_REGISTRY[name]
    except KeyError as error:
        raise ValueError(f"unknown ASR component: {name!r}") from error
    return constructor(**params)
