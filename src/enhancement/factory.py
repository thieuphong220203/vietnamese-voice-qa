from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from .base import SpeechEnhancer
from .deepfilternet import DeepFilterNetEnhancer
from .metricgan import MetricGANEnhancer
from .mock import MockEnhancer

ENHANCER_REGISTRY: dict[str, Callable[..., SpeechEnhancer]] = {
    "mock": MockEnhancer,
    "deepfilternet": DeepFilterNetEnhancer,
    "metricgan": MetricGANEnhancer,
}


def create_enhancer(config: Mapping[str, Any]) -> SpeechEnhancer:
    name = str(config.get("name", "mock")).lower()
    params = dict(config.get("params", {}))
    try:
        constructor = ENHANCER_REGISTRY[name]
    except KeyError as error:
        raise ValueError(f"unknown enhancement component: {name!r}") from error
    return constructor(**params)
