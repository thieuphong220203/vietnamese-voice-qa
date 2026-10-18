from __future__ import annotations

from pathlib import Path

import pytest

from src.audio.types import AudioData
from src.pipeline.pipeline import VoiceQAPipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_yaml_to_factory_to_pipeline_to_output() -> None:
    pipeline = VoiceQAPipeline.from_yaml(PROJECT_ROOT / "configs" / "pipeline.yaml")

    result = pipeline.run(AudioData.silence(0.25, sample_rate=16_000))

    assert result.transcript == "Xin chào, đây là câu hỏi thử nghiệm."
    assert result.answer == "Đây là câu trả lời giả lập cho: Xin chào, đây là câu hỏi thử nghiệm."
    assert result.output_audio.frame_count > 0
    assert result.output_audio.metadata["mock_tts"] is True
    assert [stage.name for stage in result.trace] == [
        "vad",
        "separation",
        "enhancement",
        "asr",
        "qa",
        "tts",
    ]


def test_dependency_injection_supports_multiple_sources() -> None:
    config = {
        "components": {
            "vad": {"name": "mock", "params": {}},
            "separation": {"name": "mock", "params": {"source_count": 2}},
            "enhancement": {"name": "mock", "params": {}},
            "asr": {"name": "mock", "params": {"transcript": "một nguồn"}},
            "qa": {"name": "mock", "params": {"response_template": "Q: {question}"}},
            "tts": {"name": "mock", "params": {}},
        },
        "pipeline": {"transcript_separator": " | "},
    }

    result = VoiceQAPipeline.from_config(config).run(AudioData.silence(0.1))

    assert len(result.separated_sources) == 2
    assert result.transcript == "một nguồn | một nguồn"
    assert result.answer == "Q: một nguồn | một nguồn"


def test_unknown_component_fails_during_construction() -> None:
    config = {
        "components": {
            "vad": {"name": "does-not-exist"},
            "separation": {"name": "mock"},
            "enhancement": {"name": "mock"},
            "asr": {"name": "mock"},
            "qa": {"name": "mock"},
            "tts": {"name": "mock"},
        }
    }

    with pytest.raises(ValueError, match="unknown VAD component"):
        VoiceQAPipeline.from_config(config)
