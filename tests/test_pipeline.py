from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.audio.types import AudioData
from src.pipeline.pipeline import VoiceQAPipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_yaml_to_factory_to_pipeline_to_output(tmp_path: Path) -> None:
    # Exercise the wiring with mocks without downloading the real models
    # selected in the application's pipeline.yaml.
    config = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "pipeline.yaml").read_text(encoding="utf-8")
    )
    config["components"]["asr"] = {
        "name": "mock",
        "params": {"transcript": "Xin chào, đây là câu hỏi thử nghiệm."},
    }
    config["components"]["qa"] = {
        "type": "mock",
        "answer_text": "Đây là câu trả lời thử nghiệm.",
    }
    config["components"]["tts"] = {"name": "mock", "params": {}}
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    pipeline = VoiceQAPipeline.from_yaml(config_path)

    result = pipeline.run(AudioData.silence(0.25, sample_rate=16_000))

    assert result.transcript == "Xin chào, đây là câu hỏi thử nghiệm."
    assert result.answer == "Đây là câu trả lời thử nghiệm."
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
            "qa": {"type": "mock", "answer_text": "Câu trả lời thử nghiệm"},
            "tts": {"name": "mock", "params": {}},
        },
        "pipeline": {"transcript_separator": " | "},
    }

    result = VoiceQAPipeline.from_config(config).run(AudioData.silence(0.1))

    assert len(result.separated_sources) == 2
    assert result.transcript == "một nguồn | một nguồn"
    assert result.answer == "Câu trả lời thử nghiệm"


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
