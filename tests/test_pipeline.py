from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

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
    config["components"]["separation"] = {"name": "mock", "params": {"source_count": 1}}
    config["components"]["enhancement"] = {"name": "mock", "params": {}}
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


@pytest.mark.parametrize("enable_separation,enable_enhancement", [
    (False, False), (False, True), (True, False), (True, True),
])
@pytest.mark.parametrize("streaming", [False, True])
def test_optional_audio_stages_apply_per_request(
    enable_separation: bool, enable_enhancement: bool, streaming: bool,
) -> None:
    config = {
        "components": {
            "vad": {"name": "mock", "params": {}},
            "separation": {"name": "mock", "params": {"source_count": 2}},
            "enhancement": {"name": "mock", "params": {}},
            "asr": {"name": "mock", "params": {}},
            "qa": {"type": "mock", "answer_text": "Trả lời."},
            "tts": {"name": "mock", "params": {}},
        }
    }
    pipeline = VoiceQAPipeline.from_config(config)
    pipeline.separator.separate = Mock(wraps=pipeline.separator.separate)
    pipeline.enhancer.enhance = Mock(wraps=pipeline.enhancer.enhance)
    seen_sources = []

    def transcribe(source: AudioData) -> str:
        seen_sources.append(source)
        return "đã nhận dạng"

    pipeline.asr.transcribe = transcribe
    audio = AudioData.silence(0.01)
    options = dict(enable_separation=enable_separation, enable_enhancement=enable_enhancement)

    if streaming:
        updates = list(pipeline.stream(audio, **options))
        assert updates[0].transcript == " ".join(["đã nhận dạng"] * len(seen_sources))
        assert updates[-1].answer == "Trả lời."
        assert any(update.audio_chunk is not None for update in updates)
    else:
        result = pipeline.run(audio, **options)
        assert result.transcript == " ".join(["đã nhận dạng"] * len(seen_sources))
        assert result.answer == "Trả lời."
        assert result.output_audio.frame_count > 0
        assert len(result.separated_sources) == len(seen_sources)
        assert len(result.enhanced_sources) == len(seen_sources)
        assert [stage.name for stage in result.trace] == [
            "vad",
            *(["separation"] if enable_separation else []),
            *(["enhancement"] if enable_enhancement else []),
            "asr", "qa", "tts",
        ]

    source_count = 2 if enable_separation else 1
    assert len(seen_sources) == source_count
    assert pipeline.separator.separate.call_count == int(enable_separation)
    assert pipeline.enhancer.enhance.call_count == source_count * int(enable_enhancement)
    for source in seen_sources:
        assert ("mock_source_index" in source.metadata) == enable_separation
        assert ("mock_enhanced" in source.metadata) == enable_enhancement


def test_toggle_does_not_change_default_behavior_on_next_request() -> None:
    config = {
        "components": {
            "vad": {"name": "mock"}, "separation": {"name": "mock"},
            "enhancement": {"name": "mock"}, "asr": {"name": "mock"},
            "qa": {"type": "mock"}, "tts": {"name": "mock"},
        }
    }
    pipeline = VoiceQAPipeline.from_config(config)
    audio = AudioData.silence(0.01)
    off = pipeline.run(audio, enable_separation=False, enable_enhancement=False)
    on = pipeline.run(audio)
    assert "mock_source_index" not in off.enhanced_sources[0].metadata
    assert "mock_enhanced" not in off.enhanced_sources[0].metadata
    assert on.enhanced_sources[0].metadata["mock_source_index"] == 0
    assert on.enhanced_sources[0].metadata["mock_enhanced"] is True
