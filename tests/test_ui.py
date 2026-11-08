"""Check the Gradio controls and callbacks without loading real models."""

import numpy as np
import pytest

from app.ui import build_app, run_pipeline, run_pipeline_stream
from src.pipeline import VoiceQAPipeline


@pytest.fixture
def pipeline() -> VoiceQAPipeline:
    return VoiceQAPipeline.from_config({
        "components": {
            "vad": {"name": "mock"},
            "separation": {"name": "mock", "params": {"source_count": 2}},
            "enhancement": {"name": "mock"},
            "asr": {"name": "mock", "params": {"transcript": "hello"}},
            "qa": {"type": "mock", "answer_text": "answer."},
            "tts": {"name": "mock"},
        }
    })


def test_both_buttons_receive_enabled_checkboxes(pipeline: VoiceQAPipeline) -> None:
    config = build_app(pipeline).get_config_file()
    components = {component["id"]: component for component in config["components"]}
    for api_name in ("run_pipeline", "run_pipeline_stream"):
        dependency = next(
            item for item in config["dependencies"] if item.get("api_name") == api_name
        )
        inputs = [components[index] for index in dependency["inputs"]]
        assert [item["type"] for item in inputs] == ["audio", "checkbox", "checkbox"]
        assert "DTLN" in inputs[1]["props"]["label"]
        assert "SepFormer" in inputs[2]["props"]["label"]
        assert [item["props"]["value"] for item in inputs[1:]] == [True, True]


@pytest.mark.parametrize("enhancement,separation", [
    (False, False), (False, True), (True, False), (True, True),
])
def test_callbacks_forward_both_options(
    pipeline: VoiceQAPipeline, enhancement: bool, separation: bool,
) -> None:
    audio = (16_000, np.zeros(160, dtype=np.float32))
    transcript, answer, output, trace, status = run_pipeline(
        audio, pipeline, enhancement, separation
    )
    assert transcript == ("hello hello" if separation else "hello")
    assert answer == "answer."
    assert output is not None and status.startswith("✅")
    assert [row[0] for row in trace] == [
        "vad", *(["separation"] if separation else []),
        *(["enhancement"] if enhancement else []), "asr", "qa", "tts",
    ]

    updates = list(run_pipeline_stream(audio, pipeline, enhancement, separation))
    assert updates[-1][0] == transcript
    assert updates[-1][1] == answer
    assert updates[-1][3].startswith("✅")
