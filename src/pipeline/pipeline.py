"""Model-agnostic orchestration for the end-to-end voice QA flow."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.asr import SpeechRecognizer, create_asr
from src.audio import AudioData, SpeechSegment, load_wav
from src.enhancement import SpeechEnhancer, create_enhancer
from src.qa import QuestionAnswerer, create_qa
from src.separation import SpeechSeparator, create_separator
from src.tts import SpeechSynthesizer, create_tts
from src.vad import VoiceActivityDetector, create_vad


@dataclass(frozen=True, slots=True)
class StageTrace:
    name: str
    input_count: int
    output_count: int


@dataclass(frozen=True, slots=True)
class PipelineResult:
    transcript: str
    answer: str
    output_audio: AudioData
    speech_segments: tuple[SpeechSegment, ...]
    separated_sources: tuple[AudioData, ...]
    enhanced_sources: tuple[AudioData, ...]
    trace: tuple[StageTrace, ...]


class VoiceQAPipeline:
    """Coordinate replaceable components using only their public contracts."""

    def __init__(
        self,
        *,
        vad: VoiceActivityDetector,
        separator: SpeechSeparator,
        enhancer: SpeechEnhancer,
        asr: SpeechRecognizer,
        qa: QuestionAnswerer,
        tts: SpeechSynthesizer,
        transcript_separator: str = " ",
    ) -> None:
        self.vad = vad
        self.separator = separator
        self.enhancer = enhancer
        self.asr = asr
        self.qa = qa
        self.tts = tts
        self.transcript_separator = transcript_separator

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "VoiceQAPipeline":
        components = config.get("components")
        if not isinstance(components, Mapping):
            raise ValueError("pipeline config must contain a 'components' mapping")

        required = ("vad", "separation", "enhancement", "asr", "qa", "tts")
        missing = [name for name in required if not isinstance(components.get(name), Mapping)]
        if missing:
            raise ValueError(f"missing component configs: {', '.join(missing)}")

        options = config.get("pipeline", {})
        if not isinstance(options, Mapping):
            raise ValueError("'pipeline' must be a mapping when provided")

        return cls(
            vad=create_vad(components["vad"]),
            separator=create_separator(components["separation"]),
            enhancer=create_enhancer(components["enhancement"]),
            asr=create_asr(components["asr"]),
            qa=create_qa(components["qa"]),
            tts=create_tts(components["tts"]),
            transcript_separator=str(options.get("transcript_separator", " ")),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "VoiceQAPipeline":
        with Path(path).open("r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file) or {}
        if not isinstance(config, Mapping):
            raise ValueError("pipeline YAML root must be a mapping")
        return cls.from_config(config)

    def run_file(self, path: str | Path) -> PipelineResult:
        return self.run(load_wav(path))

    def run(self, audio: AudioData) -> PipelineResult:
        segments = tuple(self.vad.detect(audio))

        separated = tuple(
            source
            for segment in segments
            for source in self.separator.separate(segment.audio)
        )
        enhanced = tuple(self.enhancer.enhance(source) for source in separated)
        transcript_parts = tuple(
            text for source in enhanced if (text := self.asr.transcribe(source).strip())
        )
        transcript = self.transcript_separator.join(transcript_parts)
        answer = self.qa.answer(transcript)
        output_audio = self.tts.synthesize(answer)

        trace = (
            StageTrace("vad", 1, len(segments)),
            StageTrace("separation", len(segments), len(separated)),
            StageTrace("enhancement", len(separated), len(enhanced)),
            StageTrace("asr", len(enhanced), len(transcript_parts)),
            StageTrace("qa", 1, 1),
            StageTrace("tts", 1, 1),
        )
        return PipelineResult(
            transcript=transcript,
            answer=answer,
            output_audio=output_audio,
            speech_segments=segments,
            separated_sources=separated,
            enhanced_sources=enhanced,
            trace=trace,
        )
