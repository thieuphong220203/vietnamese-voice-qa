"""Model-agnostic orchestration for the end-to-end voice QA flow."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
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

from time import perf_counter


@dataclass(frozen=True, slots=True)
class StageTrace:
    name: str
    input_count: int
    output_count: int
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class PipelineResult:
    transcript: str
    answer: str
    output_audio: AudioData
    speech_segments: tuple[SpeechSegment, ...]
    separated_sources: tuple[AudioData, ...]
    enhanced_sources: tuple[AudioData, ...]
    trace: tuple[StageTrace, ...]

@dataclass(frozen=True, slots=True)
class StreamUpdate:
    transcript: str
    answer: str
    audio_chunk: AudioData | None = None


def _take_phrase(buffer: str) -> tuple[str, str] | None:
    """Lấy một câu đã hoàn chỉnh; tránh gửi từng token lẻ vào TTS."""
    for index, char in enumerate(buffer):
        if char in ".!?\n":
            phrase = buffer[: index + 1].strip()
            if phrase:
                return phrase, buffer[index + 1 :].lstrip()

    # Câu quá dài mà chưa có dấu kết thúc: ngắt ở khoảng trắng.
    if len(buffer) >= 120:
        split_at = buffer.rfind(" ", 0, 120)
        if split_at > 0:
            return buffer[:split_at].strip(), buffer[split_at + 1 :]

    return None


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
        t0 = perf_counter()
        segments = tuple(self.vad.detect(audio))
        t1 = perf_counter()

        separated = tuple(
            source
            for segment in segments
            for source in self.separator.separate(segment.audio)
        )
        t2 = perf_counter()
        enhanced = tuple(self.enhancer.enhance(source) for source in separated)
        t3 = perf_counter()
        transcript_parts = tuple(
            text for source in enhanced if (text := self.asr.transcribe(source).strip())
        )
        transcript = self.transcript_separator.join(transcript_parts)
        t4 = perf_counter()
        answer = self.qa.answer(transcript)
        t5 = perf_counter()
        output_audio = self.tts.synthesize(answer)
        t6 = perf_counter()

        trace = (
            StageTrace("vad", 1, len(segments), t1 - t0),
            StageTrace("separation", len(segments), len(separated), t2 - t1),
            StageTrace("enhancement", len(separated), len(enhanced), t3 - t2),
            StageTrace("asr", len(enhanced), len(transcript_parts), t4 - t3),
            StageTrace("qa", 1, 1, t5 - t4),
            StageTrace("tts", 1, 1, t6 - t5),
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

    def stream(self, audio: AudioData) -> Iterator[StreamUpdate]:
        """Stream câu trả lời và audio, giữ nguyên run() cũ."""

        segments = tuple(self.vad.detect(audio))
        separated = tuple(
            source
            for segment in segments
            for source in self.separator.separate(segment.audio)
        )
        enhanced = tuple(self.enhancer.enhance(source) for source in separated)
        transcript_parts = tuple(
            text
            for source in enhanced
            if (text := self.asr.transcribe(source).strip())
        )
        transcript = self.transcript_separator.join(transcript_parts)

        yield StreamUpdate(transcript=transcript, answer="")

        answer = ""
        pending = ""

        for fragment in self.qa.stream_answer(transcript):
            if not fragment:
                continue

            answer += fragment
            pending += fragment
            yield StreamUpdate(transcript=transcript, answer=answer)

            while (ready := _take_phrase(pending)) is not None:
                phrase, pending = ready
                for chunk in self.tts.stream_synthesize(phrase):
                    if chunk.frame_count:
                        yield StreamUpdate(
                            transcript=transcript,
                            answer=answer,
                            audio_chunk=chunk,
                        )

        # Đọc nốt phần cuối nếu Qwen không kết thúc bằng dấu câu.
        if pending.strip():
            for chunk in self.tts.stream_synthesize(pending.strip()):
                if chunk.frame_count:
                    yield StreamUpdate(
                        transcript=transcript,
                        answer=answer,
                        audio_chunk=chunk,
                    )
