"""Gradio user interface for the end-to-end voice QA pipeline."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TypeAlias

import gradio as gr
import numpy as np

from src.audio import AudioData
from src.pipeline import VoiceQAPipeline


AudioInput: TypeAlias = tuple[int, np.ndarray] | None
AudioOutput: TypeAlias = tuple[int, np.ndarray] | None

TARGET_SAMPLE_RATE = 16_000


def _to_float_waveform(samples: np.ndarray) -> np.ndarray:
    """Convert Gradio audio samples to a mono float32 waveform."""

    waveform = np.asarray(samples)

    if waveform.size == 0:
        raise ValueError("Audio không chứa sample nào.")

    if np.issubdtype(waveform.dtype, np.integer):
        dtype_info = np.iinfo(waveform.dtype)
        scale = float(max(abs(dtype_info.min), dtype_info.max))
        waveform = waveform.astype(np.float32) / scale
    else:
        waveform = waveform.astype(np.float32)

    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    elif waveform.ndim != 1:
        raise ValueError(f"Audio có shape không được hỗ trợ: {waveform.shape}")

    return waveform


def _resample_linear(
    waveform: np.ndarray,
    source_sample_rate: int,
    target_sample_rate: int,
) -> np.ndarray:
    """Resample a mono waveform with linear interpolation."""

    if source_sample_rate <= 0:
        raise ValueError("Sample rate đầu vào phải lớn hơn 0.")

    if source_sample_rate == target_sample_rate:
        return waveform

    target_length = round(
        len(waveform) * target_sample_rate / source_sample_rate
    )
    if target_length <= 0:
        raise ValueError("Audio quá ngắn để resample.")

    source_positions = np.arange(len(waveform), dtype=np.float64)
    target_positions = np.linspace(
        0,
        len(waveform) - 1,
        num=target_length,
        dtype=np.float64,
    )
    return np.interp(
        target_positions,
        source_positions,
        waveform,
    ).astype(np.float32)


def _to_audio_data(audio_input: AudioInput) -> AudioData:
    """Adapt Gradio audio to the model-agnostic pipeline type."""

    if audio_input is None:
        raise ValueError("Hãy upload hoặc thu âm trước khi chạy pipeline.")

    sample_rate, samples = audio_input
    waveform = _to_float_waveform(samples)
    waveform = _resample_linear(
        waveform,
        source_sample_rate=int(sample_rate),
        target_sample_rate=TARGET_SAMPLE_RATE,
    )

    return AudioData.from_iterable(
        waveform,
        sample_rate=TARGET_SAMPLE_RATE,
        channels=1,
        metadata={"source": "gradio"},
    )


def _to_gradio_audio(audio: AudioData) -> AudioOutput:
    """Adapt the shared AudioData output for the Gradio audio player."""

    if audio.frame_count == 0:
        return None

    waveform = np.asarray(audio.samples, dtype=np.float32)
    if audio.channels > 1:
        waveform = waveform.reshape(-1, audio.channels)

    # Chuyển đổi sang PCM 16-bit để Gradio audio player phát mượt mà không warning
    pcm_16 = np.clip(waveform * 32767.0, -32768.0, 32767.0).astype(np.int16)

    return audio.sample_rate, pcm_16


def _trace_rows(result_trace: tuple[object, ...]) -> list[list[object]]:
    return [
        [stage.name, stage.input_count, stage.output_count]
        for stage in result_trace
    ]


def run_pipeline(
    audio_input: AudioInput,
    pipeline: VoiceQAPipeline,
    progress: gr.Progress = gr.Progress(),
) -> tuple[str, str, AudioOutput, list[list[object]], str]:
    """Run one request through the already initialized pipeline."""

    started_at = time.perf_counter()

    try:
        progress(0.1, desc="Đang chuẩn bị audio...")
        audio = _to_audio_data(audio_input)

        progress(0.25, desc="Đang chạy pipeline...")
        result = pipeline.run(audio)

        progress(0.9, desc="Đang chuẩn bị kết quả...")
        output_audio = _to_gradio_audio(result.output_audio)
        trace = _trace_rows(result.trace)
        elapsed = time.perf_counter() - started_at

        status = (
            f"✅ Hoàn tất trong **{elapsed:.2f} giây** · "
            f"Input **{audio.duration_seconds:.2f} giây**, "
            f"mono **{audio.sample_rate} Hz**"
        )
        progress(1.0, desc="Hoàn tất")

        return (
            result.transcript,
            result.answer,
            output_audio,
            trace,
            status,
        )
    except Exception as error:
        error_message = f"{type(error).__name__}: {error}"
        return "", "", None, [], f"❌ **{error_message}**"


def build_app(pipeline: VoiceQAPipeline) -> gr.Blocks:
    """Build the UI around one reusable pipeline instance."""

    def submit(audio_input: AudioInput, progress: gr.Progress = gr.Progress()):
        return run_pipeline(audio_input, pipeline, progress)

    with gr.Blocks(
        title="Vietnamese Voice Assistant",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown(
            """
            # Vietnamese Voice Assistant
            Upload hoặc thu âm tiếng Việt, sau đó chạy toàn bộ pipeline
            **VAD → Separation → Enhancement → ASR → QA → TTS**.

            Các component mock sẽ tự được thay thế khi bạn đổi model trong
            `configs/pipeline.yaml`; giao diện không cần gọi model trực tiếp.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    label="Audio đầu vào",
                    sources=["upload", "microphone"],
                    type="numpy",
                    format="wav",
                )

                with gr.Row():
                    run_button = gr.Button(
                        "Chạy pipeline",
                        variant="primary",
                    )
                    clear_button = gr.ClearButton(
                        value="Xóa kết quả",
                    )

                status = gr.Markdown(
                    "Sẵn sàng. Model đã được load khi app khởi động."
                )

            with gr.Column(scale=1):
                transcript = gr.Textbox(
                    label="Transcript (ASR)",
                    lines=5,
                    interactive=False,
                )
                answer = gr.Textbox(
                    label="Câu trả lời (QA)",
                    lines=5,
                    interactive=False,
                )

        with gr.Row():
            output_audio = gr.Audio(
                label="Audio trả lời (TTS)",
                interactive=False,
            )
            trace = gr.Dataframe(
                headers=["Stage", "Inputs", "Outputs"],
                datatype=["str", "number", "number"],
                label="Pipeline trace",
                interactive=False,
            )

        outputs = [
            transcript,
            answer,
            output_audio,
            trace,
            status,
        ]

        run_button.click(
            fn=submit,
            inputs=[audio_input],
            outputs=outputs,
            show_progress="full",
            concurrency_limit=1,
            api_name="run_pipeline",
        )

        # Mẫu âm thanh tiếng Việt có sẵn để người dùng test nhanh
        examples_dir = Path("data/samples/fleurs_vi/audio")
        if examples_dir.exists():
            sample_files = sorted(list(examples_dir.glob("*.wav")))[:3]
            if sample_files:
                gr.Examples(
                    examples=[str(path) for path in sample_files],
                    inputs=[audio_input],
                    label="Audio mẫu tiếng Việt (FLEURS)",
                )

    return app
