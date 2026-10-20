"""PhoWhisper implementation of the speech-recognition interface."""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from src.asr import SpeechRecognizer
from src.audio import AudioData


DeviceOption = Literal["auto", "cpu", "cuda"]
DtypeOption = Literal["auto", "float16", "float32", "bfloat16"]


class PhoWhisperRecognizer(SpeechRecognizer):
    """Vietnamese speech recognition using a pretrained PhoWhisper model."""

    REQUIRED_SAMPLE_RATE = 16_000

    def __init__(
        self,
        model_id: str = "vinai/PhoWhisper-small",
        device: DeviceOption = "auto",
        dtype: DtypeOption = "auto",
    ) -> None:
        self.model_id = model_id
        self.device = self._resolve_device(device)
        self.torch_dtype = self._resolve_dtype(dtype)

        print(f"Loading model: {self.model_id}")
        print(f"Device: {self.device}")
        print(f"Dtype: {self.torch_dtype}")

        self.processor = AutoProcessor.from_pretrained(
            self.model_id
        )

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
        )

        # original_torch_load = torch.load

        # try:
        #     def compatible_torch_load(*args, **kwargs):
        #         kwargs["weights_only"] = False
        #         return original_torch_load(*args, **kwargs)

        #     torch.load = compatible_torch_load

        #     self.processor = AutoProcessor.from_pretrained(self.model_id)

        #     self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
        #         self.model_id,
        #         dtype=self.torch_dtype,
        #     )
        # finally:
        #     torch.load = original_torch_load

        self.model.to(self.device)
        self.model.eval()

        print("PhoWhisper loaded successfully.")

    def transcribe(self, audio: AudioData) -> str:
        """Convert one AudioData object into Vietnamese text."""

        if audio.frame_count == 0:
            return ""

        if audio.sample_rate != self.REQUIRED_SAMPLE_RATE:
            raise ValueError(
                f"PhoWhisper requires 16 kHz audio, but received {audio.sample_rate} Hz."
            )

        waveform = self._prepare_waveform(audio)

        inputs = self.processor(
            waveform,
            sampling_rate=audio.sample_rate,
            return_tensors="pt",
            return_attention_mask=True,
        )

        input_features = inputs.input_features.to(
            device=self.device,
            dtype=self.torch_dtype,
        )

        attention_mask = inputs.get("attention_mask")

        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        
        with torch.inference_mode():
            predicted_ids = self.model.generate(
                input_features=input_features,
                attention_mask=attention_mask,
            )

        transcript = self.processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True,
        )[0]

        return transcript.strip()

    @staticmethod
    def _prepare_waveform(audio: AudioData) -> np.ndarray:
        """Convert interleaved AudioData samples to a mono NumPy waveform."""

        waveform = np.asarray(
            audio.samples,
            dtype=np.float32,
        )

        if audio.channels > 1:
            waveform = waveform.reshape(
                -1,
                audio.channels,
            )
            waveform = waveform.mean(axis=1)

        return waveform

    @staticmethod
    def _resolve_device(
        requested_device: DeviceOption,
    ) -> torch.device:
        if requested_device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")

            return torch.device("cpu")

        if requested_device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA device was requested, but PyTorch did not detect a CUDA GPU."
                )

            return torch.device("cuda")

        if requested_device == "cpu":
            return torch.device("cpu")

        raise ValueError(
            f"Invalid device option: {requested_device!r}"
        )

    def _resolve_dtype(
        self,
        requested_dtype: DtypeOption,
    ) -> torch.dtype:
        if requested_dtype == "auto":
            if self.device.type == "cuda":
                return torch.float16

            return torch.float32

        dtype_mapping: dict[str, torch.dtype] = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }

        try:
            selected_dtype = dtype_mapping[requested_dtype]
        except KeyError as error:
            raise ValueError(
                f"Invalid dtype option: {requested_dtype!r}"
            ) from error

        if self.device.type == "cpu" and selected_dtype == torch.float16:
            raise ValueError(
                "float16 is not recommended on CPU. Please use dtype='float32' or dtype='auto'."
            )

        return selected_dtype