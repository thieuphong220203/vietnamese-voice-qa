from __future__ import annotations

from collections.abc import Iterator
from threading import Thread

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)

from .base import QuestionAnswerer


class QwenQA(QuestionAnswerer):
    """Question-answering component powered by Qwen."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str = "auto",
        max_new_tokens: int = 128,
        system_prompt: str = (
            "Bạn là một trợ lý hữu ích. "
            "Hãy trả lời bằng tiếng Việt, rõ ràng và ngắn gọn."
        ),
    ) -> None:
        self.model_id = model_id
        self.device = self._resolve_device(device)
        self.dtype = self._resolve_dtype(self.device)
        self.max_new_tokens = max_new_tokens
        self.system_prompt = system_prompt

        print(f"Loading QA model: {self.model_id}")
        print(f"Device: {self.device}")
        print(f"Dtype: {self.dtype}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            dtype=self.dtype,
        )

        self.model.to(self.device)
        self.model.eval()

    def answer(self, question: str) -> str:
        """Return the complete answer."""

        return "".join(self.stream_answer(question)).strip()

    def stream_answer(self, question: str) -> Iterator[str]:
        """Yield answer fragments while Qwen is generating text."""

        question = question.strip()

        if not question:
            return

        messages = [
            {
                "role": "system",
                "content": self.system_prompt,
            },
            {
                "role": "user",
                "content": question,
            },
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        model_inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        )

        model_inputs = {
            name: tensor.to(self.device)
            for name, tensor in model_inputs.items()
        }

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        generation_kwargs = {
            **model_inputs,
            "streamer": streamer,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        generation_thread = Thread(
            target=self._generate,
            kwargs=generation_kwargs,
            daemon=True,
        )

        generation_thread.start()

        for fragment in streamer:
            if fragment:
                yield fragment

        generation_thread.join()

    def _generate(self, **generation_kwargs: object) -> None:
        with torch.inference_mode():
            self.model.generate(**generation_kwargs)

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but no CUDA GPU is available."
            )

        if device not in {"cpu", "cuda"}:
            raise ValueError(
                "device must be 'auto', 'cpu', or 'cuda'"
            )

        return device

    @staticmethod
    def _resolve_dtype(device: str) -> torch.dtype:
        if device == "cuda":
            return torch.float16

        return torch.float32