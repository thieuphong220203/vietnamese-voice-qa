"""Adapter for the fine-tuned DTLN Keras model used by the Colab notebook."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

# DTLN's upstream code targets TensorFlow's bundled Keras 2 API.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np

from src.audio.types import AudioData
from src.data.wav_utils import resample_linear

from .base import SpeechEnhancer

DTLN_SOURCE_URL = "https://github.com/breizhn/DTLN.git"
DTLN_INPUT_RATE = 16_000
DTLN_PADDING = 384


class DTLNEnhancer(SpeechEnhancer):
    """Run DTLN on mono audio and return it in the input sample-rate format.

    ``checkpoint`` must point to the fine-tuned ``.weights.h5`` file produced by
    ``ML (2).ipynb``. The upstream DTLN source is cloned lazily into ``source_dir``
    if it is not already there. TensorFlow and the upstream module are imported
    only when the first audio is enhanced.
    """

    def __init__(
        self,
        checkpoint: str | Path | None = "checkpoints/enhancement/dtln/DTLN_vi_snr5_lr1e-4_run01.weights.h5",
        baseline_checkpoint: str | Path | None = None,
        source_dir: str | Path = "checkpoints/enhancement/DTLN",
        sample_rate: int = DTLN_INPUT_RATE,
        device: str = "auto",
        model: Any = None,
    ) -> None:
        if sample_rate != DTLN_INPUT_RATE:
            raise ValueError(f"DTLN expects {DTLN_INPUT_RATE} Hz audio; got {sample_rate}")
        if device not in {"auto", "cpu", "gpu"}:
            raise ValueError("device must be 'auto', 'cpu', or 'gpu'")
        self.checkpoint = Path(checkpoint) if checkpoint else None
        self.source_dir = Path(source_dir)
        self.baseline_checkpoint = (
            Path(baseline_checkpoint)
            if baseline_checkpoint
            else self.source_dir / "pretrained_model" / "model.h5"
        )
        self.sample_rate = sample_rate
        self.device = device
        self._model = model
        self._tf: Any = None
        self.loaded_checkpoint: Path | None = None

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model

        source_file = self.source_dir / "DTLN_model.py"
        if not source_file.is_file():
            self.source_dir.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["git", "clone", "--depth", "1", DTLN_SOURCE_URL, str(self.source_dir)],
                check=True,
            )
        if not source_file.is_file():
            raise FileNotFoundError(f"DTLN_model.py was not found under {self.source_dir}")

        # Failover logic tương tự SepFormer: ưu tiên fine-tune, fallback về pretrained gốc
        target_checkpoint: Path | None = None
        if self.checkpoint and self.checkpoint.is_file():
            target_checkpoint = self.checkpoint
        elif self.baseline_checkpoint and self.baseline_checkpoint.is_file():
            warnings.warn(
                f"Không thấy checkpoint fine-tune {self.checkpoint}; "
                f"tự động dùng trọng số gốc của DTLN tại {self.baseline_checkpoint}.",
                stacklevel=2,
            )
            target_checkpoint = self.baseline_checkpoint
        else:
            raise FileNotFoundError(
                f"Không tìm thấy checkpoint DTLN tại {self.checkpoint} "
                f"và không thấy baseline tại {self.baseline_checkpoint}. "
                "Vui lòng đặt checkpoint vào checkpoints/enhancement/dtln/ "
                "hoặc để DTLN tải pretrained model gốc."
            )

        # The original DTLN implementation uses the legacy Keras 2 API.
        try:
            import tensorflow as tf
        except ImportError as error:
            raise RuntimeError(
                "TensorFlow is required for DTLN. Install the optional dependencies "
                "with: python -m pip install -r requirements-dtln.txt"
            ) from error

        source_path = str(self.source_dir.resolve())
        if source_path not in sys.path:
            sys.path.insert(0, source_path)

        # DTLN_model.py gọi tf.config.experimental.set_memory_growth() cả ở top-level và bên trong __init__().
        # Khi GPU context đã được khởi tạo, TF sẽ văng lỗi RuntimeError. Bọc an toàn để bỏ qua lỗi này:
        for cfg_obj in (getattr(tf, "config", None), getattr(getattr(tf, "config", None), "experimental", None)):
            if cfg_obj and hasattr(cfg_obj, "set_memory_growth"):
                _orig_fn = getattr(cfg_obj, "set_memory_growth")
                def _make_safe(fn: Any) -> Any:
                    def _safe_call(*args: Any, **kwargs: Any) -> Any:
                        try:
                            return fn(*args, **kwargs)
                        except RuntimeError:
                            return None
                    return _safe_call
                setattr(cfg_obj, "set_memory_growth", _make_safe(_orig_fn))

        try:
            dtln_module = importlib.import_module("DTLN_model")
            dtln_class = dtln_module.DTLN_model
        except (ImportError, AttributeError) as error:
            raise RuntimeError(
                f"Could not import DTLN_model from {self.source_dir}. "
                "Install the DTLN dependencies (including wavinfo) and retry."
            ) from error

        has_gpu = bool(tf.config.list_physical_devices("GPU"))
        if self.device == "gpu" and not has_gpu:
            raise RuntimeError("DTLN device='gpu' was requested, but TensorFlow found no GPU")
        tf_device = "/GPU:0" if self.device == "gpu" or (self.device == "auto" and has_gpu) else "/CPU:0"

        with tf.device(tf_device):
            model_wrapper = dtln_class()
            model_wrapper.build_DTLN_model()
            model_wrapper.model.load_weights(str(target_checkpoint))
        self._tf = tf
        self._model = model_wrapper.model
        self.loaded_checkpoint = target_checkpoint
        return self._model

    def _enhance_mono(self, samples: np.ndarray) -> np.ndarray:
        model = self._load_model()
        padded = np.pad(samples.astype(np.float32, copy=False), (DTLN_PADDING, DTLN_PADDING))
        batch = padded[np.newaxis, :]
        if self._tf is None:
            prediction = model.predict_on_batch(batch)
        else:
            has_gpu = bool(self._tf.config.list_physical_devices("GPU"))
            tf_device = "/GPU:0" if self.device == "gpu" or (self.device == "auto" and has_gpu) else "/CPU:0"
            with self._tf.device(tf_device):
                prediction = model.predict_on_batch(batch)
        prediction = np.asarray(prediction, dtype=np.float32).reshape(-1)
        # DTLN sử dụng STFT với hop size 128 mẫu, nên phần padding ở cuối có thể bị lệch một vài mẫu
        # do làm tròn khung STFT. Cắt đúng đoạn tiếng nói gốc và bù padding nếu cần:
        start = min(DTLN_PADDING, len(prediction))
        end = min(start + len(samples), len(prediction))
        out = prediction[start:end]
        if len(out) < len(samples):
            out = np.pad(out, (0, len(samples) - len(out)))
        return out

    def enhance(self, audio: AudioData) -> AudioData:
        if audio.frame_count == 0:
            return audio

        # The pipeline audio contract preserves channel layout; DTLN itself is mono.
        interleaved = np.asarray(audio.samples, dtype=np.float32).reshape(-1, audio.channels)
        mono = interleaved.mean(axis=1, dtype=np.float32)
        model_input = resample_linear(mono, audio.sample_rate, self.sample_rate)
        enhanced = self._enhance_mono(model_input)
        if audio.sample_rate != self.sample_rate:
            enhanced = resample_linear(enhanced, self.sample_rate, audio.sample_rate)

        target_length = audio.frame_count
        enhanced = enhanced[:target_length]
        if len(enhanced) < target_length:
            enhanced = np.pad(enhanced, (0, target_length - len(enhanced)))

        return audio.with_samples(
            enhanced.tolist(),
            channels=1,
            metadata={
                **audio.metadata,
                "enhancement_model": "dtln",
                "enhancement_checkpoint": str(self.loaded_checkpoint or self.checkpoint),
                "enhancement_sample_rate": self.sample_rate,
            },
        )
