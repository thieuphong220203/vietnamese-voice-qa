"""SepFormer (SpeechBrain) cho bài toán tách giọng.

Model mặc định: ``speechbrain/sepformer-whamr16k`` (train trên tiếng Anh,
WHAMR!, 16 kHz, tách 2 người nói khi có nhiễu và vang). Có thể nạp thêm
checkpoint đã fine-tune trên tiếng Việt qua tham số ``checkpoint``.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Literal

import numpy as np

from src.audio.types import AudioData

from .base import SpeechSeparator
from .selection import SELECTION_MODES, rank_by_energy

DEFAULT_MODEL_ID = "speechbrain/sepformer-whamr16k"


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def model_sample_rate(model: Any, model_id: str) -> int:
    """Lấy sample rate từ hparams của model; nếu không có thì đoán theo tên."""

    rate = getattr(getattr(model, "hparams", None), "sample_rate", None)
    if rate:
        return int(rate)
    if "48k" in model_id:
        return 48_000
    if "16k" in model_id:
        return 16_000
    return 8_000  # các model WSJ0-2mix / Libri2Mix gốc chạy 8 kHz


def resample(signal: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float32)
    if source_rate == target_rate:
        return signal
    try:
        import torch
        import torchaudio.functional as F

        tensor = torch.from_numpy(np.ascontiguousarray(signal))
        return F.resample(tensor, source_rate, target_rate).numpy()
    except ImportError:
        duration = len(signal) / source_rate
        target_len = max(1, round(duration * target_rate))
        old_x = np.arange(len(signal)) / source_rate
        new_x = np.arange(target_len) / target_rate
        return np.interp(new_x, old_x, signal).astype(np.float32)


def load_sepformer(model_id: str, savedir: str | Path | None, device: str) -> Any:
    try:
        from speechbrain.inference.separation import SepformerSeparation
    except ImportError:  # SpeechBrain < 1.0
        from speechbrain.pretrained import SepformerSeparation  # type: ignore

    if savedir is None:
        savedir = Path("checkpoints/separation/pretrained") / model_id.replace("/", "__")
    return SepformerSeparation.from_hparams(
        source=model_id,
        savedir=str(savedir),
        run_opts={"device": device},
    )


def load_finetuned_weights(model: Any, checkpoint: str | Path, device: str) -> None:
    """Nạp trọng số do ``src.separation.train`` lưu (encoder/masknet/decoder)."""

    import torch

    state = torch.load(str(checkpoint), map_location=device, weights_only=False)
    for name in ("encoder", "masknet", "decoder"):
        if name not in state:
            raise KeyError(f"checkpoint {checkpoint} thiếu khóa {name!r}")
        getattr(model.mods, name).load_state_dict(state[name])


class SepFormerSeparator(SpeechSeparator):
    """Tách một đoạn audio thành các giọng riêng.

    Tham số (đặt trong YAML, mục ``params``):
        model_id: model trên Hugging Face.
        device: ``auto`` | ``cpu`` | ``cuda``.
        savedir: thư mục cache model (mặc định trong ``checkpoints/separation``).
        checkpoint: file ``.pt`` fine-tune (tùy chọn).
        output: ``primary`` = chỉ trả giọng to nhất (dùng cho pipeline QA);
            ``all`` = trả mọi giọng, xếp từ to đến nhỏ.
        normalize: chuẩn hóa biên độ đầu ra về ``target_peak``.
        model: đối tượng model dựng sẵn (dùng cho test, không đặt trong YAML).
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "auto",
        savedir: str | None = None,
        checkpoint: str | None = None,
        output: Literal["primary", "all"] = "primary",
        normalize: bool = True,
        target_peak: float = 0.9,
        model: Any = None,
    ) -> None:
        if output not in SELECTION_MODES:
            raise ValueError(f"output phải là một trong {SELECTION_MODES}, nhận {output!r}")
        self.model_id = model_id
        self.device = resolve_device(device)
        self.output = output
        self.normalize = normalize
        self.target_peak = target_peak

        self.model = model if model is not None else load_sepformer(model_id, savedir, self.device)
        self.checkpoint = None
        if checkpoint:
            if Path(checkpoint).exists():
                load_finetuned_weights(self.model, checkpoint, self.device)
                self.checkpoint = str(checkpoint)
            else:
                # Thành viên khác clone repo sẽ không có file .pt (bị .gitignore) -> dùng model gốc.
                warnings.warn(
                    f"Không thấy checkpoint {checkpoint}; dùng trọng số gốc của {model_id}.",
                    stacklevel=2,
                )
        self.sample_rate = model_sample_rate(self.model, model_id)

    def separate_array(self, mixture: np.ndarray, sample_rate: int) -> list[np.ndarray]:
        """Mixture mono -> danh sách nguồn (cùng sample rate, cùng độ dài với đầu vào)."""

        import torch

        mixture = np.asarray(mixture, dtype=np.float32)
        signal = resample(mixture, sample_rate, self.sample_rate)
        batch = torch.from_numpy(np.ascontiguousarray(signal)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            estimates = self.model.separate_batch(batch)  # [1, T, S]
        estimates = estimates[0].detach().cpu().float().numpy()

        sources = []
        for index in range(estimates.shape[-1]):
            source = resample(estimates[:, index], self.sample_rate, sample_rate)
            source = source[: len(mixture)]
            if len(source) < len(mixture):
                source = np.pad(source, (0, len(mixture) - len(source)))
            sources.append(source.astype(np.float32))
        return sources

    def _postprocess(self, source: np.ndarray) -> np.ndarray:
        if not self.normalize:
            return source
        peak = float(np.max(np.abs(source))) if source.size else 0.0
        return source if peak == 0 else source * (self.target_peak / peak)

    def separate(self, audio: AudioData) -> tuple[AudioData, ...]:
        if audio.frame_count == 0:
            return ()
        samples = np.asarray(audio.samples, dtype=np.float32)
        if audio.channels > 1:
            samples = samples.reshape(-1, audio.channels).mean(axis=1)

        sources = self.separate_array(samples, audio.sample_rate)
        order = rank_by_energy(sources)  # chọn theo năng lượng TRƯỚC khi chuẩn hóa
        if self.output == "primary":
            order = order[:1]

        return tuple(
            AudioData.from_iterable(
                self._postprocess(sources[index]).tolist(),
                sample_rate=audio.sample_rate,
                channels=1,
                metadata={
                    **audio.metadata,
                    "separation_model": self.model_id,
                    "separation_checkpoint": self.checkpoint,
                    "source_index": index,
                    "energy_rank": rank,
                },
            )
            for rank, index in enumerate(order)
        )
