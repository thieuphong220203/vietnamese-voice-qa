from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np

from src.tts.factory import create_tts


def test_vieneu_factory_streams_and_combines_audio(monkeypatch) -> None:
    class FakeVieneu:
        sample_rate = 48_000

        def __init__(self, **options):
            assert options == {"backend": "onnx", "precision": "fp32"}

        def infer_stream(self, text, *, voice):
            assert (text, voice) == ("Xin chào", "Hải Đăng")
            yield np.array([0.1, 0.2], dtype=np.float32)
            yield np.array([0.3], dtype=np.float32)

    monkeypatch.setitem(sys.modules, "vieneu", SimpleNamespace(Vieneu=FakeVieneu))
    tts = create_tts({"name": "vieneu", "params": {"voice": "Hải Đăng"}})

    chunks = list(tts.stream_synthesize("Xin chào"))
    assert [chunk.frame_count for chunk in chunks] == [2, 1]
    assert all(chunk.sample_rate == 48_000 for chunk in chunks)

    complete = tts.synthesize("Xin chào")
    assert complete.frame_count == 3
    assert complete.samples == tuple(sample for chunk in chunks for sample in chunk.samples)
