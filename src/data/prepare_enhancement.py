"""Chuẩn bị dữ liệu kiểm thử và đánh giá cho module Speech Enhancement.

Kết hợp giọng nói sạch (FLEURS hoặc VIVOS) với tiếng ồn môi trường (DEMAND)
tại các mức SNR tùy chọn (-5, 0, 5, 10, 15 dB).

Ví dụ::

    # Tạo nhanh bộ test 10 mẫu từ FLEURS + DEMAND noise (SNR 5 dB)
    python -m src.data.prepare_enhancement

    # Chỉ định mức SNR khác hoặc thư mục output tùy chọn
    python -m src.data.prepare_enhancement --snr-db 0.0 --output-dir data/processed/enhancement
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.make_enhancement_test_mix import (  # noqa: E402
    CSV_FIELDS,
    SAMPLE_RATE,
    load_audio_16k,
    main,
    mix_at_snr,
)

__all__ = ["CSV_FIELDS", "SAMPLE_RATE", "load_audio_16k", "mix_at_snr", "main"]

if __name__ == "__main__":
    raise SystemExit(main())
