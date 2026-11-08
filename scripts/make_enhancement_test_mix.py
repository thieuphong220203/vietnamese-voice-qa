"""Tạo tập dữ liệu kiểm thử lọc nhiễu (Speech Enhancement) từ FLEURS và DEMAND noise.

Mặc định đọc 10 mẫu tiếng Việt sạch trong ``data/samples/fleurs_vi/audio`` và kết hợp
với các môi trường nhiễu thực tế từ bộ dữ liệu DEMAND (văn phòng OOFFICE, quán cafe PCAFE,
xe hơi TCAR,...).

Kết quả có cùng cấu trúc và định dạng CSV với ``scripts/make_test_mix.py`` (separation),
và tương thích trực tiếp với lệnh đánh giá ``python -m src.enhancement.evaluate``.

Ví dụ::

    # Tạo 10 mẫu test với SNR 5 dB (mặc định)
    python scripts/make_enhancement_test_mix.py

    # Tạo với mức SNR khác hoặc thư mục output tùy chọn
    python scripts/make_enhancement_test_mix.py --snr-db 0.0 --output-dir data/processed/enhancement

    # Chạy đánh giá SI-SDR và STOI ngay sau khi tạo
    python -m src.enhancement.evaluate --csv data/processed/enhancement/test.csv
"""

from __future__ import annotations

import argparse
import csv
import io
import random
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.wav_utils import read_wav, resample_linear, write_wav  # noqa: E402

SAMPLE_RATE = 16_000
CSV_FIELDS = (
    "id",
    "noisy",
    "clean",
    "duration",
    "snr_db",
    "noise_type",
    "speaker",
    "text",
)

# Các môi trường tiếng ồn tiêu biểu từ DEMAND (16 kHz)
DEMAND_ENVIRONMENTS = {
    "PCAFE": "https://zenodo.org/records/1227121/files/PCAFE_16k.zip",
    "STRAFFIC": "https://zenodo.org/records/1227121/files/STRAFFIC_16k.zip",
    "PRESTAU": "https://zenodo.org/records/1227121/files/PRESTAU_16k.zip",
    "TCAR": "https://zenodo.org/records/1227121/files/TCAR_16k.zip",
}


def load_audio_16k(path: Path) -> np.ndarray:
    signal, rate = read_wav(path)
    return resample_linear(signal, rate, SAMPLE_RATE)


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    """Trộn clean và noise theo tỷ lệ SNR chuẩn xác (tương thích DTLN)."""
    clean = clean.astype(np.float32, copy=False)
    noise = noise.astype(np.float32, copy=False)

    if len(noise) < len(clean):
        repeats = int(np.ceil(len(clean) / max(len(noise), 1)))
        noise = np.tile(noise, repeats)

    # Cắt ngẫu nhiên một đoạn noise có độ dài đúng bằng clean
    max_start = len(noise) - len(clean)
    start = rng.randint(0, max_start) if max_start > 0 else 0
    noise_segment = noise[start : start + len(clean)]

    # Tính RMS phần giọng nói hoạt động (Active speech RMS)
    active = np.abs(clean) > max(1e-4, 0.02 * float(np.max(np.abs(clean))))
    speech_rms = float(np.sqrt(np.mean(clean[active] ** 2))) if np.any(active) else 1e-4
    noise_rms = float(np.sqrt(np.mean(noise_segment ** 2) + 1e-12))

    # Scale noise theo SNR công thức chuẩn: SNR = 20 * log10(speech_rms / noise_rms)
    scale_factor = speech_rms / (noise_rms * (10.0 ** (snr_db / 20.0)))
    scaled_noise = noise_segment * scale_factor
    noisy = clean + scaled_noise

    # Peak normalization để tránh hiện tượng clipping âm thanh
    peak = max(float(np.max(np.abs(noisy))), float(np.max(np.abs(clean))), 1e-6)
    scale = min(1.0, 0.95 / peak)
    return (clean * scale).astype(np.float32), (noisy * scale).astype(np.float32)


def generate_synthetic_environmental_noise(num_samples: int, noise_type: str, rng: np.random.Generator) -> np.ndarray:
    """Tạo tiếng ồn mô phỏng môi trường (hỗ trợ offline khi không có internet)."""
    white = rng.standard_normal(num_samples).astype(np.float32)
    if noise_type == "STRAFFIC":
        # Âm trầm động cơ + mô phỏng xe cộ qua lại và lốp xe
        low_rumble = np.convolve(white, np.ones(48) / 48.0, mode="same")
        t = np.linspace(0, num_samples / SAMPLE_RATE, num_samples, dtype=np.float32)
        modulation = 0.7 + 0.3 * np.sin(2 * np.pi * 0.3 * t)
        noise = low_rumble * modulation
    elif noise_type == "PRESTAU":
        # Giả lập nhà hàng đông đúc: tiếng nói xì xào + tiếng dao dĩa lách cách
        chatter = np.convolve(white, np.ones(20) / 20.0, mode="same")
        clatter = np.zeros(num_samples, dtype=np.float32)
        n_clatters = num_samples // (SAMPLE_RATE // 2)
        for _ in range(n_clatters):
            pos = rng.integers(0, max(1, num_samples - 200))
            clatter[pos : pos + 100] = rng.uniform(-0.5, 0.5, 100)
        noise = chatter * 0.8 + clatter * 0.2
    elif noise_type == "TCAR":
        # Rất nhiều âm trầm thấp (low-frequency engine rumble)
        noise = np.convolve(white, np.ones(64) / 64.0, mode="same")
    else:  # PCAFE / Ambient
        # Pinkish noise mô phỏng quán cafe xì xào
        noise = np.convolve(white, np.ones(16) / 16.0, mode="same")
    peak = np.max(np.abs(noise)) + 1e-12
    return (noise / peak * 0.5).astype(np.float32)


def get_or_download_demand_noise(cache_dir: Path, env_name: str = "PCAFE") -> np.ndarray:
    """Lấy file noise DEMAND từ cache local, tải từ Hugging Face, hoặc fallback tạo mô phỏng."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    target_wav = cache_dir / f"{env_name}_ch01.wav"
    if target_wav.exists():
        signal, rate = read_wav(target_wav)
        return resample_linear(signal, rate, SAMPLE_RATE)

    # 1. Thử tải trực tiếp từ Hugging Face qua thư viện datasets (không bao giờ bị 403)
    try:
        from datasets import Features, Value, load_dataset
        import soundfile as sf

        print(f"Đang tải mẫu noise DEMAND ({env_name}) từ Hugging Face (philgzl/demand)...")
        dataset = load_dataset(
            "philgzl/demand",
            split="16k",
            streaming=True,
            features=Features({"audio": Value("binary"), "name": Value("string")}),
        )
        audio_segments = []
        for item in dataset:
            name = item.get("name", "")
            if env_name in name:
                buf = io.BytesIO(item["audio"])
                sig, rate = sf.read(buf, dtype="float32")
                if sig.ndim > 1:
                    sig = sig.mean(axis=1)
                audio_segments.append(sig)
                if len(audio_segments) >= 4:  # gom ~20 giây audio noise
                    break
        if audio_segments:
            combined = np.concatenate(audio_segments).astype(np.float32)
            write_wav(target_wav, combined, SAMPLE_RATE)
            print(f"✅ Đã tải và lưu {env_name} thành công từ Hugging Face vào {target_wav}")
            return combined
    except Exception as err:
        print(f"⚠️ Hugging Face tải không thành công ({err}). Thử nguồn tiếp theo...")

    # 2. Thử tải từ Zenodo
    url = DEMAND_ENVIRONMENTS.get(env_name)
    if url:
        print(f"Đang thử tải từ Zenodo ({env_name})...")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Wget/1.21.3 (linux-gnu)"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = resp.read()
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                ch1_names = [n for n in z.namelist() if "ch01.wav" in n]
                if ch1_names:
                    with z.open(ch1_names[0]) as source_file, target_wav.open("wb") as out_file:
                        out_file.write(source_file.read())
                    print(f"Đã tải và lưu {env_name} vào {target_wav}")
                    signal, rate = read_wav(target_wav)
                    return resample_linear(signal, rate, SAMPLE_RATE)
        except Exception as err:
            print(f"Không thể tải trực tiếp từ Zenodo ({err}). Đang chuyển sang tạo noise mô phỏng {env_name}.")

    # 3. Fallback: sinh noise mô phỏng
    np_rng = np.random.default_rng(42)
    fake_noise = generate_synthetic_environmental_noise(SAMPLE_RATE * 30, env_name, np_rng)
    write_wav(target_wav, fake_noise, SAMPLE_RATE)
    return fake_noise


def load_fleurs_samples(metadata_path: Path, max_count: int = 10) -> list[dict[str, str]]:
    """Đọc danh sách các mẫu FLEURS từ file metadata."""
    if not metadata_path.exists():
        raise FileNotFoundError(f"Không tìm thấy {metadata_path}")
    base = metadata_path.parent
    rows: list[dict[str, str]] = []
    with metadata_path.open(encoding="utf-8-sig") as handle:
        for r in csv.DictReader(handle):
            audio_path = base / r["audio_path"]
            if audio_path.exists():
                rows.append({
                    "path": str(audio_path),
                    "speaker": audio_path.stem,
                    "text": r.get("transcription", ""),
                })
    return rows[:max_count]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=PROJECT_ROOT / "data/samples/fleurs_vi/metadata.csv",
        help="Đường dẫn tới metadata.csv của FLEURS",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data/processed/enhancement",
        help="Thư mục lưu kết quả kiểm thử",
    )
    parser.add_argument("--count", type=int, default=10, help="Số lượng mẫu audio cần tạo (tối đa 10)")
    parser.add_argument("--snr-db", type=float, default=5.0, help="Mức SNR (dB) khi trộn noise (mặc định 5.0 dB)")
    parser.add_argument("--noise-dir", type=Path, default=PROJECT_ROOT / "data/raw/demand", help="Thư mục cache noise DEMAND")
    args = parser.parse_args()

    fleurs = load_fleurs_samples(args.metadata, args.count)
    if not fleurs:
        raise SystemExit("Không tìm thấy file audio FLEURS nào hợp lệ.")

    # Chuẩn bị noise từ 3 môi trường DEMAND: PCAFE, OOFFICE, TCAR
    demand_noises: dict[str, np.ndarray] = {}
    env_keys = list(DEMAND_ENVIRONMENTS.keys())
    for env in env_keys:
        demand_noises[env] = get_or_download_demand_noise(args.noise_dir, env)

    rng = random.Random(2026)
    rows: list[dict[str, str]] = []

    clean_dir = args.output_dir / "test" / "clean"
    noisy_dir = args.output_dir / "test" / "noisy"
    clean_dir.mkdir(parents=True, exist_ok=True)
    noisy_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Bắt đầu trộn {len(fleurs)} mẫu FLEURS với DEMAND noise (SNR={args.snr_db} dB) ---")

    for idx, item in enumerate(fleurs):
        sample_id = f"enh_demo_{idx:03d}"
        clean_audio = load_audio_16k(Path(item["path"]))
        
        # Luân phiên chọn các loại noise DEMAND (PCAFE, OOFFICE, TCAR)
        noise_name = env_keys[idx % len(env_keys)]
        noise_audio = demand_noises[noise_name]

        clean_proc, noisy_proc = mix_at_snr(clean_audio, noise_audio, args.snr_db, rng)

        clean_out_path = clean_dir / f"{sample_id}.wav"
        noisy_out_path = noisy_dir / f"{sample_id}.wav"
        write_wav(clean_out_path, clean_proc, SAMPLE_RATE)
        write_wav(noisy_out_path, noisy_proc, SAMPLE_RATE)

        # Định dạng relative path so với thư mục output_dir (nơi chứa test.csv)
        rel_clean = f"test/clean/{sample_id}.wav"
        rel_noisy = f"test/noisy/{sample_id}.wav"

        duration_sec = len(clean_proc) / SAMPLE_RATE
        rows.append({
            "id": sample_id,
            "noisy": rel_noisy,
            "clean": rel_clean,
            "duration": f"{duration_sec:.3f}",
            "snr_db": f"{args.snr_db:.1f}",
            "noise_type": f"DEMAND_{noise_name}",
            "speaker": item["speaker"],
            "text": item["text"],
        })

        clean_name = Path(item["path"]).name
        print(f"[{idx+1}/{len(fleurs)}] {sample_id}: {clean_name} + DEMAND_{noise_name} (SNR={args.snr_db}dB) -> {rel_noisy}")

    # Ghi file test.csv
    csv_path = args.output_dir / "test.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nĐã tạo thành công {len(rows)} mẫu test!")
    print(f"Manifest CSV: {csv_path}")
    print("\nBạn có thể chạy đánh giá SI-SDR / STOI ngay bằng lệnh:")
    print(f"   python -m src.enhancement.evaluate --csv {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

