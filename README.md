# Vietnamese Voice QA & Audio Processing Pipeline

Hệ thống xử lý âm thanh và hỏi đáp giọng nói tiếng Việt (End-to-End Vietnamese Voice QA System).

---

## 📁 Cấu Trúc Thư Mục (Project Structure)

```text
AdvancedML/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── configs/
│   ├── pipeline.yaml
│   ├── enhancement.yaml
│   ├── separation.yaml
│   ├── asr.yaml
│   ├── qa.yaml
│   └── tts.yaml
│
├── src/
│   ├── audio/
│   │   ├── types.py
│   │   ├── loader.py
│   │   └── preprocessing.py
│   │
│   ├── enhancement/
│   │   ├── base.py
│   │   ├── factory.py
│   │   ├── deepfilternet.py
│   │   ├── metricgan.py
│   │   ├── train.py
│   │   ├── infer.py
│   │   └── evaluate.py
│   │
│   ├── separation/
│   │   ├── base.py
│   │   ├── factory.py
│   │   ├── sepformer.py
│   │   ├── train.py
│   │   ├── infer.py
│   │   └── evaluate.py
│   │
│   ├── asr/
│   │   ├── base.py
│   │   ├── factory.py
│   │   ├── phowhisper.py
│   │   ├── whisper.py
│   │   ├── train.py
│   │   ├── infer.py
│   │   └── evaluate.py
│   │
│   ├── qa/
│   │   ├── base.py
│   │   ├── factory.py
│   │   ├── qwen.py
│   │   ├── train.py
│   │   ├── infer.py
│   │   └── evaluate.py
│   │
│   ├── tts/
│   │   ├── base.py
│   │   ├── factory.py
│   │   ├── viettts.py
│   │   ├── train.py
│   │   ├── infer.py
│   │   └── evaluate.py
│   │
│   ├── vad/
│   │   ├── base.py
│   │   ├── factory.py
│   │   └── silero.py
│   │
│   ├── data/
│   │   ├── audio_utils.py
│   │   ├── prepare_enhancement.py
│   │   └── prepare_separation.py
│   │
│   └── pipeline/
│       └── pipeline.py
│
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── run_pipeline.py
│   └── download_data.py
│
├── notebooks/
│   └── runner.ipynb
│
├── checkpoints/
│   ├── enhancement/
│   ├── separation/
│   ├── asr/
│   ├── qa/
│   └── tts/
│
├── data/
│   ├── raw/
│   │   ├── vivos/
│   │   ├── musan/
│   │   └── fleurs/
│   ├── processed/
│   │   ├── enhancement/
│   │   └── separation/
│   └── metadata/
│
├── outputs/
│   ├── audio/
│   ├── metrics/
│   └── logs/
│
├── app/
│   ├── main.py
│   └── ui.py
│
└── tests/
```

---

## 💡 Ý Nghĩa Các Thư Mục & Thành Phần

- **`configs/`**: Chọn model, checkpoint, tham số huấn luyện và inference mà không cần sửa code.
  - `pipeline.yaml`: Cấu hình toàn bộ hệ thống end-to-end (VAD, Separation, Enhancement, ASR, QA, TTS).
  - `enhancement.yaml`, `separation.yaml`, `asr.yaml`, `qa.yaml`, `tts.yaml`: Cấu hình chi tiết cho từng task riêng biệt.
- **`src/<module>/`**: Code triển khai thực tế của từng module.
  - `base.py`: Interface / abstract class chung cho model trong module.
  - `factory.py`: Khởi tạo model tương ứng dựa theo YAML configuration.
  - `<model>.py`: Wrapper cho từng model cụ thể (DeepFilterNet, MetricGAN, PhoWhisper, Qwen, VietTTS, Silero VAD,...).
  - `train.py`: Logic huấn luyện / fine-tuning độc lập.
  - `infer.py`: Chạy inference / dự đoán độc lập.
  - `evaluate.py`: Đánh giá chất lượng và tính toán metric.
- **`src/audio/`**: Định nghĩa types, loader và preprocessing dùng chung cho toàn bộ luồng audio.
- **`src/data/`**: Chuẩn bị dataset, tổng hợp nhiễu, trộn giọng nói cho enhancement & separation.
- **`src/pipeline/pipeline.py`**: Ghép nối các module thành luồng end-to-end:
  $$\text{Input Audio} \rightarrow \text{VAD} \rightarrow \text{Separation} \rightarrow \text{Enhancement} \rightarrow \text{ASR} \rightarrow \text{QA} \rightarrow \text{TTS} \rightarrow \text{Output Audio}$$
- **`scripts/`**: Entry points để chạy các tác vụ từ dòng lệnh hoặc Google Colab (`train.py`, `evaluate.py`, `run_pipeline.py`, `download_data.py`).
- **`notebooks/runner.ipynb`**: Notebook chạy thực nghiệm trên Google Colab.
- **`checkpoints/`**: Lưu trữ weights/checkpoints do quá trình train/fine-tune tạo ra.
- **`data/`**: Chứa dataset raw (`vivos`, `musan`, `fleurs`), processed và metadata.
- **`outputs/`**: Chứa output audio sinh ra, file metrics và log thực nghiệm.
- **`app/`**: Giao diện ứng dụng demo (`main.py`, `ui.py`).
- **`tests/`**: Unit test và integration test cho source code.

---

## 📐 Project Design Principles

1. **Tính độc lập**: Mỗi tác vụ ML là một module độc lập.
2. **Khả năng kiểm thử & phát triển riêng biệt**: Từng module có thể train, fine-tune, infer và evaluate riêng biệt.
3. **Decoupling**: Pipeline end-to-end không phụ thuộc cứng vào bất kỳ model implementation cụ thể nào.
4. **Cấu hình hóa**: Việc lựa chọn model và tham số được điều khiển hoàn toàn qua YAML configuration.
5. **Cấu trúc chuẩn hóa cho mỗi module**:
   - `base.py`: common interface
   - `factory.py`: select/create model
   - `<model_name>.py`: model-specific wrapper
   - `train.py`: training / fine-tuning
   - `infer.py`: inference
   - `evaluate.py`: evaluation
6. **Dễ mở rộng**: Khi thêm model mới, chỉ cần thêm wrapper model, đăng ký vào `factory.py` và cấu hình trong YAML mà không cần sửa pipeline chính.
7. **Quản lý weights**: Pretrained weights tải từ Hugging Face / Model Hub; checkpoint tự huấn luyện lưu riêng và chọn qua YAML.
8. **Môi trường tính toán**: Google Colab được sử dụng làm compute runner; dữ liệu lớn có thể liên kết qua Google Drive.
9. **Phân tách trách nhiệm**: `pipeline.py` chỉ điều phối luồng thực thi; UI/App gọi pipeline thay vì gọi trực tiếp model ML.

---

## Phase 1: Mock Pipeline

Skeleton hiện tại chạy toàn bộ luồng bằng mock component, không tải model hoặc
checkpoint thật:

```text
pipeline.yaml
  -> factory của từng module
  -> VAD -> Separation -> Enhancement -> ASR -> QA -> TTS
  -> PipelineResult
```

Cài dependency nhẹ và chạy demo với audio giả lập:

```bash
pip install -r requirements.txt
python scripts/run_pipeline.py
```

Chạy với WAV PCM và ghi audio output:

```bash
python scripts/run_pipeline.py --input path/to/input.wav --output outputs/audio/mock.wav
```

Chạy test:

```bash
pytest -q
```

Để thay model sau này, implement interface trong `base.py`, đăng ký wrapper ở
`factory.py`, rồi đổi `name` và `params` trong `configs/pipeline.yaml`. Pipeline
không chứa preprocessing riêng cho bất kỳ model nào; tensor conversion,
resampling bắt buộc hoặc tokenizer của model phải nằm trong wrapper tương ứng.
