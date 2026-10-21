# Vietnamese Voice QA & Audio Processing Pipeline

Hệ thống xử lý âm thanh và hỏi đáp giọng nói tiếng Việt (End-to-End Vietnamese Voice QA System).

---

## 1. Tổng Quan Kiến Trúc

Hệ thống hoạt động theo luồng xử lý pipeline nối tiếp:

```text
Input Audio -> VAD -> Speech Separation -> Speech Enhancement -> ASR -> QA -> TTS -> Output Audio
```

- VAD (Voice Activity Detection): Phát hiện các đoạn có tiếng nói và loại bỏ khoảng lặng.
- Speech Separation: Tách riêng từng giọng nói khi có nhiều người nói đồng thời.
- Speech Enhancement: Khử nhiễu môi trường, làm sạch tín hiệu tiếng nói.
- ASR (Automatic Speech Recognition): Nhận dạng giọng nói và chuyển thành văn bản (PhoWhisper).
- QA (Question Answering): Tiếp nhận câu hỏi dạng văn bản và sinh câu trả lời (Qwen / LLM).
- TTS (Text-to-Speech): Chuyển văn bản câu trả lời thành âm thanh nói tự nhiên.

---

## 2. Cấu Trúc Thư Mục

```text
AdvancedML/
├── README.md
├── requirements.txt
├── .gitignore
│
├── configs/
│   ├── pipeline.yaml          # Cấu hình pipeline tổng end-to-end
│   ├── asr.yaml               # Cấu hình riêng cho module ASR
│   ├── enhancement.yaml       # Cấu hình riêng cho Speech Enhancement
│   ├── separation.yaml        # Cấu hình riêng cho Speech Separation
│   ├── qa.yaml                # Cấu hình riêng cho QA / LLM
│   └── tts.yaml               # Cấu hình riêng cho TTS
│
├── src/
│   ├── audio/                 # Xử lý audio dùng chung (AudioData, loader, preprocessing)
│   ├── asr/                   # Module ASR (base, factory, phowhisper, whisper, mock, infer, evaluate)
│   ├── enhancement/           # Module Speech Enhancement (base, factory, deepfilternet, metricgan)
│   ├── separation/            # Module Speech Separation (base, factory, sepformer)
│   ├── qa/                    # Module Question Answering (base, factory, qwen, mock)
│   ├── tts/                   # Module Text-to-Speech (base, factory, vieneu, mock)
│   ├── vad/                   # Module VAD (base, factory, silero, mock)
│   ├── data/                  # Code chuẩn bị và tổng hợp dataset
│   └── pipeline/              # Logic điều phối toàn bộ pipeline (pipeline.py)
│
├── scripts/
│   ├── download_fleurs_samples.py   # Tải audio mẫu tiếng Việt từ tập FLEURS
│   ├── run_pipeline.py              # Chạy pipeline end-to-end từ dòng lệnh
│   ├── train.py                     # Entry point huấn luyện chung
│   ├── evaluate.py                  # Entry point đánh giá chung
│   └── download_data.py             # Script hỗ trợ tải dataset lớn
│
├── app/
│   ├── main.py                # Entry point khởi chạy web app
│   └── ui.py                  # Giao diện Gradio
│
├── notebooks/
│   └── runner.ipynb           # Notebook chạy thực nghiệm trên Google Colab
│
├── checkpoints/               # Lưu trữ weights/checkpoints do quá trình train tạo ra
├── data/                      # Dataset (raw, processed, samples, metadata)
├── outputs/                   # Kết quả sinh ra khi chạy (audio, metrics, logs)
└── tests/                     # Unit tests và integration tests
```

---

## 3. Ý Nghĩa Các Thư Mục và Thành Phần

### configs/
Chứa toàn bộ file cấu hình định dạng YAML. Mục tiêu là thay đổi model, checkpoint, thiết bị tính toán (CPU/CUDA) hoặc siêu tham số mà không cần sửa code Python:
- `pipeline.yaml`: Cấu hình toàn bộ luồng end-to-end. Chọn model nào cho từng module: VAD, Separation, Enhancement, ASR, QA, TTS.
- `asr.yaml`, `enhancement.yaml`, `separation.yaml`, `qa.yaml`, `tts.yaml`: Cấu hình chi tiết riêng cho từng task độc lập.

### src/<module>/
Chứa mã nguồn triển khai thực tế của từng tác vụ ML. Mỗi module tuân theo cấu trúc chuẩn hóa:
- `base.py`: Interface / abstract class chung. Quy định phương thức bắt buộc mà mọi model trong module phải có (ví dụ ASR bắt buộc có `transcribe(audio)`).
- `factory.py`: Đọc tên model từ file YAML và khởi tạo đúng class tương ứng (Factory Pattern).
- `<model>.py`: Wrapper cho từng model cụ thể (như `phowhisper.py`, `deepfilternet.py`, `qwen.py`). Chịu trách nhiệm tải pretrained weights, chuyển đổi tensor và chạy inference.
- `mock.py`: Mô hình giả lập nhẹ dùng để test thông suốt toàn bộ hệ thống mà không cần tải model nặng hay GPU.
- `train.py`: Logic huấn luyện / fine-tuning độc lập cho module.
- `infer.py`: Chạy suy luận độc lập trên file âm thanh mới.
- `evaluate.py`: Đánh giá chất lượng mô hình trên tập dữ liệu kiểm thử (tính toán WER, CER, RTF, PESQ, STOI,...).

### src/audio/
Xử lý dữ liệu âm thanh dùng chung trong toàn bộ hệ thống:
- `types.py`: Định nghĩa kiểu dữ liệu `AudioData` và `SpeechSegment` (mẫu âm thanh, sample rate, số kênh channels, thời lượng duration).
- `loader.py`: Đọc và ghi file WAV sang định dạng `AudioData` chuẩn nội bộ.
- `preprocessing.py`: Các bước tiền xử lý âm thanh chung (resampling, chuyển Stereo sang Mono, chuẩn hóa biên độ).

### src/data/
Chuẩn bị dữ liệu phục vụ huấn luyện và đánh giá:
- `audio_utils.py`: Tiện ích cắt, ghép, chuẩn hóa âm thanh.
- `prepare_enhancement.py`: Tổng hợp cặp âm thanh nhiễu / sạch (noisy/clean pairs) để train enhancement.
- `prepare_separation.py`: Trộn các giọng nói đơn lẻ thành đoạn audio hỗn hợp (mixture) để train separation.

### src/pipeline/pipeline.py
Điều phối toàn bộ luồng hoạt động từ đầu đến cuối. Đọc cấu hình từ `pipeline.yaml` và gọi factory của từng module để kết nối các bước xử lý lại với nhau. Pipeline không chứa logic riêng của bất kỳ mô hình cụ thể nào.

### scripts/
Các entry point dòng lệnh phục vụ việc chạy nhanh từ terminal hoặc Google Colab:
- `download_fleurs_samples.py`: Tải các mẫu audio tiếng Việt chuẩn từ Google FLEURS để kiểm thử nhanh.
- `run_pipeline.py`: Chạy toàn bộ pipeline end-to-end trên một file audio bất kỳ.
- `train.py`, `evaluate.py`: Điều hướng tới script huấn luyện hoặc đánh giá của từng module tương ứng.

### app/
Tầng ứng dụng demo giao diện người dùng:
- `ui.py`: Xây dựng giao diện web bằng Gradio (khung thu âm mic, upload file, bảng kết quả, chuyển đổi giao diện Sáng/Tối).
- `main.py`: Khởi chạy web server, tự động nhận diện môi trường Google Colab để sinh public link chia sẻ.

### checkpoints/
Lưu trữ trọng số (weights) mô hình do quá trình train / fine-tune của dự án sinh ra. Pretrained weights từ Hugging Face được tự động lưu trong cache.

### data/
Chứa dữ liệu sử dụng trong dự án:
- `raw/`: Dataset gốc chưa qua xử lý (VIVOS, MUSAN, FLEURS).
- `processed/`: Dataset đã qua xử lý hoặc tổng hợp cho enhancement và separation.
- `samples/`: Các mẫu âm thanh nhỏ phục vụ kiểm thử nhanh module ASR.
- `metadata/`: Các file CSV/JSON mô tả đường dẫn, transcript, speaker ID, SNR.

### outputs/
Lưu trữ kết quả đầu ra khi chạy thực nghiệm:
- `audio/`: File âm thanh sinh ra sau khi khử nhiễu, tách giọng hoặc tổng hợp giọng nói.
- `metrics/`: Kết quả đánh giá mô hình (file CSV lưu WER, CER, RTF,...).
- `logs/`: Nhật ký thực nghiệm và huấn luyện.

### tests/
Chứa các bài kiểm thử tự động (unit tests, integration tests) đảm bảo code luôn hoạt động đúng khi có thay đổi.

---

## 4. Cài Đặt Môi Trường

Cài đặt các thư viện phụ thuộc:

```bash
pip install -r requirements.txt
```

---

## 5. Tải Dữ Liệu Mẫu Để Kiểm Thử (Sample Audio)

Sử dụng script sau để tải các mẫu âm thanh tiếng Việt chuẩn từ dataset Google FLEURS:

```bash
# Mặc định tải 20 mẫu về data/samples/fleurs_vi/
python scripts/download_fleurs_samples.py

# Hoặc tùy chọn số lượng mẫu cần tải (ví dụ 5 mẫu)
python scripts/download_fleurs_samples.py --count 5
```

Kết quả tải về bao gồm:
- Thư mục audio: `data/samples/fleurs_vi/audio/` (chứa các file `.wav`)
- File metadata: `data/samples/fleurs_vi/metadata.csv` (chứa transcript chuẩn tương ứng để đối chiếu)

---

## 6. Hướng Dẫn Chạy Từng Module Độc Lập

Mỗi module trong thư mục `src/` đều có thể chạy suy luận và đánh giá độc lập:

### Module ASR (Speech-to-Text - PhoWhisper)

1. **Chạy suy luận trên 1 file âm thanh bất kỳ (Inference):**

```bash
python -m src.asr.infer --audio data/samples/fleurs_vi/audio/sample_000.wav
```

Có thể chỉ định file cấu hình khác:
```bash
python -m src.asr.infer --audio path/to/file.wav --config configs/asr.yaml
```

2. **Chay đánh giá chất lượng trên tập dữ liệu (Evaluation):**

Đánh giá tự động độ chính xác bằng các chỉ số WER (Word Error Rate), CER (Character Error Rate) và RTF (Real-Time Factor):

```bash
# Đánh giá trên toàn bộ mẫu trong metadata.csv
python -m src.asr.evaluate

# Đánh giá giới hạn số lượng mẫu (ví dụ 5 mẫu đầu tiên)
python -m src.asr.evaluate --limit 5

# Chỉ định file lưu kết quả chi tiết
python -m src.asr.evaluate --metadata data/samples/fleurs_vi/metadata.csv --output outputs/metrics/asr_evaluation.csv
```

---

### Module TTS (VieNeu)

Chạy riêng để nghe thử giọng và đo thời gian tới audio đầu tiên cùng RTF:

```bash
python -m src.tts.infer --text "Thủ đô của Hàn Quốc là Seoul."
```

File kết quả mặc định: `outputs/audio/tts_sample.wav`. Có thể thêm `--output path/to/file.wav` hoặc đổi giọng trong `configs/tts.yaml`. VieNeu dùng CPU/ONNX để dành GPU cho ASR và QA.

---

## 7. Hướng Dẫn Chạy Pipeline End-to-End

Chạy toàn bộ luồng xử lý từ audio đầu vào đến âm thanh phản hồi:

```bash
# Chạy pipeline với file audio đầu vào
python scripts/run_pipeline.py --input data/samples/fleurs_vi/audio/sample_000.wav

# Chạy và lưu file âm thanh câu trả lời ra đĩa
python scripts/run_pipeline.py --input data/samples/fleurs_vi/audio/sample_000.wav --output outputs/audio/response.wav
```

Pipeline đọc cấu hình từ `configs/pipeline.yaml`. Cấu hình hiện tại dùng PhoWhisper, Qwen và VieNeu; VAD, Separation và Enhancement vẫn dùng mock. Giao diện có chế độ chạy thường và streaming đầu ra: Qwen trả chữ dần, VieNeu tạo audio theo từng phần. Đầu vào vẫn cần thu âm xong rồi mới xử lý.

---

## 8. Hướng Dẫn Chạy Giao Diện Web (Gradio App)

Giao diện web hỗ trợ ghi âm trực tiếp qua microphone, tải lên file WAV, chọn mẫu audio có sẵn, xem transcript nhận dạng, câu trả lời và chuyển đổi giao diện Sáng / Tối. Nút **Chạy pipeline** hiển thị trace từng module; nút **Chạy streaming** phát audio theo từng phần và hiển thị thời gian tới audio đầu tiên (chưa có trace riêng).

Chế độ streaming cần chương trình **FFmpeg** trên `PATH` (kiểm tra bằng `ffmpeg -version`). Chỉ cài thư viện Python trong `requirements.txt` là chưa đủ cho phần phát audio streaming của Gradio.

### Chạy trên máy cục bộ (Local)

```bash
python app/main.py
```

Mở trình duyệt và truy cập:
```text
http://127.0.0.1:7860
```

### Chạy trên Google Colab hoặc chia sẻ link ra ngoài (Public Link)

```bash
python app/main.py --share
```

Hệ thống sẽ tạo đường link public `https://xxxx.gradio.live` để truy cập từ xa. Trên Google Colab, app tự động kích hoạt tính năng này.

---

## 9. Chạy Unit Tests

Kiểm tra tính đúng đắn của toàn bộ pipeline và các thành phần:

```bash
pytest -q
```

---

## 10. Nguyên Tắc Thiết Kế Hệ Thống

1. **Tính độc lập**: Mỗi tác vụ ML là một module độc lập (có thể train, fine-tune, infer và evaluate riêng biệt).
2. **Khử ghép nối (Decoupling)**: Pipeline tổng không gọi trực tiếp model cụ thể nào mà điều phối thông qua interface `base.py` và `factory.py`.
3. **Cấu hình hóa qua YAML**: Mọi thay đổi về lựa chọn model, thiết bị tính toán (CPU/CUDA), checkpoint hoặc tham số đều được điều khiển qua thư mục `configs/`, không can thiệp vào mã nguồn.
4. **Tách biệt tầng ứng dụng**: Giao diện người dùng (`app/`) chỉ tương tác với `pipeline.run(audio)`, không gọi trực tiếp các mô hình ML bên dưới.
