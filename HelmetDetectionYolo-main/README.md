# Helmet Detection Application

Ứng dụng phát hiện người điều khiển xe máy có/không có mũ bảo hiểm sử dụng YOLOv8.

## Cấu trúc thư mục

```
.
├── app/                    # Code chính của ứng dụng
│   ├── __init__.py
│   ├── detector.py        # Class chính cho detection
│   ├── models/            # Model weights
│   │   └── best.pt
│   ├── config/            # File cấu hình
│   │   ├── config.yaml
│   │   └── data.yaml
│   └── utils/             # Utilities
│       ├── __init__.py
│       ├── config_loader.py
│       └── visualizer.py
├── input/                 # Thư mục chứa dữ liệu đầu vào
│   ├── images/
│   └── videos/
├── output/                # Thư mục chứa kết quả
│   ├── images/
│   └── videos/
├── scripts/               # Scripts để chạy ứng dụng
│   ├── detect_image.py
│   ├── detect_video.py
│   └── detect_webcam.py
├── requirements.txt       # Dependencies
└── README.md             # File này
```

## Cài đặt

1. **Cài đặt dependencies:**

```bash
pip install -r requirements.txt
```

2. **Chạy Web App (Streamlit):**

```bash
streamlit run streamlit_app.py
```

Sau đó mở trình duyệt tại địa chỉ: `http://localhost:8501`

Hoặc sử dụng script:

- Windows: `run_app.bat`
- Linux/Mac: `bash run_app.sh`

3. **Kiểm tra model:**
   Model đã được copy vào `app/models/best.pt`. Nếu chưa có, hãy đảm bảo file `results/runs/detect/train/weights/best.pt` tồn tại.

## Sử dụng

### 1. Detect trên ảnh đơn

```bash
python scripts/detect_image.py --source input/images/your_image.jpg --output output/images
```

Hoặc detect trên nhiều ảnh trong thư mục:

```bash
python scripts/detect_image.py --source input/images/ --output output/images
```

**Tùy chọn:**

- `--source`: Đường dẫn đến ảnh hoặc thư mục chứa ảnh (bắt buộc)
- `--output`: Thư mục lưu kết quả (mặc định: `output/images`)
- `--model`: Đường dẫn đến model weights (mặc định: từ config)
- `--conf`: Ngưỡng confidence (mặc định: 0.25)
- `--show`: Hiển thị kết quả

### 2. Detect trên video

```bash
python scripts/detect_video.py --source input/videos/your_video.mp4 --output output/videos/result.mp4
```

**Tùy chọn:**

- `--source`: Đường dẫn đến file video (bắt buộc)
- `--output`: Đường dẫn lưu video kết quả (mặc định: `output/videos/input_name_result.mp4`)
- `--model`: Đường dẫn đến model weights
- `--conf`: Ngưỡng confidence
- `--show`: Hiển thị video khi xử lý

### 3. Detect trên webcam

```bash
python scripts/detect_webcam.py --camera 0
```

**Tùy chọn:**

- `--camera`: ID của camera (mặc định: 0)
- `--model`: Đường dẫn đến model weights
- `--conf`: Ngưỡng confidence

Nhấn `q` để thoát.

### 4. Sử dụng trong code Python

```python
from app.detector import HelmetDetector

# Khởi tạo detector
detector = HelmetDetector()

# Detect trên ảnh
result = detector.predict_image('input/images/test.jpg', save_path='output/images/result.jpg')

# In kết quả
print(f"Phát hiện {result['count']} đối tượng:")
for box, label, conf in zip(result['boxes'], result['labels'], result['confidences']):
    print(f"  - {label}: {conf:.2f}")

# Detect trên video
stats = detector.predict_video('input/videos/test.mp4', output_path='output/videos/result.mp4')
print(f"Xử lý {stats['frames']} frames")

# Detect trên webcam
detector.predict_webcam(camera_id=0, show=True)
```

## Classes được phát hiện

Model có thể phát hiện 4 loại đối tượng:

1. **with helmet** - Người điều khiển có mũ bảo hiểm
2. **without helmet** - Người điều khiển không có mũ bảo hiểm
3. **rider** - Người điều khiển xe
4. **number plate** - Biển số xe

## Cấu hình

File cấu hình nằm tại `app/config/config.yaml`. Bạn có thể chỉnh sửa:

- `model.conf_threshold`: Ngưỡng confidence (mặc định: 0.25)
- `model.iou_threshold`: Ngưỡng IoU cho NMS (mặc định: 0.7)
- `classes.colors`: Màu sắc cho từng class
- `paths`: Đường dẫn mặc định cho input/output

## Model Performance

Model đã được train trên 50 epochs và đạt:

- **mAP50**: 0.9394 (94%)
- **mAP50-95**: 0.75922 (76%)
- **Precision**: 0.91223
- **Recall**: 0.89643

## Yêu cầu hệ thống

- Python >= 3.8
- CUDA-capable GPU (khuyến nghị) hoặc CPU
- RAM >= 8GB (khuyến nghị)

## License

MIT
