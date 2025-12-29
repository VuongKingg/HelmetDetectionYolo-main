# Quick Start Guide

## Cài đặt nhanh

```bash
# 1. Cài đặt dependencies
pip install -r requirements.txt
```

## Chạy Web App (Khuyến nghị)

```bash
# Chạy ứng dụng web
streamlit run streamlit_app.py
```

Sau đó mở trình duyệt tại: `http://localhost:8501`

Ứng dụng web có giao diện thân thiện với các tính năng:

- ✅ Upload và detect ảnh
- ✅ Upload và xử lý video
- ✅ Real-time webcam detection
- ✅ Điều chỉnh confidence threshold
- ✅ Hiển thị thống kê và kết quả chi tiết

## Sử dụng qua Command Line

### 1. Detect trên ảnh

```bash
# Ảnh đơn
python scripts/detect_image.py --source input/images/your_image.jpg

# Nhiều ảnh trong thư mục
python scripts/detect_image.py --source input/images/

# Với script tổng hợp
python scripts/run_detection.py image --source input/images/test.jpg
```

### 2. Detect trên video

```bash
python scripts/detect_video.py --source input/videos/your_video.mp4

# Hoặc
python scripts/run_detection.py video --source input/videos/test.mp4
```

### 3. Detect trên webcam

```bash
python scripts/detect_webcam.py

# Hoặc
python scripts/run_detection.py webcam
```

### 4. Sử dụng trong code Python

```python
from app.detector import HelmetDetector

detector = HelmetDetector()

# Detect ảnh
result = detector.predict_image('input/images/test.jpg')
print(f"Phát hiện {result['count']} đối tượng")

# Detect video
stats = detector.predict_video('input/videos/test.mp4')
print(f"Xử lý {stats['frames']} frames")
```

## Cấu trúc thư mục

```
app/
  ├── detector.py          # Class chính
  ├── models/
  │   └── best.pt          # Model đã train
  ├── config/
  │   └── config.yaml      # Cấu hình
  └── utils/               # Utilities

scripts/
  ├── detect_image.py      # Script detect ảnh
  ├── detect_video.py      # Script detect video
  ├── detect_webcam.py     # Script detect webcam
  └── run_detection.py     # Script tổng hợp

input/
  ├── images/              # Đặt ảnh vào đây
  └── videos/              # Đặt video vào đây

output/
  ├── images/              # Kết quả ảnh
  └── videos/              # Kết quả video
```

## Lưu ý

- Đảm bảo đã có model tại `app/models/best.pt`
- Đặt file input vào thư mục `input/` tương ứng
- Kết quả sẽ được lưu vào thư mục `output/`
