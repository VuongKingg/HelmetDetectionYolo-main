# Hướng dẫn sử dụng Web App

## 🚀 Khởi chạy ứng dụng

### Cách 1: Sử dụng Streamlit CLI

```bash
streamlit run streamlit_app.py
```

### Cách 2: Sử dụng script (Windows)

```bash
run_app.bat
```

### Cách 3: Sử dụng script (Linux/Mac)

```bash
bash run_app.sh
```

### Cách 4: Sử dụng Python module

```bash
python -m streamlit run streamlit_app.py
```

Sau khi chạy, ứng dụng sẽ tự động mở trình duyệt tại địa chỉ:
**http://localhost:8501**

Nếu không tự động mở, hãy copy địa chỉ này vào trình duyệt.

## 📱 Giao diện ứng dụng

### Tab 1: Image Detection 📸

1. **Upload ảnh**: Click vào "Choose an image..." và chọn file ảnh
2. **Xem ảnh gốc**: Ảnh sẽ hiển thị ngay sau khi upload
3. **Nhấn nút "🔍 Detect"**: Bắt đầu phát hiện đối tượng
4. **Xem kết quả**:
   - Ảnh đã được vẽ bounding boxes
   - Thống kê số lượng detection
   - Bảng chi tiết từng detection
   - Biểu đồ phân bố các class

### Tab 2: Video Detection 🎥

1. **Upload video**: Chọn file video (mp4, avi, mov, mkv)
2. **Xem video gốc**: Video sẽ được hiển thị trong trình phát
3. **Nhấn "🔍 Process Video"**: Bắt đầu xử lý video
4. **Chờ xử lý**: Video sẽ được xử lý frame by frame (có thể mất thời gian)
5. **Download kết quả**: Sau khi xử lý xong, có thể download video đã detect

### Tab 3: Webcam 📹

1. **Nhấn "🎥 Start Webcam"**: Bắt đầu kết nối với webcam
2. **Cửa sổ mới sẽ mở**: Hiển thị video từ webcam với detection real-time
3. **Nhấn 'q'**: Để dừng webcam

### Tab 4: About ℹ️

Thông tin về ứng dụng, model, và hướng dẫn sử dụng.

## ⚙️ Cài đặt (Sidebar)

### Detection Settings

- **Confidence Threshold**: Ngưỡng tin cậy (0.0 - 1.0)

  - Giá trị cao hơn = ít detection hơn nhưng chính xác hơn
  - Giá trị thấp hơn = nhiều detection hơn nhưng có thể có false positives
  - Mặc định: 0.25

- **IoU Threshold**: Ngưỡng IoU cho Non-Maximum Suppression (0.0 - 1.0)
  - Điều chỉnh cách loại bỏ các box trùng lặp
  - Mặc định: 0.7

### Model Information

- Hiển thị thông tin model và các class có thể detect

### Model Performance

- Hiển thị các metrics: mAP50, mAP50-95, Precision, Recall

## 💡 Tips sử dụng

1. **Ảnh chất lượng tốt**:

   - Sử dụng ảnh có độ phân giải rõ ràng
   - Đảm bảo đủ ánh sáng
   - Tránh ảnh mờ, tối

2. **Điều chỉnh threshold**:

   - Nếu có quá nhiều false positives → tăng confidence threshold
   - Nếu thiếu detection → giảm confidence threshold

3. **Xử lý video**:

   - Video ngắn sẽ xử lý nhanh hơn
   - Video dài có thể mất vài phút
   - Đảm bảo có đủ dung lượng disk cho video output

4. **Webcam**:
   - Đảm bảo webcam không bị sử dụng bởi ứng dụng khác
   - Đủ ánh sáng cho detection tốt hơn
   - Giữ khoảng cách phù hợp với camera

## 🐛 Troubleshooting

### Lỗi: "Model not found"

- Kiểm tra file `app/models/best.pt` có tồn tại không
- Nếu không có, copy từ `results/runs/detect/train/weights/best.pt`

### Lỗi: "Streamlit not found"

```bash
pip install streamlit
```

### Lỗi: "Cannot access camera"

- Đảm bảo webcam không bị sử dụng bởi ứng dụng khác
- Kiểm tra quyền truy cập camera trên hệ thống
- Thử thay đổi `camera_id` trong code (0, 1, 2...)

### Web app chạy chậm

- Giảm độ phân giải ảnh/video trước khi upload
- Giảm số lượng frames xử lý cho video
- Sử dụng GPU nếu có (tự động nếu có CUDA)

## 🌐 Deploy lên web

### Streamlit Cloud (Miễn phí)

1. Push code lên GitHub
2. Đăng ký tại [streamlit.io](https://streamlit.io/cloud)
3. Connect với repository
4. Deploy tự động

### Heroku

1. Tạo file `Procfile`:

```
web: streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0
```

2. Deploy như ứng dụng Streamlit thông thường

### Docker

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py"]
```

## 📞 Hỗ trợ

Nếu gặp vấn đề, vui lòng kiểm tra:

1. Đã cài đặt đầy đủ dependencies chưa
2. Model file có tồn tại không
3. Logs trong terminal để xem lỗi cụ thể
