"""
Ví dụ sử dụng HelmetDetector
"""
from app.detector import HelmetDetector
from pathlib import Path

def main():
    # Khởi tạo detector
    print("Đang khởi tạo detector...")
    detector = HelmetDetector()
    
    # Ví dụ 1: Detect trên ảnh
    print("\n=== Ví dụ 1: Detect trên ảnh ===")
    image_path = Path("input/images")
    if image_path.exists() and any(image_path.iterdir()):
        # Lấy ảnh đầu tiên
        image_file = next(image_path.glob("*.jpg"), None) or next(image_path.glob("*.png"), None)
        if image_file:
            print(f"Đang xử lý ảnh: {image_file}")
            result = detector.predict_image(
                image_path=image_file,
                save_path=Path("output/images") / f"{image_file.stem}_result{image_file.suffix}"
            )
            print(f"Phát hiện {result['count']} đối tượng:")
            for box, label, conf in zip(result['boxes'], result['labels'], result['confidences']):
                print(f"  - {label}: {conf:.2f}")
        else:
            print("Không tìm thấy ảnh trong input/images/")
    else:
        print("Thư mục input/images/ trống hoặc không tồn tại")
    
    # Ví dụ 2: Detect trên video
    print("\n=== Ví dụ 2: Detect trên video ===")
    video_path = Path("input/videos")
    if video_path.exists() and any(video_path.iterdir()):
        video_file = next(video_path.glob("*.mp4"), None) or next(video_path.glob("*.avi"), None)
        if video_file:
            print(f"Đang xử lý video: {video_file}")
            stats = detector.predict_video(
                video_path=video_file,
                output_path=Path("output/videos") / f"{video_file.stem}_result{video_file.suffix}"
            )
            print(f"Đã xử lý {stats['frames']} frames")
            print(f"Tổng số detection: {stats['total_detections']}")
            print(f"Trung bình detection mỗi frame: {stats['avg_detections_per_frame']:.2f}")
        else:
            print("Không tìm thấy video trong input/videos/")
    else:
        print("Thư mục input/videos/ trống hoặc không tồn tại")
    
    # Ví dụ 3: Sử dụng webcam (uncomment để chạy)
    # print("\n=== Ví dụ 3: Detect trên webcam ===")
    # print("Nhấn 'q' để thoát")
    # detector.predict_webcam(camera_id=0, show=True)

if __name__ == "__main__":
    main()

