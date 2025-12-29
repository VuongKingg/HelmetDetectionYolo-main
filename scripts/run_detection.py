"""
Script tổng hợp để chạy detection với nhiều tùy chọn
"""
import argparse
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.detector import HelmetDetector


def main():
    parser = argparse.ArgumentParser(
        description='Helmet Detection - Detect riders with/without helmets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Detect on single image
  python scripts/run_detection.py image --source input/images/test.jpg
  
  # Detect on all images in directory
  python scripts/run_detection.py image --source input/images/
  
  # Detect on video
  python scripts/run_detection.py video --source input/videos/test.mp4
  
  # Detect using webcam
  python scripts/run_detection.py webcam
        """
    )
    
    subparsers = parser.add_subparsers(dest='mode', help='Detection mode')
    
    # Image detection parser
    img_parser = subparsers.add_parser('image', help='Detect on image(s)')
    img_parser.add_argument('--source', type=str, required=True, help='Image file or directory')
    img_parser.add_argument('--output', type=str, default=None, help='Output directory')
    img_parser.add_argument('--model', type=str, default=None, help='Model path')
    img_parser.add_argument('--conf', type=float, default=None, help='Confidence threshold')
    img_parser.add_argument('--show', action='store_true', help='Show results')
    
    # Video detection parser
    vid_parser = subparsers.add_parser('video', help='Detect on video')
    vid_parser.add_argument('--source', type=str, required=True, help='Video file')
    vid_parser.add_argument('--output', type=str, default=None, help='Output video path')
    vid_parser.add_argument('--model', type=str, default=None, help='Model path')
    vid_parser.add_argument('--conf', type=float, default=None, help='Confidence threshold')
    vid_parser.add_argument('--show', action='store_true', help='Show video while processing')
    
    # Webcam detection parser
    webcam_parser = subparsers.add_parser('webcam', help='Detect using webcam')
    webcam_parser.add_argument('--camera', type=int, default=0, help='Camera ID')
    webcam_parser.add_argument('--model', type=str, default=None, help='Model path')
    webcam_parser.add_argument('--conf', type=float, default=None, help='Confidence threshold')
    
    args = parser.parse_args()
    
    if not args.mode:
        parser.print_help()
        sys.exit(1)
    
    # Initialize detector
    try:
        detector = HelmetDetector(model_path=args.model)
        if args.conf:
            detector.conf_threshold = args.conf
    except Exception as e:
        print(f"Error initializing detector: {e}")
        sys.exit(1)
    
    # Execute based on mode
    if args.mode == 'image':
        source_path = Path(args.source)
        output_dir = Path(args.output) if args.output else Path('output/images')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if source_path.is_file():
            print(f"Processing image: {source_path}")
            result = detector.predict_image(
                image_path=source_path,
                save_path=output_dir / f"{source_path.stem}_result{source_path.suffix}",
                show=args.show
            )
            print(f"\nDetected {result['count']} objects:")
            for box, label, conf in zip(result['boxes'], result['labels'], result['confidences']):
                print(f"  - {label}: {conf:.2f}")
        
        elif source_path.is_dir():
            print(f"Processing directory: {source_path}")
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            image_paths = [p for p in source_path.iterdir() if p.suffix.lower() in image_extensions]
            
            if not image_paths:
                print(f"No images found in {source_path}")
                sys.exit(1)
            
            print(f"Found {len(image_paths)} images")
            results = detector.predict_batch(image_paths, save_dir=output_dir)
            
            total_detections = sum(r['count'] for r in results)
            print(f"\nTotal detections: {total_detections}")
            print(f"Average detections per image: {total_detections / len(image_paths):.2f}")
        else:
            print(f"Error: {args.source} is not a valid file or directory")
            sys.exit(1)
    
    elif args.mode == 'video':
        if not Path(args.source).exists():
            print(f"Error: Video file not found: {args.source}")
            sys.exit(1)
        
        if args.output:
            output_path = Path(args.output)
        else:
            source_path = Path(args.source)
            output_dir = Path('output/videos')
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{source_path.stem}_result{source_path.suffix}"
        
        print(f"Processing video: {args.source}")
        print(f"Output will be saved to: {output_path}")
        
        stats = detector.predict_video(
            video_path=args.source,
            output_path=output_path,
            show=args.show
        )
        
        print("\nProcessing complete!")
        print(f"Frames processed: {stats['frames']}")
        print(f"Total detections: {stats['total_detections']}")
        print(f"Average detections per frame: {stats['avg_detections_per_frame']:.2f}")
    
    elif args.mode == 'webcam':
        print(f"Starting webcam detection (Camera ID: {args.camera})")
        print("Press 'q' to quit")
        
        try:
            detector.predict_webcam(camera_id=args.camera, show=True)
        except KeyboardInterrupt:
            print("\nStopped by user")
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)


if __name__ == '__main__':
    main()

