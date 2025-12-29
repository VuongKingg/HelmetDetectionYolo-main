"""
Script to detect helmets in video
"""
import argparse
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.detector import HelmetDetector


def main():
    parser = argparse.ArgumentParser(description='Detect helmets in video')
    parser.add_argument(
        '--source',
        type=str,
        required=True,
        help='Path to input video file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output video path (default: output/videos/input_name_result.mp4)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to model weights (default: from config)'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=None,
        help='Confidence threshold (default: from config)'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display video while processing'
    )
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = HelmetDetector(model_path=args.model)
    
    if args.conf:
        detector.conf_threshold = args.conf
    
    # Set output path
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


if __name__ == '__main__':
    main()

