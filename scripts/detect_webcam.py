"""
Script to detect helmets using webcam
"""
import argparse
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.detector import HelmetDetector


def main():
    parser = argparse.ArgumentParser(description='Detect helmets using webcam')
    parser.add_argument(
        '--camera',
        type=int,
        default=0,
        help='Camera device ID (default: 0)'
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
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = HelmetDetector(model_path=args.model)
    
    if args.conf:
        detector.conf_threshold = args.conf
    
    print(f"Starting webcam detection (Camera ID: {args.camera})")
    print("Press 'q' to quit")
    
    try:
        detector.predict_webcam(camera_id=args.camera, show=True)
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main()

