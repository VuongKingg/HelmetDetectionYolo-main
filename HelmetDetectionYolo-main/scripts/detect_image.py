"""
Script to detect helmets on single image or batch of images
"""
import argparse
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.detector import HelmetDetector


def main():
    parser = argparse.ArgumentParser(description='Detect helmets in image(s)')
    parser.add_argument(
        '--source',
        type=str,
        required=True,
        help='Path to image file or directory containing images'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory for results (default: output/images)'
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
        help='Display results'
    )
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = HelmetDetector(model_path=args.model)
    
    if args.conf:
        detector.conf_threshold = args.conf
    
    # Set output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path('output/images')
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    source_path = Path(args.source)
    
    # Process single image or directory
    if source_path.is_file():
        print(f"Processing image: {source_path}")
        result = detector.predict_image(
            image_path=source_path,
            save_path=output_dir / f"{source_path.stem}_result{source_path.suffix}",
            show=args.show
        )
        print(f"Detected {result['count']} objects")
        for box, label, conf in zip(result['boxes'], result['labels'], result['confidences']):
            print(f"  - {label}: {conf:.2f}")
    
    elif source_path.is_dir():
        print(f"Processing directory: {source_path}")
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_paths = [
            p for p in source_path.iterdir()
            if p.suffix.lower() in image_extensions
        ]
        
        print(f"Found {len(image_paths)} images")
        results = detector.predict_batch(image_paths, save_dir=output_dir)
        
        total_detections = sum(r['count'] for r in results)
        print(f"\nTotal detections: {total_detections}")
        print(f"Average detections per image: {total_detections / len(image_paths):.2f}")
    
    else:
        print(f"Error: {args.source} is not a valid file or directory")
        sys.exit(1)


if __name__ == '__main__':
    main()

