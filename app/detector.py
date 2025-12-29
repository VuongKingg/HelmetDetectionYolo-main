"""
Main detector class for helmet detection
"""
from ultralytics import YOLO
from pathlib import Path
from typing import Union, List, Tuple
import numpy as np
from .utils.config_loader import load_config


class HelmetDetector:
    """YOLOv8 model wrapper for helmet detection"""
    
    def __init__(self, model_path: str = None, config_path: str = "app/config/config.yaml"):
        """
        Initialize detector
        
        Args:
            model_path: Path to model weights (.pt file)
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        
        # Load model
        model_path = model_path or self.config['model']['path']
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.model = YOLO(model_path)
        self.conf_threshold = self.config['model']['conf_threshold']
        self.iou_threshold = self.config['model']['iou_threshold']
        
        # Class mapping
        self.class_names = self.config['classes']['names']
        self.id2class = {i: name for i, name in enumerate(self.class_names)}
    
    def predict_image(self, 
                     image_path: Union[str, Path],
                     save_path: Union[str, Path] = None,
                     show: bool = False) -> dict:
        """
        Predict on single image
        
        Args:
            image_path: Path to input image
            save_path: Optional path to save result
            show: Whether to display result
        
        Returns:
            Dictionary with predictions
        """
        results = self.model.predict(
            source=str(image_path),
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            save=save_path is not None,
            project=str(Path(save_path).parent) if save_path else None,
            name=Path(save_path).stem if save_path else None,
            show=show
        )
        
        return self._parse_results(results[0])
    
    def predict_batch(self, 
                     image_paths: List[Union[str, Path]],
                     save_dir: Union[str, Path] = None) -> List[dict]:
        """
        Predict on multiple images
        
        Args:
            image_paths: List of image paths
            save_dir: Optional directory to save results
        
        Returns:
            List of prediction dictionaries
        """
        results = self.model.predict(
            source=[str(p) for p in image_paths],
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            save=save_dir is not None,
            project=str(save_dir) if save_dir else None
        )
        
        return [self._parse_results(r) for r in results]
    
    def predict_video(self,
                     video_path: Union[str, Path],
                     output_path: Union[str, Path] = None,
                     show: bool = False) -> dict:
        """
        Predict on video
        
        Args:
            video_path: Path to input video
            output_path: Optional path to save output video
            show: Whether to display video
        
        Returns:
            Dictionary with video statistics
        """
        results = self.model.predict(
            source=str(video_path),
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            save=output_path is not None,
            project=str(Path(output_path).parent) if output_path else None,
            name=Path(output_path).stem if output_path else None,
            show=show,
            stream=True
        )
        
        # Process all frames
        frame_count = 0
        total_detections = 0
        
        for result in results:
            frame_count += 1
            parsed = self._parse_results(result)
            total_detections += len(parsed['boxes'])
        
        return {
            'frames': frame_count,
            'total_detections': total_detections,
            'avg_detections_per_frame': total_detections / frame_count if frame_count > 0 else 0
        }
    
    def predict_webcam(self, camera_id: int = 0, show: bool = True):
        """
        Predict on webcam stream
        
        Args:
            camera_id: Camera device ID
            show: Whether to display stream
        """
        results = self.model.predict(
            source=camera_id,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            show=show,
            stream=True
        )
        
        # Lặp qua stream để giữ webcam chạy
        # YOLOv8 tự động xử lý hiển thị khi show=True
        try:
            for _ in results:
                pass
        except KeyboardInterrupt:
            print("\nWebcam stream stopped")
    
    def _parse_results(self, result) -> dict:
        """
        Parse YOLO results to structured format
        
        Args:
            result: YOLO result object
        
        Returns:
            Dictionary with boxes, labels, confidences
        """
        boxes = result.boxes.xyxy.cpu().numpy().tolist() if result.boxes is not None else []
        labels = [self.id2class[int(cls)] for cls in result.boxes.cls.cpu().numpy()] if result.boxes is not None else []
        confidences = result.boxes.conf.cpu().numpy().tolist() if result.boxes is not None else []
        
        return {
            'boxes': boxes,
            'labels': labels,
            'confidences': confidences,
            'count': len(boxes)
        }

