"""
Utility functions for visualization
"""
import cv2
import numpy as np
from typing import List, Tuple


class Visualizer:
    """Class for drawing bounding boxes and labels on images"""
    
    def __init__(self, class_colors: dict = None):
        """
        Args:
            class_colors: Dictionary mapping class names to RGB colors
        """
        self.class_colors = class_colors or {
            "with helmet": (0, 255, 128),
            "without helmet": (255, 51, 51),
            "rider": (51, 255, 255),
            "number plate": (224, 102, 255)
        }
    
    def draw_boxes(self, 
                   image: np.ndarray, 
                   boxes: List[List[float]], 
                   labels: List[str],
                   confidences: List[float] = None) -> np.ndarray:
        """
        Draw bounding boxes and labels on image
        
        Args:
            image: Input image (BGR format)
            boxes: List of bounding boxes [x_min, y_min, x_max, y_max]
            labels: List of class labels
            confidences: Optional list of confidence scores
        
        Returns:
            Image with drawn boxes and labels
        """
        img = image.copy()
        
        for i, (box, label) in enumerate(zip(boxes, labels)):
            x_min, y_min, x_max, y_max = [int(coord) for coord in box]
            
            # Get color for this class
            color = self.class_colors.get(label, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
            
            # Prepare label text
            label_text = label
            if confidences and i < len(confidences):
                label_text += f" {confidences[i]:.2f}"
            
            # Draw label background
            (text_width, text_height), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                img,
                (x_min, y_min - text_height - 10),
                (x_min + text_width, y_min),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                img,
                label_text,
                (x_min, y_min - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )
        
        return img

