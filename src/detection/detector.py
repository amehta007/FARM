"""ONNX Runtime-based YOLOv8 detector for person detection."""

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import onnxruntime as ort
from loguru import logger


class ONNXDetector:
    """YOLOv8 detector using ONNX Runtime (CPU-optimized)."""
    
    # COCO class names - we only care about 'person' (class 0)
    PERSON_CLASS_ID = 0
    
    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        input_size: Tuple[int, int] = (640, 640)
    ):
        """
        Initialize ONNX detector.
        
        Args:
            model_path: Path to ONNX model file
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            input_size: Model input size (width, height)
        """
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size
        
        # Initialize ONNX Runtime session with CPU execution provider
        logger.info(f"Loading ONNX model from {self.model_path}")
        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider"]
        )
        
        # Get model input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        
        logger.success(f"ONNX model loaded successfully")
    
    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Preprocess image for inference.
        
        Args:
            image: Input image (BGR)
        
        Returns:
            Preprocessed image tensor, scale factor, and padding
        """
        # Resize while maintaining aspect ratio
        h, w = image.shape[:2]
        scale = min(self.input_size[0] / w, self.input_size[1] / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Pad to input size
        pad_w = self.input_size[0] - new_w
        pad_h = self.input_size[1] - new_h
        padded = cv2.copyMakeBorder(
            resized, 0, pad_h, 0, pad_w,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        
        # Convert to RGB and normalize
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        
        # HWC to CHW
        tensor = np.transpose(tensor, (2, 0, 1))
        
        # Add batch dimension
        tensor = np.expand_dims(tensor, axis=0)
        
        return tensor, scale, (pad_w, pad_h)
    
    def postprocess(
        self,
        outputs: List[np.ndarray],
        scale: float,
        original_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Postprocess model outputs to get detections.
        
        Args:
            outputs: Model output tensors
            scale: Scale factor from preprocessing
            original_shape: Original image shape (h, w)
        
        Returns:
            Detections array (N x 5): [x1, y1, x2, y2, confidence]
        """
        # YOLOv8 output shape: (1, 84, 8400) or similar
        # Format: [x_center, y_center, width, height, class_0_conf, ..., class_79_conf]
        output = outputs[0][0]  # Remove batch dimension
        
        # Transpose to (8400, 84)
        output = output.T
        
        # Extract boxes and scores
        boxes = output[:, :4]  # x_center, y_center, w, h
        scores = output[:, 4:]  # class confidences
        
        # Get person class confidence
        person_scores = scores[:, self.PERSON_CLASS_ID]
        
        # Filter by confidence threshold
        mask = person_scores >= self.conf_threshold
        boxes = boxes[mask]
        person_scores = person_scores[mask]
        
        if len(boxes) == 0:
            return np.empty((0, 5))
        
        # Convert from center format to corner format
        x_center, y_center, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = x_center - w / 2
        y1 = y_center - h / 2
        x2 = x_center + w / 2
        y2 = y_center + h / 2
        
        # Scale back to original image size
        x1 /= scale
        y1 /= scale
        x2 /= scale
        y2 /= scale
        
        # Clip to image boundaries
        h_orig, w_orig = original_shape
        x1 = np.clip(x1, 0, w_orig)
        y1 = np.clip(y1, 0, h_orig)
        x2 = np.clip(x2, 0, w_orig)
        y2 = np.clip(y2, 0, h_orig)
        
        # Apply NMS
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
        indices = self.nms(boxes_xyxy, person_scores, self.iou_threshold)
        
        # Combine boxes and scores
        detections = np.column_stack([
            boxes_xyxy[indices],
            person_scores[indices]
        ])
        
        return detections
    
    @staticmethod
    def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
        """
        Non-Maximum Suppression.
        
        Args:
            boxes: Boxes (N x 4) in [x1, y1, x2, y2] format
            scores: Confidence scores (N,)
            iou_threshold: IoU threshold
        
        Returns:
            Indices of kept boxes
        """
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            intersection = w * h
            
            iou = intersection / (areas[i] + areas[order[1:]] - intersection + 1e-6)
            
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        
        return np.array(keep)
    
    def detect(self, image: np.ndarray) -> np.ndarray:
        """
        Run detection on an image.
        
        Args:
            image: Input image (BGR)
        
        Returns:
            Detections (N x 5): [x1, y1, x2, y2, confidence]
        """
        original_shape = image.shape[:2]
        
        # Preprocess
        tensor, scale, padding = self.preprocess(image)
        
        # Inference
        outputs = self.session.run(self.output_names, {self.input_name: tensor})
        
        # Postprocess
        detections = self.postprocess(outputs, scale, original_shape)
        
        return detections

