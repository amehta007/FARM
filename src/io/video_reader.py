"""Video reader supporting files and webcam."""

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from loguru import logger


class VideoReader:
    """Unified video reader for files and webcam."""
    
    def __init__(self, source: str, fps_override: Optional[int] = None):
        """
        Initialize video reader.
        
        Args:
            source: Video file path or "webcam:N" where N is device index
            fps_override: Override FPS (for processing rate control)
        """
        self.source = source
        self.fps_override = fps_override
        
        # Parse source
        if source.lower().startswith("webcam:"):
            device_id = int(source.split(":")[1])
            self.is_webcam = True
            self.cap = cv2.VideoCapture(device_id)
            logger.info(f"Opened webcam device {device_id}")
        else:
            self.is_webcam = False
            source_path = Path(source)
            if not source_path.exists():
                raise FileNotFoundError(f"Video file not found: {source}")
            self.cap = cv2.VideoCapture(str(source_path))
            logger.info(f"Opened video file: {source}")
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {source}")
        
        # Get video properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = fps_override if fps_override else self.cap.get(cv2.CAP_PROP_FPS)
        
        if self.is_webcam and self.fps == 0:
            self.fps = 30.0  # Default for webcam
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not self.is_webcam else -1
        
        logger.info(f"Video properties: {self.width}x{self.height} @ {self.fps:.2f} FPS")
        if not self.is_webcam:
            logger.info(f"Total frames: {self.total_frames}")
        
        self.frame_idx = 0
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read next frame.
        
        Returns:
            (success, frame) tuple
        """
        ret, frame = self.cap.read()
        if ret:
            self.frame_idx += 1
        return ret, frame
    
    def get_fps(self) -> float:
        """Get video FPS."""
        return self.fps
    
    def get_frame_shape(self) -> Tuple[int, int]:
        """Get frame shape (height, width)."""
        return (self.height, self.width)
    
    def get_current_frame_idx(self) -> int:
        """Get current frame index."""
        return self.frame_idx
    
    def release(self):
        """Release video capture."""
        self.cap.release()
        logger.info("Video source released")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()

