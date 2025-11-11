"""Heatmap generation for spatial occupancy visualization."""

from typing import Tuple

import cv2
import numpy as np
from loguru import logger


class OccupancyHeatmap:
    """Generates heatmap showing where workers spend time in the frame."""
    
    def __init__(self, frame_shape: Tuple[int, int], grid_size: Tuple[int, int]):
        """
        Initialize heatmap.
        
        Args:
            frame_shape: (height, width) of video frame
            grid_size: (grid_width, grid_height) number of cells
        """
        self.frame_h, self.frame_w = frame_shape
        self.grid_w, self.grid_h = grid_size
        
        # Cell dimensions
        self.cell_w = self.frame_w / self.grid_w
        self.cell_h = self.frame_h / self.grid_h
        
        # Accumulator grid
        self.grid = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        
        logger.info(f"Heatmap initialized: frame={frame_shape}, grid={grid_size}, cell_size=({self.cell_w:.1f}, {self.cell_h:.1f})")
    
    def update(self, center: Tuple[float, float], dt: float = 1.0):
        """
        Update heatmap with object center position.
        
        Args:
            center: (x, y) center coordinates
            dt: Time weight (e.g., frame duration in seconds)
        """
        x, y = center
        
        # Convert to grid coordinates
        grid_x = int(x / self.cell_w)
        grid_y = int(y / self.cell_h)
        
        # Clamp to grid bounds
        grid_x = max(0, min(grid_x, self.grid_w - 1))
        grid_y = max(0, min(grid_y, self.grid_h - 1))
        
        # Accumulate time
        self.grid[grid_y, grid_x] += dt
    
    def get_heatmap_image(self, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """
        Generate heatmap visualization.
        
        Args:
            colormap: OpenCV colormap to use
        
        Returns:
            Heatmap image (H x W x 3) BGR
        """
        # Normalize grid to 0-255
        if self.grid.max() > 0:
            normalized = (self.grid / self.grid.max() * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(self.grid, dtype=np.uint8)
        
        # Resize to frame size
        resized = cv2.resize(normalized, (self.frame_w, self.frame_h), interpolation=cv2.INTER_LINEAR)
        
        # Apply colormap
        heatmap = cv2.applyColorMap(resized, colormap)
        
        return heatmap
    
    def get_normalized_grid(self) -> np.ndarray:
        """
        Get normalized grid values.
        
        Returns:
            Grid normalized to [0, 1]
        """
        if self.grid.max() > 0:
            return self.grid / self.grid.max()
        else:
            return self.grid.copy()
    
    def overlay_on_frame(self, frame: np.ndarray, alpha: float = 0.4) -> np.ndarray:
        """
        Overlay heatmap on frame.
        
        Args:
            frame: Original frame (BGR)
            alpha: Heatmap transparency (0=invisible, 1=opaque)
        
        Returns:
            Frame with heatmap overlay
        """
        heatmap = self.get_heatmap_image()
        blended = cv2.addWeighted(frame, 1 - alpha, heatmap, alpha, 0)
        return blended

