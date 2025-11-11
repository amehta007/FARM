"""Timing and profiling utilities."""

import time
from contextlib import contextmanager
from typing import Optional

from loguru import logger


@contextmanager
def timer(name: str, log_level: str = "DEBUG"):
    """
    Context manager for timing code blocks.
    
    Args:
        name: Name of the timed operation
        log_level: Logging level (DEBUG, INFO, etc.)
    
    Usage:
        with timer("processing"):
            process_frame()
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.log(log_level, f"{name} took {elapsed:.4f}s")


class FPSCounter:
    """Simple FPS counter with exponential moving average."""
    
    def __init__(self, alpha: float = 0.1):
        """
        Initialize FPS counter.
        
        Args:
            alpha: Smoothing factor for EMA (0-1)
        """
        self.alpha = alpha
        self.fps: Optional[float] = None
        self.last_time: Optional[float] = None
    
    def tick(self) -> float:
        """
        Update FPS counter with current frame.
        
        Returns:
            Current FPS estimate
        """
        current_time = time.perf_counter()
        
        if self.last_time is not None:
            frame_time = current_time - self.last_time
            instant_fps = 1.0 / frame_time if frame_time > 0 else 0.0
            
            if self.fps is None:
                self.fps = instant_fps
            else:
                self.fps = self.alpha * instant_fps + (1 - self.alpha) * self.fps
        
        self.last_time = current_time
        return self.fps if self.fps is not None else 0.0

