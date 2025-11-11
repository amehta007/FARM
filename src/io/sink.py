"""Output sinks for saving results (video, CSV, parquet)."""

from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from loguru import logger


class VideoWriter:
    """Video writer for annotated output."""
    
    def __init__(
        self,
        output_path: str,
        frame_size: tuple,
        fps: float,
        codec: str = "mp4v"
    ):
        """
        Initialize video writer.
        
        Args:
            output_path: Output video file path
            frame_size: (width, height)
            fps: Output FPS
            codec: FourCC codec code
        """
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            fps,
            frame_size
        )
        
        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to open video writer: {output_path}")
        
        logger.info(f"Video writer initialized: {output_path}")
    
    def write(self, frame: np.ndarray):
        """Write a frame."""
        self.writer.write(frame)
    
    def release(self):
        """Release video writer."""
        self.writer.release()
        logger.info(f"Video saved to {self.output_path}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()


class MetricsSink:
    """Sink for saving metrics to CSV/parquet."""
    
    def __init__(self, output_dir: str, run_id: Optional[str] = None):
        """
        Initialize metrics sink.
        
        Args:
            output_dir: Output directory
            run_id: Unique run identifier (timestamp if None)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = run_id
        
        logger.info(f"Metrics sink initialized: run_id={run_id}")
    
    def save_summary(self, df: pd.DataFrame, name: str = "summary"):
        """
        Save summary metrics to CSV.
        
        Args:
            df: Summary DataFrame
            name: File name prefix
        """
        output_path = self.output_dir / f"{name}_{self.run_id}.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Summary saved to {output_path}")
    
    def save_history(self, df: pd.DataFrame, name: str = "history"):
        """
        Save per-frame history to parquet.
        
        Args:
            df: History DataFrame
            name: File name prefix
        """
        output_path = self.output_dir / f"{name}_{self.run_id}.parquet"
        df.to_parquet(output_path, index=False)
        logger.info(f"History saved to {output_path}")
    
    def save_heatmap(self, heatmap_image: np.ndarray, name: str = "heatmap"):
        """
        Save heatmap image.
        
        Args:
            heatmap_image: Heatmap visualization
            name: File name prefix
        """
        output_path = self.output_dir / f"{name}_{self.run_id}.png"
        cv2.imwrite(str(output_path), heatmap_image)
        logger.info(f"Heatmap saved to {output_path}")
    
    def save_config(self, config_dict: dict, name: str = "config"):
        """
        Save configuration.
        
        Args:
            config_dict: Configuration dictionary
            name: File name prefix
        """
        import yaml
        output_path = self.output_dir / f"{name}_{self.run_id}.yaml"
        with open(output_path, "w") as f:
            yaml.safe_dump(config_dict, f)
        logger.info(f"Config saved to {output_path}")

