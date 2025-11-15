"""Configuration and settings management."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field


class VideoConfig(BaseModel):
    """Video input configuration."""
    source: str
    fps_override: Optional[int] = None
    skip_frames: int = 2


class ModelConfig(BaseModel):
    """Model inference configuration."""
    path: str
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    input_size: List[int] = Field(default=[640, 640])


class TrackingConfig(BaseModel):
    """Tracking configuration."""
    type: str = "bytetrack"
    track_thresh: float = 0.5
    track_buffer: int = 30
    match_thresh: float = 0.8
    min_box_area: float = 10.0
    min_track_hits: int = 3
    duplicate_iou_thresh: float = 0.7


class Zone(BaseModel):
    """Polygon zone definition."""
    name: str
    points: List[List[float]]


class MetricsConfig(BaseModel):
    """Metrics computation configuration."""
    idle_speed_px_s: float = 8.0
    heatmap_grid: List[int] = Field(default=[48, 27])
    min_track_frames: int = 10
    smoothing_window: int = 15  # Number of frames to smooth active/idle status (reduces flickering)


class OutputConfig(BaseModel):
    """Output configuration."""
    dir: str = "data/outputs"
    save_annotated: bool = True
    save_per_frame: bool = True
    blur_faces: bool = False


class Config(BaseModel):
    """Main configuration."""
    video: VideoConfig
    model: ModelConfig
    tracking: TrackingConfig
    zones: List[Zone]
    metrics: MetricsConfig
    output: OutputConfig

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        with open(path, "w") as f:
            yaml.safe_dump(self.model_dump(), f, default_flow_style=False)

