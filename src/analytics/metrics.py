"""Per-track metrics computation (activity, idle time, etc.)."""

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.utils.geometry import bbox_center, euclidean_distance


class TrackMetrics:
    """Tracks metrics for a single tracked object."""
    
    def __init__(self, track_id: int, idle_speed_threshold: float):
        """
        Initialize track metrics.
        
        Args:
            track_id: Unique track identifier
            idle_speed_threshold: Speed threshold (px/s) below which object is idle
        """
        self.track_id = track_id
        self.idle_speed_threshold = idle_speed_threshold
        
        self.total_frames = 0
        self.active_frames = 0
        self.idle_frames = 0
        
        self.last_center: Tuple[float, float] = None
        self.last_time: float = None
        
        # Frame-by-frame history
        self.history: List[Dict] = []
        
    def update(
        self,
        bbox: np.ndarray,
        timestamp: float,
        frame_idx: int,
        fps: float
    ):
        """
        Update metrics with new detection.
        
        Args:
            bbox: [x1, y1, x2, y2]
            timestamp: Current timestamp in seconds
            frame_idx: Frame index
            fps: Video FPS
        """
        center = bbox_center(bbox)
        
        # Compute instantaneous speed
        speed = 0.0
        is_active = False
        
        if self.last_center is not None and self.last_time is not None:
            dt = timestamp - self.last_time
            if dt > 0:
                distance = euclidean_distance(center, self.last_center)
                speed = distance / dt  # pixels per second
                is_active = speed >= self.idle_speed_threshold
        
        # Update counters
        self.total_frames += 1
        if is_active:
            self.active_frames += 1
        else:
            self.idle_frames += 1
        
        # Store history
        self.history.append({
            "frame_idx": frame_idx,
            "timestamp": timestamp,
            "center_x": center[0],
            "center_y": center[1],
            "speed_px_s": speed,
            "is_active": is_active
        })
        
        self.last_center = center
        self.last_time = timestamp
    
    def get_summary(self, fps: float) -> Dict:
        """
        Get summary metrics.
        
        Args:
            fps: Video FPS for time conversion
        
        Returns:
            Dictionary of summary metrics
        """
        presence_time = self.total_frames / fps if fps > 0 else 0
        active_time = self.active_frames / fps if fps > 0 else 0
        idle_time = self.idle_frames / fps if fps > 0 else 0
        
        return {
            "track_id": self.track_id,
            "total_frames": self.total_frames,
            "presence_time_s": presence_time,
            "active_time_s": active_time,
            "idle_time_s": idle_time,
            "active_ratio": active_time / presence_time if presence_time > 0 else 0,
        }
    
    def get_history_df(self) -> pd.DataFrame:
        """
        Get per-frame history as DataFrame.
        
        Returns:
            DataFrame with per-frame metrics
        """
        if not self.history:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.history)
        df["track_id"] = self.track_id
        return df


class MetricsTracker:
    """Tracks metrics for all objects across the video."""
    
    def __init__(self, idle_speed_threshold: float, fps: float):
        """
        Initialize metrics tracker.
        
        Args:
            idle_speed_threshold: Speed threshold (px/s) for idle detection
            fps: Video FPS
        """
        self.idle_speed_threshold = idle_speed_threshold
        self.fps = fps
        
        self.tracks: Dict[int, TrackMetrics] = {}
        
    def update(self, track_id: int, bbox: np.ndarray, timestamp: float, frame_idx: int):
        """
        Update metrics for a track.
        
        Args:
            track_id: Track identifier
            bbox: [x1, y1, x2, y2]
            timestamp: Current timestamp
            frame_idx: Frame index
        """
        if track_id not in self.tracks:
            self.tracks[track_id] = TrackMetrics(track_id, self.idle_speed_threshold)
        
        self.tracks[track_id].update(bbox, timestamp, frame_idx, self.fps)
    
    def get_all_summaries(self) -> pd.DataFrame:
        """
        Get summary metrics for all tracks.
        
        Returns:
            DataFrame with per-track summaries
        """
        summaries = [track.get_summary(self.fps) for track in self.tracks.values()]
        if not summaries:
            return pd.DataFrame()
        return pd.DataFrame(summaries)
    
    def get_all_histories(self) -> pd.DataFrame:
        """
        Get full per-frame history for all tracks.
        
        Returns:
            DataFrame with all frame-level data
        """
        histories = [track.get_history_df() for track in self.tracks.values()]
        if not histories:
            return pd.DataFrame()
        return pd.concat(histories, ignore_index=True)
    
    def get_track_summary(self, track_id: int) -> Dict:
        """
        Get summary for a specific track.
        
        Args:
            track_id: Track identifier
        
        Returns:
            Summary dictionary
        """
        if track_id not in self.tracks:
            return {}
        return self.tracks[track_id].get_summary(self.fps)

