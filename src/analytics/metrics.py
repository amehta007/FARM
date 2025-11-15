"""Per-track metrics computation (activity, idle time, etc.)."""

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.utils.geometry import bbox_center, euclidean_distance


class TrackMetrics:
    """Tracks metrics for a single tracked object."""
    
    def __init__(self, track_id: int, idle_speed_threshold: float, smoothing_window: int = 10):
        """
        Initialize track metrics.
        
        Args:
            track_id: Unique track identifier
            idle_speed_threshold: Speed threshold (px/s) below which object is idle
            smoothing_window: Number of recent frames to use for smoothing active/idle status
        """
        self.track_id = track_id
        self.idle_speed_threshold = idle_speed_threshold
        self.smoothing_window = smoothing_window
        
        self.total_frames = 0
        self.active_frames = 0
        self.idle_frames = 0
        
        self.last_center: Tuple[float, float] = None
        self.last_time: float = None
        
        # Frame-by-frame history
        self.history: List[Dict] = []
        
        # Smoothing: track recent speeds and statuses
        self.recent_speeds: List[float] = []
        self.recent_active_statuses: List[bool] = []
        self.smoothed_is_active: bool = False  # Current smoothed status
        
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
        is_active_instant = False
        
        if self.last_center is not None and self.last_time is not None:
            dt = timestamp - self.last_time
            if dt > 0:
                distance = euclidean_distance(center, self.last_center)
                speed = distance / dt  # pixels per second
                is_active_instant = speed >= self.idle_speed_threshold
        
        # Smooth the active/idle status to prevent flickering
        # Use hysteresis: different thresholds for switching states
        self.recent_speeds.append(speed)
        self.recent_active_statuses.append(is_active_instant)
        
        # Keep only recent window
        if len(self.recent_speeds) > self.smoothing_window:
            self.recent_speeds.pop(0)
            self.recent_active_statuses.pop(0)
        
        # Calculate smoothed speed (average of recent speeds)
        if len(self.recent_speeds) > 0:
            smoothed_speed = sum(self.recent_speeds) / len(self.recent_speeds)
        else:
            smoothed_speed = speed
        
        # Use hysteresis thresholds to prevent flickering:
        # - To become active: need speed above threshold
        # - To become idle: need speed well below threshold (more lenient)
        idle_threshold = self.idle_speed_threshold * 0.7  # 70% of threshold to become idle
        active_threshold = self.idle_speed_threshold * 1.2  # 120% of threshold to become active
        
        # Determine smoothed status with hysteresis
        if self.smoothed_is_active:
            # Currently active: switch to idle only if speed is consistently low
            if smoothed_speed < idle_threshold:
                # Check if majority of recent frames are idle
                idle_count = sum(1 for s in self.recent_active_statuses if not s)
                if idle_count >= len(self.recent_active_statuses) * 0.6:  # 60% must be idle
                    self.smoothed_is_active = False
        else:
            # Currently idle: switch to active only if speed is consistently high
            if smoothed_speed > active_threshold:
                # Check if majority of recent frames are active
                active_count = sum(1 for s in self.recent_active_statuses if s)
                if active_count >= len(self.recent_active_statuses) * 0.6:  # 60% must be active
                    self.smoothed_is_active = True
        
        # Use smoothed status for metrics
        is_active = self.smoothed_is_active
        
        # Update counters
        self.total_frames += 1
        if is_active:
            self.active_frames += 1
        else:
            self.idle_frames += 1
        
        # Store history (store both instantaneous and smoothed)
        self.history.append({
            "frame_idx": frame_idx,
            "timestamp": timestamp,
            "center_x": center[0],
            "center_y": center[1],
            "speed_px_s": speed,
            "smoothed_speed_px_s": smoothed_speed,
            "is_active": is_active,  # Smoothed status for display
            "is_active_instant": is_active_instant  # Instantaneous status for reference
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
    
    def __init__(self, idle_speed_threshold: float, fps: float, smoothing_window: int = 10):
        """
        Initialize metrics tracker.
        
        Args:
            idle_speed_threshold: Speed threshold (px/s) for idle detection
            fps: Video FPS
            smoothing_window: Number of frames to use for smoothing active/idle status
        """
        self.idle_speed_threshold = idle_speed_threshold
        self.fps = fps
        self.smoothing_window = smoothing_window
        
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
            self.tracks[track_id] = TrackMetrics(track_id, self.idle_speed_threshold, self.smoothing_window)
        
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
    
    def is_track_active(self, track_id: int) -> bool:
        """
        Get current active/idle status for a track.
        
        Args:
            track_id: Track identifier
        
        Returns:
            True if track is currently active, False if idle
        """
        if track_id not in self.tracks:
            return False
        
        track_metrics = self.tracks[track_id]
        # Check the most recent status from history
        if not track_metrics.history:
            return False
        
        # Return the most recent active status
        return track_metrics.history[-1].get("is_active", False)

