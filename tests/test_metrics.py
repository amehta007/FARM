"""Tests for metrics module."""

import numpy as np
import pytest


def test_metrics_tracker_initialization():
    """Test metrics tracker initialization."""
    from src.analytics.metrics import MetricsTracker
    
    tracker = MetricsTracker(idle_speed_threshold=10.0, fps=30.0)
    
    assert tracker.idle_speed_threshold == 10.0
    assert tracker.fps == 30.0
    assert len(tracker.tracks) == 0


def test_track_metrics():
    """Test per-track metrics computation."""
    from src.analytics.metrics import TrackMetrics
    
    metrics = TrackMetrics(track_id=1, idle_speed_threshold=10.0)
    
    # Simulate stationary object (idle)
    bbox1 = np.array([100, 100, 200, 200])
    metrics.update(bbox1, timestamp=0.0, frame_idx=0, fps=30.0)
    
    bbox2 = np.array([101, 100, 201, 200])  # Slight movement
    metrics.update(bbox2, timestamp=0.033, frame_idx=1, fps=30.0)
    
    assert metrics.total_frames == 2
    
    summary = metrics.get_summary(fps=30.0)
    assert summary["track_id"] == 1
    assert summary["total_frames"] == 2


def test_zone_containment():
    """Test zone point containment."""
    from src.analytics.zones import Zone
    
    zone = Zone(name="TestZone", points=[(0, 0), (100, 0), (100, 100), (0, 100)])
    
    # Point inside
    assert zone.contains_point((50, 50)) == True
    
    # Point outside
    assert zone.contains_point((150, 150)) == False


def test_zone_manager():
    """Test zone manager."""
    from src.analytics.zones import ZoneManager
    
    zones_config = [
        {"name": "Zone1", "points": [[0, 0], [100, 0], [100, 100], [0, 100]]},
        {"name": "Zone2", "points": [[100, 100], [200, 100], [200, 200], [100, 200]]}
    ]
    
    manager = ZoneManager(zones_config)
    
    assert len(manager.zones) == 2
    
    # Update with point in Zone1
    manager.update(track_id=1, center=(50, 50), dt=1.0)
    
    time_in_zone1 = manager.get_zone_time(1, "Zone1")
    assert time_in_zone1 == 1.0


def test_heatmap():
    """Test heatmap generation."""
    from src.analytics.heatmap import OccupancyHeatmap
    
    heatmap = OccupancyHeatmap(frame_shape=(480, 640), grid_size=(32, 24))
    
    # Add some points
    heatmap.update((320, 240), dt=1.0)  # Center
    heatmap.update((320, 240), dt=1.0)  # Same spot
    
    # Get heatmap image
    heatmap_img = heatmap.get_heatmap_image()
    
    assert heatmap_img.shape == (480, 640, 3)

