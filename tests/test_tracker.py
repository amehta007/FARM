"""Tests for tracking module."""

import numpy as np
import pytest


def test_tracker_initialization():
    """Test tracker initialization."""
    from src.tracking.bytetrack import ByteTracker
    
    tracker = ByteTracker(
        track_thresh=0.5,
        track_buffer=30,
        match_thresh=0.8
    )
    
    assert tracker.track_thresh == 0.5
    assert tracker.track_buffer == 30
    assert len(tracker.tracked_tracks) == 0


def test_tracker_single_detection():
    """Test tracker with single detection."""
    from src.tracking.bytetrack import ByteTracker
    
    tracker = ByteTracker()
    
    # Single detection
    detections = np.array([
        [100, 100, 200, 200, 0.9]
    ])
    
    tracks = tracker.update(detections)
    
    assert len(tracks) == 1
    assert tracks[0, 4] == 0  # Track ID should be 0


def test_tracker_multiple_frames():
    """Test tracker across multiple frames."""
    from src.tracking.bytetrack import ByteTracker
    
    tracker = ByteTracker()
    
    # Frame 1
    detections1 = np.array([
        [100, 100, 200, 200, 0.9]
    ])
    tracks1 = tracker.update(detections1)
    track_id = int(tracks1[0, 4])
    
    # Frame 2 - same object moved slightly
    detections2 = np.array([
        [105, 105, 205, 205, 0.9]
    ])
    tracks2 = tracker.update(detections2)
    
    # Should maintain same track ID
    assert len(tracks2) == 1
    assert int(tracks2[0, 4]) == track_id


def test_iou_computation():
    """Test IoU computation."""
    from src.tracking.bytetrack import ByteTracker
    
    boxes1 = np.array([
        [0, 0, 10, 10]
    ])
    boxes2 = np.array([
        [0, 0, 10, 10],  # Perfect overlap
        [5, 5, 15, 15],  # Partial overlap
        [20, 20, 30, 30]  # No overlap
    ])
    
    iou = ByteTracker._compute_iou(boxes1, boxes2)
    
    assert iou.shape == (1, 3)
    assert iou[0, 0] == pytest.approx(1.0)  # Perfect overlap
    assert 0 < iou[0, 1] < 1  # Partial overlap
    assert iou[0, 2] == pytest.approx(0.0)  # No overlap

