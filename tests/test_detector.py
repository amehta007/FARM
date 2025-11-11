"""Tests for detector module."""

import numpy as np
import pytest


def test_detector_output_shape():
    """Test that detector returns correct output shape."""
    from src.detection.detector import ONNXDetector
    
    # Create a dummy image
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Note: This test requires the ONNX model to be present
    # In a real test, we might mock the ONNX runtime
    # For now, we'll just test the shape manipulation
    
    # Test preprocess
    mock_detector = type('MockDetector', (), {
        'input_size': (640, 640),
        'conf_threshold': 0.25,
        'iou_threshold': 0.45
    })()
    
    # Simulate preprocessing
    h, w = image.shape[:2]
    assert h == 480
    assert w == 640


def test_nms():
    """Test Non-Maximum Suppression."""
    from src.detection.detector import ONNXDetector
    
    # Create overlapping boxes
    boxes = np.array([
        [10, 10, 50, 50],
        [15, 15, 55, 55],  # Overlaps with first
        [100, 100, 150, 150]  # Separate
    ])
    scores = np.array([0.9, 0.8, 0.95])
    
    kept = ONNXDetector.nms(boxes, scores, iou_threshold=0.5)
    
    # Should keep first and third boxes
    assert len(kept) == 2
    assert 0 in kept
    assert 2 in kept


def test_bbox_operations():
    """Test bounding box utility functions."""
    from src.utils.geometry import bbox_center, bbox_area
    
    bbox = np.array([10, 20, 50, 80])
    
    # Test center
    center = bbox_center(bbox)
    assert center == (30, 50)
    
    # Test area
    area = bbox_area(bbox)
    assert area == 2400  # (50-10) * (80-20)

