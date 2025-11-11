"""Geometry utility functions."""

import numpy as np
from shapely.geometry import Point, Polygon as ShapelyPolygon
from typing import List, Tuple


def point_in_polygon(point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
    """
    Check if a point is inside a polygon.
    
    Args:
        point: (x, y) coordinates
        polygon: List of (x, y) vertices
    
    Returns:
        True if point is inside polygon
    """
    poly = ShapelyPolygon(polygon)
    pt = Point(point)
    return poly.contains(pt)


def bbox_center(bbox: np.ndarray) -> Tuple[float, float]:
    """
    Compute center of bounding box.
    
    Args:
        bbox: [x1, y1, x2, y2]
    
    Returns:
        (cx, cy) center coordinates
    """
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def bbox_area(bbox: np.ndarray) -> float:
    """
    Compute area of bounding box.
    
    Args:
        bbox: [x1, y1, x2, y2]
    
    Returns:
        Area in pixels
    """
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    Compute Euclidean distance between two points.
    
    Args:
        p1: First point (x, y)
        p2: Second point (x, y)
    
    Returns:
        Distance in pixels
    """
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

