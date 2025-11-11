"""Drawing utilities for visualization."""

from typing import List, Optional, Tuple

import cv2
import numpy as np


# Color palette for tracks (BGR format)
COLORS = [
    (255, 0, 0),      # Blue
    (0, 255, 0),      # Green
    (0, 0, 255),      # Red
    (255, 255, 0),    # Cyan
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Yellow
    (128, 0, 128),    # Purple
    (0, 128, 128),    # Olive
    (128, 128, 0),    # Teal
    (192, 192, 192),  # Silver
]


def get_color(track_id: int) -> Tuple[int, int, int]:
    """
    Get consistent color for track ID.
    
    Args:
        track_id: Track identifier
    
    Returns:
        BGR color tuple
    """
    return COLORS[track_id % len(COLORS)]


def draw_bbox(
    frame: np.ndarray,
    bbox: np.ndarray,
    track_id: int,
    score: float,
    label: Optional[str] = None,
    thickness: int = 2
) -> np.ndarray:
    """
    Draw bounding box with label.
    
    Args:
        frame: Frame to draw on
        bbox: [x1, y1, x2, y2]
        track_id: Track ID
        score: Confidence score
        label: Optional label text
        thickness: Line thickness
    
    Returns:
        Frame with drawing
    """
    x1, y1, x2, y2 = map(int, bbox[:4])
    color = get_color(track_id)
    
    # Draw box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    
    # Prepare label
    if label is None:
        label = f"ID:{track_id} {score:.2f}"
    
    # Draw label background
    (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - label_h - baseline - 5), (x1 + label_w + 5, y1), color, -1)
    
    # Draw label text
    cv2.putText(frame, label, (x1 + 2, y1 - baseline - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame


def draw_zones(
    frame: np.ndarray,
    zones: List,
    alpha: float = 0.3
) -> np.ndarray:
    """
    Draw polygon zones on frame.
    
    Args:
        frame: Frame to draw on
        zones: List of Zone objects
        alpha: Zone fill transparency
    
    Returns:
        Frame with zones
    """
    overlay = frame.copy()
    
    for i, zone in enumerate(zones):
        color = COLORS[i % len(COLORS)]
        vertices = zone.get_vertices()
        
        # Draw filled polygon
        cv2.fillPoly(overlay, [vertices], color)
        
        # Draw border
        cv2.polylines(frame, [vertices], isClosed=True, color=color, thickness=2)
        
        # Draw zone name
        centroid = vertices.mean(axis=0).astype(int)
        cv2.putText(frame, zone.name, tuple(centroid), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Blend overlay
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    
    return frame


def draw_trail(
    frame: np.ndarray,
    centers: List[Tuple[int, int]],
    track_id: int,
    max_length: int = 30,
    thickness: int = 2
) -> np.ndarray:
    """
    Draw motion trail.
    
    Args:
        frame: Frame to draw on
        centers: List of (x, y) center points
        track_id: Track ID for color
        max_length: Maximum trail length
        thickness: Line thickness
    
    Returns:
        Frame with trail
    """
    if len(centers) < 2:
        return frame
    
    color = get_color(track_id)
    centers = centers[-max_length:]
    
    for i in range(1, len(centers)):
        pt1 = tuple(map(int, centers[i - 1]))
        pt2 = tuple(map(int, centers[i]))
        
        # Fade older points
        alpha = i / len(centers)
        line_color = tuple(int(c * alpha) for c in color)
        
        cv2.line(frame, pt1, pt2, line_color, thickness)
    
    return frame


def draw_info_panel(
    frame: np.ndarray,
    info: dict,
    position: str = "top-left"
) -> np.ndarray:
    """
    Draw information panel on frame.
    
    Args:
        frame: Frame to draw on
        info: Dictionary of key-value pairs to display
        position: Panel position ("top-left", "top-right", etc.)
    
    Returns:
        Frame with info panel
    """
    h, w = frame.shape[:2]
    
    # Panel dimensions
    line_height = 25
    panel_height = (len(info) + 1) * line_height
    panel_width = 300
    
    # Position
    if position == "top-left":
        x, y = 10, 10
    elif position == "top-right":
        x, y = w - panel_width - 10, 10
    elif position == "bottom-left":
        x, y = 10, h - panel_height - 10
    else:  # bottom-right
        x, y = w - panel_width - 10, h - panel_height - 10
    
    # Draw semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + panel_width, y + panel_height), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    
    # Draw text
    for i, (key, value) in enumerate(info.items()):
        text = f"{key}: {value}"
        cv2.putText(frame, text, (x + 10, y + (i + 1) * line_height), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame


def blur_bbox(frame: np.ndarray, bbox: np.ndarray, kernel_size: int = 31) -> np.ndarray:
    """
    Blur region inside bounding box (for privacy).
    
    Args:
        frame: Frame to modify
        bbox: [x1, y1, x2, y2]
        kernel_size: Gaussian blur kernel size (must be odd)
    
    Returns:
        Frame with blurred region
    """
    x1, y1, x2, y2 = map(int, bbox[:4])
    
    # Ensure within frame bounds
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    if x2 <= x1 or y2 <= y1:
        return frame
    
    # Extract and blur region
    roi = frame[y1:y2, x1:x2]
    blurred = cv2.GaussianBlur(roi, (kernel_size, kernel_size), 0)
    frame[y1:y2, x1:x2] = blurred
    
    return frame

