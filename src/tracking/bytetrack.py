"""ByteTrack implementation for CPU-based multi-object tracking."""

from typing import List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from loguru import logger


class KalmanFilter:
    """Simple Kalman Filter for bounding box tracking (constant velocity model)."""
    
    def __init__(self):
        """Initialize Kalman Filter with 8D state and 4D measurement."""
        # State: [x_center, y_center, area, aspect_ratio, vx, vy, va, vr]
        self.state = np.zeros(8)
        self.covariance = np.eye(8)
        
        # Measurement noise
        self.measurement_noise = np.eye(4)
        self.measurement_noise[2:, 2:] *= 10.0
        
        # Process noise
        self.process_noise = np.eye(8)
        self.process_noise[4:, 4:] *= 0.01
        
    def initiate(self, measurement: np.ndarray):
        """
        Initialize state from measurement.
        
        Args:
            measurement: [x1, y1, x2, y2]
        """
        x1, y1, x2, y2 = measurement
        w, h = x2 - x1, y2 - y1
        x_center, y_center = x1 + w / 2, y1 + h / 2
        area = w * h
        aspect_ratio = w / (h + 1e-6)
        
        self.state[:4] = [x_center, y_center, area, aspect_ratio]
        self.state[4:] = 0  # Zero velocity
        
    def predict(self):
        """Predict next state."""
        # State transition matrix (constant velocity)
        F = np.eye(8)
        F[0, 4] = 1  # x += vx
        F[1, 5] = 1  # y += vy
        F[2, 6] = 1  # area += va
        F[3, 7] = 1  # aspect += vr
        
        self.state = F @ self.state
        self.covariance = F @ self.covariance @ F.T + self.process_noise
        
    def update(self, measurement: np.ndarray):
        """
        Update state with measurement.
        
        Args:
            measurement: [x1, y1, x2, y2]
        """
        x1, y1, x2, y2 = measurement
        w, h = x2 - x1, y2 - y1
        x_center, y_center = x1 + w / 2, y1 + h / 2
        area = w * h
        aspect_ratio = w / (h + 1e-6)
        
        z = np.array([x_center, y_center, area, aspect_ratio])
        
        # Measurement matrix
        H = np.eye(4, 8)
        
        # Innovation
        y = z - H @ self.state
        S = H @ self.covariance @ H.T + self.measurement_noise
        K = self.covariance @ H.T @ np.linalg.inv(S)
        
        self.state = self.state + K @ y
        self.covariance = (np.eye(8) - K @ H) @ self.covariance
        
    def get_bbox(self) -> np.ndarray:
        """
        Get current bounding box.
        
        Returns:
            [x1, y1, x2, y2]
        """
        x_center, y_center, area, aspect_ratio = self.state[:4]
        w = np.sqrt(area * aspect_ratio)
        h = area / (w + 1e-6)
        x1 = x_center - w / 2
        y1 = y_center - h / 2
        x2 = x_center + w / 2
        y2 = y_center + h / 2
        return np.array([x1, y1, x2, y2])


class Track:
    """Single track."""
    
    _count = 0
    
    def __init__(self, bbox: np.ndarray, score: float):
        """
        Initialize track.
        
        Args:
            bbox: [x1, y1, x2, y2]
            score: Detection confidence
        """
        self.track_id = Track._count
        Track._count += 1
        
        self.kf = KalmanFilter()
        self.kf.initiate(bbox)
        
        self.score = score
        self.age = 0
        self.time_since_update = 0
        self.hits = 1
        
    def predict(self):
        """Predict next state."""
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        
    def update(self, bbox: np.ndarray, score: float):
        """
        Update with detection.
        
        Args:
            bbox: [x1, y1, x2, y2]
            score: Detection confidence
        """
        self.kf.update(bbox)
        self.score = score
        self.time_since_update = 0
        self.hits += 1
        
    def get_bbox(self) -> np.ndarray:
        """Get current bounding box."""
        return self.kf.get_bbox()


class ByteTracker:
    """ByteTrack tracker for multi-object tracking."""
    
    def __init__(
        self,
        track_thresh: float = 0.5,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        min_box_area: float = 10.0
    ):
        """
        Initialize ByteTracker.
        
        Args:
            track_thresh: Detection threshold for track creation
            track_buffer: Number of frames to keep lost tracks
            match_thresh: IoU threshold for matching
            min_box_area: Minimum box area to consider
        """
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.min_box_area = min_box_area
        
        self.tracked_tracks: List[Track] = []
        self.lost_tracks: List[Track] = []
        self.removed_tracks: List[Track] = []
        
        self.frame_id = 0
        
        logger.info(f"ByteTracker initialized with thresh={track_thresh}, buffer={track_buffer}")
    
    def update(self, detections: np.ndarray) -> np.ndarray:
        """
        Update tracker with detections.
        
        Args:
            detections: (N x 5) array of [x1, y1, x2, y2, score]
        
        Returns:
            (M x 6) array of [x1, y1, x2, y2, track_id, score]
        """
        self.frame_id += 1
        
        # Separate high and low confidence detections
        high_det = detections[detections[:, 4] >= self.track_thresh]
        low_det = detections[detections[:, 4] < self.track_thresh]
        
        # Filter by area
        if len(high_det) > 0:
            areas = (high_det[:, 2] - high_det[:, 0]) * (high_det[:, 3] - high_det[:, 1])
            high_det = high_det[areas >= self.min_box_area]
        
        # Predict all tracks
        for track in self.tracked_tracks:
            track.predict()
        
        # First association with high confidence detections
        matched, unmatched_tracks, unmatched_dets = self._associate(
            self.tracked_tracks, high_det, self.match_thresh
        )
        
        # Update matched tracks
        for track_idx, det_idx in matched:
            track = self.tracked_tracks[track_idx]
            det = high_det[det_idx]
            track.update(det[:4], det[4])
        
        # Second association with low confidence detections
        unmatched_track_objs = [self.tracked_tracks[i] for i in unmatched_tracks]
        matched2, unmatched_tracks2, _ = self._associate(
            unmatched_track_objs, low_det, 0.5
        )
        
        for track_idx, det_idx in matched2:
            track = unmatched_track_objs[track_idx]
            det = low_det[det_idx]
            track.update(det[:4], det[4])
        
        # Handle unmatched tracks
        for i in unmatched_tracks2:
            track = unmatched_track_objs[i]
            if track.time_since_update <= self.track_buffer:
                # Keep as tracked (prediction only)
                pass
            else:
                # Move to lost
                self.lost_tracks.append(track)
                self.tracked_tracks.remove(track)
        
        # Handle unmatched high confidence detections - create new tracks
        for det_idx in unmatched_dets:
            det = high_det[det_idx]
            new_track = Track(det[:4], det[4])
            self.tracked_tracks.append(new_track)
        
        # Clean up old lost tracks
        self.lost_tracks = [t for t in self.lost_tracks if t.time_since_update <= self.track_buffer]
        
        # Prepare output
        output = []
        for track in self.tracked_tracks:
            bbox = track.get_bbox()
            output.append([*bbox, track.track_id, track.score])
        
        if len(output) > 0:
            return np.array(output)
        else:
            return np.empty((0, 6))
    
    @staticmethod
    def _associate(
        tracks: List[Track],
        detections: np.ndarray,
        iou_thresh: float
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Associate tracks with detections using IoU.
        
        Args:
            tracks: List of tracks
            detections: (N x 5) detections [x1, y1, x2, y2, score]
            iou_thresh: IoU threshold
        
        Returns:
            matched pairs, unmatched track indices, unmatched detection indices
        """
        if len(tracks) == 0 or len(detections) == 0:
            return [], list(range(len(tracks))), list(range(len(detections)))
        
        # Compute IoU matrix
        track_boxes = np.array([t.get_bbox() for t in tracks])
        det_boxes = detections[:, :4]
        
        iou_matrix = ByteTracker._compute_iou(track_boxes, det_boxes)
        
        # Hungarian algorithm for assignment
        cost_matrix = 1 - iou_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Filter by threshold
        matched = []
        unmatched_tracks = list(range(len(tracks)))
        unmatched_dets = list(range(len(detections)))
        
        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] >= iou_thresh:
                matched.append((r, c))
                unmatched_tracks.remove(r)
                unmatched_dets.remove(c)
        
        return matched, unmatched_tracks, unmatched_dets
    
    @staticmethod
    def _compute_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        """
        Compute IoU between two sets of boxes.
        
        Args:
            boxes1: (N x 4) boxes
            boxes2: (M x 4) boxes
        
        Returns:
            (N x M) IoU matrix
        """
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        # Broadcast to compute all pairwise intersections
        x1 = np.maximum(boxes1[:, 0][:, None], boxes2[:, 0][None, :])
        y1 = np.maximum(boxes1[:, 1][:, None], boxes2[:, 1][None, :])
        x2 = np.minimum(boxes1[:, 2][:, None], boxes2[:, 2][None, :])
        y2 = np.minimum(boxes1[:, 3][:, None], boxes2[:, 3][None, :])
        
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        union = area1[:, None] + area2[None, :] - intersection
        
        iou = intersection / (union + 1e-6)
        return iou

