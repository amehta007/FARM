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

    def _ensure_valid_state(self) -> None:
        """Clamp area and aspect ratio to valid numeric ranges."""
        area = self.state[2]
        aspect_ratio = self.state[3]
        if not np.isfinite(area) or area <= 0:
            area = 1e-6
        if not np.isfinite(aspect_ratio) or aspect_ratio <= 0:
            aspect_ratio = 1.0
        self.state[2] = area
        self.state[3] = aspect_ratio
        
    def initiate(self, measurement: np.ndarray):
        """
        Initialize state from measurement.
        
        Args:
            measurement: [x1, y1, x2, y2]
        """
        x1, y1, x2, y2 = measurement
        w, h = x2 - x1, y2 - y1
        w = max(float(w), 1e-6)
        h = max(float(h), 1e-6)
        x_center, y_center = x1 + w / 2, y1 + h / 2
        area = max(w * h, 1e-6)
        aspect_ratio = max(w / h, 1e-6)
        
        self.state[:4] = [x_center, y_center, area, aspect_ratio]
        self.state[4:] = 0  # Zero velocity
        self._ensure_valid_state()
        
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
        self._ensure_valid_state()
        
    def update(self, measurement: np.ndarray):
        """
        Update state with measurement.
        
        Args:
            measurement: [x1, y1, x2, y2]
        """
        x1, y1, x2, y2 = measurement
        w, h = x2 - x1, y2 - y1
        w = max(float(w), 1e-6)
        h = max(float(h), 1e-6)
        x_center, y_center = x1 + w / 2, y1 + h / 2
        area = max(w * h, 1e-6)
        aspect_ratio = max(w / h, 1e-6)
        
        z = np.array([x_center, y_center, area, aspect_ratio])
        
        # Measurement matrix
        H = np.eye(4, 8)
        
        # Innovation
        y = z - H @ self.state
        S = H @ self.covariance @ H.T + self.measurement_noise
        K = self.covariance @ H.T @ np.linalg.inv(S)
        
        self.state = self.state + K @ y
        self.covariance = (np.eye(8) - K @ H) @ self.covariance
        self._ensure_valid_state()
        
    def get_bbox(self) -> np.ndarray:
        """
        Get current bounding box.
        
        Returns:
            [x1, y1, x2, y2]
        """
        x_center, y_center, area, aspect_ratio = self.state[:4]
        area = float(area)
        if not np.isfinite(area) or area <= 0:
            area = 1e-6
        aspect_ratio = float(aspect_ratio)
        if not np.isfinite(aspect_ratio) or aspect_ratio <= 0:
            aspect_ratio = 1.0
        aspect_ratio = float(np.clip(aspect_ratio, 1e-3, 1e3))
        prod = area * aspect_ratio
        if not np.isfinite(prod) or prod <= 0:
            prod = 1e-6
        w = float(np.sqrt(prod))
        w = max(w, 1e-3)
        h = area / w
        h = max(h, 1e-3)
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
        min_box_area: float = 10.0,
        min_track_hits: int = 3,
        duplicate_iou_thresh: float = 0.7
    ):
        """
        Initialize ByteTracker.
        
        Args:
            track_thresh: Detection threshold for track creation
            track_buffer: Number of frames to keep lost tracks
            match_thresh: IoU threshold for matching
            min_box_area: Minimum box area to consider
            min_track_hits: Minimum number of detections before track is shown
            duplicate_iou_thresh: IoU threshold for detecting duplicate tracks
        """
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.min_box_area = min_box_area
        self.min_track_hits = min_track_hits
        self.duplicate_iou_thresh = duplicate_iou_thresh
        
        self.tracked_tracks: List[Track] = []
        self.lost_tracks: List[Track] = []
        self.removed_tracks: List[Track] = []
        
        self.frame_id = 0
        
        logger.info(f"ByteTracker initialized with thresh={track_thresh}, buffer={track_buffer}, min_hits={min_track_hits}")
    
    def update(self, detections: np.ndarray) -> np.ndarray:
        """
        Update tracker with detections.
        
        Args:
            detections: (N x 5) array of [x1, y1, x2, y2, score]
        
        Returns:
            (M x 6) array of [x1, y1, x2, y2, track_id, score]
        """
        self.frame_id += 1
        if detections is None or len(detections) == 0:
            detections = np.empty((0, 5), dtype=np.float32)
        else:
            detections = np.asarray(detections, dtype=np.float32)
            widths = detections[:, 2] - detections[:, 0]
            heights = detections[:, 3] - detections[:, 1]
            valid_mask = (widths > 1e-3) & (heights > 1e-3)
            if not np.all(valid_mask):
                logger.debug(
                    "Filtering %d invalid detections (non-positive size)",
                    int((~valid_mask).sum()),
                )
                detections = detections[valid_mask]
            if detections.size == 0:
                detections = np.empty((0, 5), dtype=np.float32)
        
        # Separate high and low confidence detections
        high_det = detections[detections[:, 4] >= self.track_thresh]
        low_det = detections[detections[:, 4] < self.track_thresh]
        
        # Filter by area
        if len(high_det) > 0:
            areas = (high_det[:, 2] - high_det[:, 0]) * (high_det[:, 3] - high_det[:, 1])
            high_det = high_det[areas >= self.min_box_area]
        if len(low_det) > 0:
            areas = (low_det[:, 2] - low_det[:, 0]) * (low_det[:, 3] - low_det[:, 1])
            low_det = low_det[areas >= self.min_box_area]
        
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
        
        # Filter duplicate tracks (tracks that are too close together)
        self._remove_duplicate_tracks()
        
        # Prepare output - only include tracks with sufficient hits
        output = []
        invalid_tracks = []
        for track in self.tracked_tracks:
            bbox = track.get_bbox()
            if not self._is_valid_bbox(bbox):
                invalid_tracks.append(track)
                continue
            
            # Only include tracks that have been seen enough times
            if track.hits < self.min_track_hits:
                continue
                
            output.append([*bbox, track.track_id, track.score])
        
        for track in invalid_tracks:
            logger.debug("Removing track %s due to invalid bbox.", track.track_id)
            if track in self.tracked_tracks:
                self.tracked_tracks.remove(track)
        
        if len(output) > 0:
            return np.array(output, dtype=np.float32)
        else:
            return np.empty((0, 6))
    
    @staticmethod
    def _is_valid_bbox(bbox: np.ndarray, min_size: float = 1e-3) -> bool:
        """Check if bounding box is valid and finite."""
        if bbox is None or bbox.shape[0] != 4:
            return False
        if np.any(~np.isfinite(bbox)):
            return False
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        return width > min_size and height > min_size
    
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
        track_boxes = np.array([t.get_bbox() for t in tracks], dtype=np.float32)
        det_boxes = detections[:, :4].astype(np.float32, copy=False)
        track_boxes = ByteTracker._sanitize_boxes(track_boxes)
        det_boxes = ByteTracker._sanitize_boxes(det_boxes)
        
        iou_matrix = ByteTracker._compute_iou(track_boxes, det_boxes)
        
        # Hungarian algorithm for assignment
        cost_matrix = 1 - iou_matrix
        np.nan_to_num(cost_matrix, nan=1.0, posinf=1.0, neginf=1.0, copy=False)
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
        widths1 = np.maximum(boxes1[:, 2] - boxes1[:, 0], 0.0)
        heights1 = np.maximum(boxes1[:, 3] - boxes1[:, 1], 0.0)
        widths2 = np.maximum(boxes2[:, 2] - boxes2[:, 0], 0.0)
        heights2 = np.maximum(boxes2[:, 3] - boxes2[:, 1], 0.0)

        area1 = widths1 * heights1
        area2 = widths2 * heights2
        
        # Broadcast to compute all pairwise intersections
        x1 = np.maximum(boxes1[:, 0][:, None], boxes2[:, 0][None, :])
        y1 = np.maximum(boxes1[:, 1][:, None], boxes2[:, 1][None, :])
        x2 = np.minimum(boxes1[:, 2][:, None], boxes2[:, 2][None, :])
        y2 = np.minimum(boxes1[:, 3][:, None], boxes2[:, 3][None, :])
        
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        union = area1[:, None] + area2[None, :] - intersection
        union = np.maximum(union, 1e-6)
        
        iou = intersection / union
        return np.nan_to_num(iou, nan=0.0, posinf=0.0, neginf=0.0)

    def _remove_duplicate_tracks(self):
        """Remove duplicate tracks that are too close together (same person detected twice)."""
        if len(self.tracked_tracks) < 2:
            return
        
        # Get all track boxes and centers
        track_boxes = []
        track_centers = []
        track_indices = []
        for idx, track in enumerate(self.tracked_tracks):
            bbox = track.get_bbox()
            if self._is_valid_bbox(bbox):
                track_boxes.append(bbox)
                # Calculate center
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                track_centers.append((center_x, center_y))
                track_indices.append(idx)
        
        if len(track_boxes) < 2:
            return
        
        track_boxes = np.array(track_boxes, dtype=np.float32)
        track_centers = np.array(track_centers, dtype=np.float32)
        track_boxes = self._sanitize_boxes(track_boxes)
        
        # Compute IoU matrix
        iou_matrix = self._compute_iou(track_boxes, track_boxes)
        
        # Compute center distance matrix
        # Calculate pairwise distances between centers
        centers_diff = track_centers[:, None, :] - track_centers[None, :, :]
        center_distances = np.sqrt(np.sum(centers_diff ** 2, axis=2))
        
        # Calculate average box size for relative distance threshold
        box_areas = (track_boxes[:, 2] - track_boxes[:, 0]) * (track_boxes[:, 3] - track_boxes[:, 1])
        avg_box_size = np.sqrt(np.mean(box_areas))
        # Consider tracks duplicates if centers are within 30% of average box size
        center_distance_thresh = 0.3 * avg_box_size
        
        # Find duplicate pairs (excluding self-comparisons)
        np.fill_diagonal(iou_matrix, 0)
        np.fill_diagonal(center_distances, np.inf)
        
        # Remove tracks with high IoU OR close centers (duplicates)
        tracks_to_remove = set()
        for i in range(len(track_indices)):
            if i in tracks_to_remove:
                continue
            
            for j in range(i + 1, len(track_indices)):
                if j in tracks_to_remove:
                    continue
                
                iou = iou_matrix[i, j]
                center_dist = center_distances[i, j]
                
                # Check if duplicate: high IoU OR very close centers
                is_duplicate = (iou >= self.duplicate_iou_thresh) or (center_dist < center_distance_thresh)
                
                if is_duplicate:
                    # Keep the track with more hits and higher score
                    track_i = self.tracked_tracks[track_indices[i]]
                    track_j = self.tracked_tracks[track_indices[j]]
                    
                    if track_i.hits > track_j.hits or (track_i.hits == track_j.hits and track_i.score > track_j.score):
                        tracks_to_remove.add(j)
                        logger.debug(
                            f"Removing duplicate track {track_j.track_id} "
                            f"(IoU={iou:.2f}, dist={center_dist:.1f}), keeping {track_i.track_id}"
                        )
                    else:
                        tracks_to_remove.add(i)
                        logger.debug(
                            f"Removing duplicate track {track_i.track_id} "
                            f"(IoU={iou:.2f}, dist={center_dist:.1f}), keeping {track_j.track_id}"
                        )
                        break
        
        # Remove duplicate tracks (in reverse order to maintain indices)
        tracks_to_remove_sorted = sorted([track_indices[idx] for idx in tracks_to_remove], reverse=True)
        for track_idx in tracks_to_remove_sorted:
            if track_idx < len(self.tracked_tracks):
                track = self.tracked_tracks[track_idx]
                self.tracked_tracks.pop(track_idx)
                logger.debug(f"Removed duplicate track {track.track_id}")
    
    @staticmethod
    def _sanitize_boxes(boxes: np.ndarray) -> np.ndarray:
        """Ensure boxes have positive width/height."""
        if boxes.size == 0:
            return boxes
        boxes = boxes.copy()
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        invalid_w = widths <= 1e-6
        invalid_h = heights <= 1e-6
        boxes[invalid_w, 2] = boxes[invalid_w, 0] + 1e-6
        boxes[invalid_h, 3] = boxes[invalid_h, 1] + 1e-6
        boxes = np.nan_to_num(boxes, nan=0.0, posinf=0.0, neginf=0.0)
        return boxes

