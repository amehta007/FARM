"""Zone management and point-in-polygon detection."""

from typing import Dict, List, Tuple

import numpy as np
from shapely.geometry import Point, Polygon as ShapelyPolygon


class Zone:
    """Represents a polygonal zone in the video frame."""
    
    def __init__(self, name: str, points: List[Tuple[float, float]]):
        """
        Initialize zone.
        
        Args:
            name: Zone name/identifier
            points: List of (x, y) vertices defining the polygon
        """
        self.name = name
        self.points = points
        self.polygon = ShapelyPolygon(points)
        self.area = self.polygon.area
    
    def contains_point(self, point: Tuple[float, float]) -> bool:
        """
        Check if a point is inside the zone.
        
        Args:
            point: (x, y) coordinates
        
        Returns:
            True if point is inside zone
        """
        return self.polygon.contains(Point(point))
    
    def get_vertices(self) -> np.ndarray:
        """
        Get zone vertices as numpy array for drawing.
        
        Returns:
            (N x 2) array of vertices
        """
        return np.array(self.points, dtype=np.int32)


class ZoneManager:
    """Manages multiple zones and tracks occupancy."""
    
    def __init__(self, zones: List[Dict]):
        """
        Initialize zone manager.
        
        Args:
            zones: List of zone configs with 'name' and 'points'
        """
        self.zones = [Zone(z["name"], [(p[0], p[1]) for p in z["points"]]) for z in zones]
        
        # Track time spent in each zone per track ID
        self.zone_times: Dict[int, Dict[str, float]] = {}
        
    def update(self, track_id: int, center: Tuple[float, float], dt: float):
        """
        Update zone occupancy for a tracked object.
        
        Args:
            track_id: Tracking ID
            center: (x, y) center point of tracked object
            dt: Time delta in seconds
        """
        if track_id not in self.zone_times:
            self.zone_times[track_id] = {zone.name: 0.0 for zone in self.zones}
        
        for zone in self.zones:
            if zone.contains_point(center):
                self.zone_times[track_id][zone.name] += dt
    
    def get_zone_time(self, track_id: int, zone_name: str) -> float:
        """
        Get time spent in a specific zone.
        
        Args:
            track_id: Tracking ID
            zone_name: Zone name
        
        Returns:
            Time in seconds
        """
        if track_id not in self.zone_times:
            return 0.0
        return self.zone_times[track_id].get(zone_name, 0.0)
    
    def get_all_zone_times(self, track_id: int) -> Dict[str, float]:
        """
        Get all zone times for a track.
        
        Args:
            track_id: Tracking ID
        
        Returns:
            Dictionary of zone_name -> time_seconds
        """
        if track_id not in self.zone_times:
            return {zone.name: 0.0 for zone in self.zones}
        return self.zone_times[track_id].copy()
    
    def get_total_occupancy(self) -> Dict[str, float]:
        """
        Get total occupancy time across all tracks for each zone.
        
        Returns:
            Dictionary of zone_name -> total_time_seconds
        """
        totals = {zone.name: 0.0 for zone in self.zones}
        for zone_dict in self.zone_times.values():
            for zone_name, time_val in zone_dict.items():
                totals[zone_name] += time_val
        return totals

