"""Utility functions for facial landmark processing."""

import numpy as np
from typing import List, Tuple

# Landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [13, 14, 78, 308, 82, 312, 87, 317]

class LandmarkCalculator:
    """Handles all landmark-based calculations."""
    
    @staticmethod
    def calculate_angle(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        """Calculate angle between two points."""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return np.degrees(np.arctan2(dy, dx))
    
    @staticmethod
    def euclidean_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two points."""
        return np.linalg.norm(np.array(p1) - np.array(p2))
    
    @staticmethod
    def calculate_ear(eye_pts: List[Tuple[int, int]]) -> float:
        """Calculate Eye Aspect Ratio."""
        p1, p2, p3, p4, p5, p6 = eye_pts
        return (LandmarkCalculator.euclidean_distance(p2, p6) + 
                LandmarkCalculator.euclidean_distance(p3, p5)) / (2.0 * LandmarkCalculator.euclidean_distance(p1, p4))
    
    @staticmethod
    def calculate_mar(mouth_pts: List[Tuple[int, int]]) -> float:
        """Calculate Mouth Aspect Ratio."""
        p1, p2, p3, p4, p5, p6 = mouth_pts[:6]
        return (LandmarkCalculator.euclidean_distance(p2, p6) + 
                LandmarkCalculator.euclidean_distance(p3, p5)) / (2.0 * LandmarkCalculator.euclidean_distance(p1, p4))
    
    @staticmethod
    def extract_points(face_landmarks, indices: List[int], w: int, h: int) -> List[Tuple[int, int]]:
        """Extract pixel coordinates for given landmark indices."""
        return [(int(face_landmarks[i].x * w), int(face_landmarks[i].y * h)) for i in indices]

class FacePointExtractor:
    """Extracts key facial points from landmarks."""
    
    def __init__(self, face_landmarks, frame_shape):
        self.landmarks = face_landmarks
        self.h, self.w, _ = frame_shape
    
    def get_nose(self) -> Tuple[int, int]:
        """Get nose tip position."""
        return self._get_point(1)
    
    def get_left_eye_outer(self) -> Tuple[int, int]:
        """Get left eye outer corner."""
        return self._get_point(33)
    
    def get_right_eye_outer(self) -> Tuple[int, int]:
        """Get right eye outer corner."""
        return self._get_point(263)
    
    def get_chin(self) -> Tuple[int, int]:
        """Get chin point."""
        return self._get_point(152)
    
    def _get_point(self, index: int) -> Tuple[int, int]:
        """Get point by index."""
        point = self.landmarks[index]
        return (int(point.x * self.w), int(point.y * self.h))