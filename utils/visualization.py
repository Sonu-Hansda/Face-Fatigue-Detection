"""Visualization utilities for drawing on frames."""

import cv2
import numpy as np
from typing import List, Tuple

from models.fatigue_state import FatigueState
from config.settings import DisplaySettings as ds

class PanelDrawer:
    """Handles drawing on the left (normal) panel."""
    
    @staticmethod
    def draw_header(frame: np.ndarray, text: str):
        """Draw panel header."""
        cv2.putText(frame, text, (30, 30), ds.FONT, 
                   ds.FONT_SCALE_LARGE, ds.COLOR_WHITE, 2)
    
    @staticmethod
    def draw_metric(frame: np.ndarray, label: str, value: float, 
                   description: str, y_pos: int):
        """Draw a metric with label and description."""
        cv2.putText(frame, f"{label}: {value:.3f}", (30, y_pos), 
                   ds.FONT, ds.FONT_SCALE_MEDIUM, ds.COLOR_WHITE, 2)
        cv2.putText(frame, f"  → {description}", (30, y_pos + 20), 
                   ds.FONT, ds.FONT_SCALE_SMALL, ds.COLOR_GRAY, 1)
        return y_pos + 50
    
    @staticmethod
    def draw_fatigue_bar(frame: np.ndarray, level: float, y_pos: int):
        """Draw fatigue level bar."""
        bar_length = 250
        filled_length = int(bar_length * level / 100)
        
        # Determine color based on level
        if level < 40:
            color = ds.COLOR_GREEN
        elif level < 70:
            color = ds.COLOR_YELLOW
        else:
            color = ds.COLOR_RED
        
        cv2.putText(frame, "Fatigue Level:", (30, y_pos), 
                   ds.FONT, ds.FONT_SCALE_MEDIUM, ds.COLOR_WHITE, 2)
        
        # Draw bar
        cv2.rectangle(frame, (30, y_pos + 10), (30 + filled_length, y_pos + 30), 
                     color, -1)
        cv2.rectangle(frame, (30, y_pos + 10), (30 + bar_length, y_pos + 30), 
                     ds.COLOR_WHITE, 2)
        
        # Draw percentage
        cv2.putText(frame, f"{level:.0f}%", (30 + bar_length + 10, y_pos + 25), 
                   ds.FONT, ds.FONT_SCALE_MEDIUM, ds.COLOR_WHITE, 2)
        
        return y_pos + 50
    
    @staticmethod
    def draw_status_indicator(frame: np.ndarray, label: str, status: str, 
                             is_abnormal: bool, y_pos: int):
        """Draw status indicator with color coding."""
        color = ds.COLOR_RED if is_abnormal else ds.COLOR_GREEN
        cv2.putText(frame, f"{label}: {status}", (30, y_pos), 
                   ds.FONT, ds.FONT_SCALE_MEDIUM, color, 2)
        return y_pos + 30
    
    @staticmethod
    def draw_alerts(frame: np.ndarray, alerts: List[Tuple[str, str]], y_pos: int):
        """Draw active alerts."""
        if not alerts:
            return y_pos
        
        cv2.putText(frame, "ACTIVE ALERTS:", (30, y_pos), 
                   ds.FONT, ds.FONT_SCALE_MEDIUM, ds.COLOR_RED, 2)
        
        for i, (alert_type, msg) in enumerate(alerts):
            cv2.putText(frame, alert_type, (30, y_pos + 30 + i * 25), 
                       ds.FONT, ds.FONT_SCALE_MEDIUM, ds.COLOR_RED, 2)
            cv2.putText(frame, f"  {msg}", (30, y_pos + 50 + i * 25), 
                       ds.FONT, ds.FONT_SCALE_SMALL, ds.COLOR_GRAY, 1)
        
        return y_pos + 30 + len(alerts) * 50

class MeshDrawer:
    """Handles drawing on the right (mesh) panel."""
    
    @staticmethod
    def draw_header(frame: np.ndarray, text: str):
        """Draw panel header."""
        cv2.putText(frame, text, (30, 30), ds.FONT, 
                   ds.FONT_SCALE_LARGE, ds.COLOR_YELLOW, 2)
    
    @staticmethod
    def draw_landmarks(frame: np.ndarray, points: List[Tuple[int, int]], 
                      color: Tuple[int, int, int], label: str = ""):
        """Draw landmarks with optional labels."""
        for pt in points:
            cv2.circle(frame, pt, 3, color, -1)
            if label and pt == points[0]:
                cv2.putText(frame, label, (pt[0] + 5, pt[1] - 5), 
                           ds.FONT, ds.FONT_SCALE_SMALL, color, 1)
    
    @staticmethod
    def draw_head_pose_lines(frame: np.ndarray, left_eye: Tuple[int, int], 
                            right_eye: Tuple[int, int], nose: Tuple[int, int], 
                            chin: Tuple[int, int]):
        """Draw head pose measurement lines."""
        # Eye line (tilt)
        cv2.line(frame, left_eye, right_eye, ds.COLOR_BLUE, 2)
        mid_eye = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
        cv2.putText(frame, "Eye Line (Tilt)", (mid_eye[0] - 50, mid_eye[1] - 10), 
                   ds.FONT, ds.FONT_SCALE_SMALL, ds.COLOR_BLUE, 1)
        
        # Nose-chin line (pitch)
        cv2.line(frame, nose, chin, ds.COLOR_CYAN, 2)
        mid_face = ((nose[0] + chin[0]) // 2, (nose[1] + chin[1]) // 2)
        cv2.putText(frame, "Nose-Chin (Pitch)", (mid_face[0] - 50, mid_face[1]), 
                   ds.FONT, ds.FONT_SCALE_SMALL, ds.COLOR_CYAN, 1)
    
    @staticmethod
    def draw_metrics_overlay(frame: np.ndarray, ear: float, mar: float, 
                            tilt: float, w: int, h: int):
        """Draw metrics overlay on mesh view."""
        cv2.putText(frame, f"EAR: {ear:.3f}", (w - 200, 80), 
                   ds.FONT, ds.FONT_SCALE_SMALL, ds.COLOR_WHITE, 1)
        cv2.putText(frame, f"MAR: {mar:.3f}", (w - 200, 110), 
                   ds.FONT, ds.FONT_SCALE_SMALL, ds.COLOR_WHITE, 1)
        cv2.putText(frame, f"Tilt: {tilt:.1f}°", (w - 200, 140), 
                   ds.FONT, ds.FONT_SCALE_SMALL, ds.COLOR_WHITE, 1)
    
    @staticmethod
    def draw_legend(frame: np.ndarray, w: int, h: int):
        """Draw color legend."""
        cv2.putText(frame, "Green: Normal", (w - 200, h - 90), 
                   ds.FONT, ds.FONT_SCALE_SMALL, ds.COLOR_GREEN, 1)
        cv2.putText(frame, "Red: Fatigue Indicator", (w - 200, h - 70), 
                   ds.FONT, ds.FONT_SCALE_SMALL, ds.COLOR_RED, 1)
        cv2.putText(frame, "Blue Lines: Measurements", (w - 200, h - 50), 
                   ds.FONT, ds.FONT_SCALE_SMALL, ds.COLOR_BLUE, 1)
    
    @staticmethod
    def draw_fatigue_warning(frame: np.ndarray, w: int, h: int):
        """Draw fatigue warning overlay."""
        overlay = frame.copy()
        cv2.rectangle(overlay, (w // 2 - 150, 50), (w // 2 + 150, 100), 
                     ds.COLOR_RED, -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        cv2.putText(frame, "FATIGUE DETECTED", (w // 2 - 140, 80), 
                   ds.FONT, ds.FONT_SCALE_LARGE, ds.COLOR_WHITE, 2)

class FrameComposer:
    """Composes the final side-by-side view."""
    
    @staticmethod
    def combine_views(left_frame: np.ndarray, right_frame: np.ndarray) -> np.ndarray:
        """Combine left and right frames side by side."""
        # Ensure same height
        if left_frame.shape[0] != right_frame.shape[0]:
            h = min(left_frame.shape[0], right_frame.shape[0])
            left_frame = cv2.resize(left_frame, 
                                   (int(left_frame.shape[1] * h / left_frame.shape[0]), h))
            right_frame = cv2.resize(right_frame, 
                                    (int(right_frame.shape[1] * h / right_frame.shape[0]), h))
        
        combined = np.hstack((left_frame, right_frame))
        
        # Add divider
        divider_x = left_frame.shape[1]
        cv2.line(combined, (divider_x, 0), (divider_x, combined.shape[0]), 
                ds.COLOR_WHITE, 2)
        
        # Add panel labels
        cv2.putText(combined, ds.LEFT_PANEL_LABEL, (30, ds.HEADER_Y_OFFSET), 
                   ds.FONT, ds.FONT_SCALE_LARGE, ds.COLOR_WHITE, 2)
        cv2.putText(combined, ds.RIGHT_PANEL_LABEL, 
                   (divider_x + 30, ds.HEADER_Y_OFFSET), 
                   ds.FONT, ds.FONT_SCALE_LARGE, ds.COLOR_YELLOW, 2)
        
        return combined