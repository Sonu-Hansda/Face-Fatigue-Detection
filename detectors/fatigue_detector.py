"""Fatigue detection logic using rule-based approach."""

from collections import deque
import numpy as np
import time
from typing import Tuple

from models.fatigue_state import FatigueState, FacialMetrics
from config.settings import DetectionThresholds, FatigueLevels

class SimpleFatigueDetector:
    """Rule-based fatigue detector using facial metrics."""
    
    def __init__(self):
        # Thresholds
        self.ear_threshold = DetectionThresholds.EAR_THRESHOLD
        self.mar_threshold = DetectionThresholds.MAR_THRESHOLD
        self.head_tilt_threshold = DetectionThresholds.HEAD_TILT_THRESHOLD
        self.closed_eye_threshold = DetectionThresholds.CLOSED_EYE_THRESHOLD
        self.alert_cooldown = DetectionThresholds.ALERT_COOLDOWN
        
        # History buffers
        self.history_size = 30
        self.ear_history = deque(maxlen=self.history_size)
        self.mar_history = deque(maxlen=self.history_size)
        self.head_tilt_history = deque(maxlen=self.history_size)
        
        # State variables
        self.consecutive_closed_eyes = 0
        self.last_alert_time = 0
    
    def update_histories(self, metrics: FacialMetrics):
        """Update metric histories."""
        self.ear_history.append(metrics.ear)
        self.mar_history.append(metrics.mar)
        self.head_tilt_history.append(metrics.roll_angle)
    
    def check_eye_closed(self, ear: float) -> Tuple[bool, str]:
        """Check if eyes are closed or nearly closed."""
        if ear < self.ear_threshold:
            self.consecutive_closed_eyes += 1
            if self.consecutive_closed_eyes > self.closed_eye_threshold:
                return True, f"Eyes closed for {self.consecutive_closed_eyes} frames"
        else:
            self.consecutive_closed_eyes = max(0, self.consecutive_closed_eyes - 2)
        return False, ""
    
    def check_yawn(self, mar: float) -> Tuple[bool, str]:
        """Check for yawning (high MAR)."""
        if mar > self.mar_threshold:
            recent_mar = list(self.mar_history)[-10:]
            if len(recent_mar) >= 5 and sum(m > self.mar_threshold for m in recent_mar) >= 3:
                return True, "Possible yawning detected"
        return False, ""
    
    def check_head_tilt(self, head_tilt: float) -> Tuple[bool, str]:
        """Check for excessive head tilting."""
        if abs(head_tilt) > self.head_tilt_threshold:
            return True, f"Excessive head tilt: {head_tilt:.1f}°"
        return False, ""
    
    def check_fatigue_trend(self) -> Tuple[bool, str]:
        """Check for gradual fatigue indicators."""
        if len(self.ear_history) < self.history_size:
            return False, ""
        
        ear_list = list(self.ear_history)
        first_half_ear = np.mean(ear_list[:15])
        second_half_ear = np.mean(ear_list[15:])
        
        if second_half_ear < first_half_ear * 0.8:
            return True, "Eyes gradually closing"
        return False, ""
    
    def calculate_fatigue_level(self, metrics: FacialMetrics) -> float:
        """Calculate overall fatigue level (0-100%)."""
        fatigue_factors = []
        
        # EAR factor (lower = more fatigue)
        ear_factor = max(0, min(1, (0.35 - metrics.ear) / 0.2))
        fatigue_factors.append(ear_factor)
        
        # MAR factor (higher = more fatigue)
        mar_factor = max(0, min(1, (metrics.mar - 0.4) / 0.3))
        fatigue_factors.append(mar_factor * 0.7)
        
        # Head tilt factor
        tilt_factor = max(0, min(1, abs(metrics.roll_angle) / 30))
        fatigue_factors.append(tilt_factor * 0.5)
        
        # Consecutive closed eyes factor
        if self.consecutive_closed_eyes > 0:
            closed_factor = min(1, self.consecutive_closed_eyes / 30)
            fatigue_factors.append(closed_factor)
        
        if fatigue_factors:
            return min(100, np.mean(fatigue_factors) * 100)
        return 0.0
    
    def detect(self, metrics: FacialMetrics, current_time: float) -> FatigueState:
        """Main detection function."""
        state = FatigueState()
        
        # Update histories
        self.update_histories(metrics)
        
        # Check individual rules
        checks = [
            self.check_eye_closed(metrics.ear),
            self.check_yawn(metrics.mar),
            self.check_head_tilt(metrics.roll_angle),
            self.check_fatigue_trend()
        ]
        
        # Collect alerts (with cooldown)
        if current_time - self.last_alert_time > self.alert_cooldown:
            alert_messages = [
                ("⚠️ DROWSY", "Eye closure detected"),
                ("😮 YAWNING", "Yawning detected"),
                ("😴 HEAD DROOP", "Head drooping detected"),
                ("⚠️ FATIGUE TREND", "Fatigue trend detected")
            ]
            
            for i, (alert_type, msg) in enumerate(alert_messages):
                if checks[i][0]:
                    state.add_alert(alert_type, checks[i][1])
            
            if state.alerts:
                self.last_alert_time = current_time
        
        # Calculate fatigue level
        state.fatigue_level = self.calculate_fatigue_level(metrics)
        
        # Final fatigue decision
        state.is_fatigued = (
            checks[0][0] or  # Eye closed
            (checks[1][0] and checks[2][0]) or  # Yawning and head tilt
            state.fatigue_level > FatigueLevels.MEDIUM
        )
        
        return state