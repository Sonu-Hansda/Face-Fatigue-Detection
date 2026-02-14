"""Data models for fatigue detection."""

from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class FatigueState:
    """Represents the current fatigue state of the user."""
    is_fatigued: bool = False
    fatigue_level: float = 0.0  # 0-100%
    alerts: List[Tuple[str, str]] = field(default_factory=list)
    
    def __post_init__(self):
        if self.alerts is None:
            self.alerts = []
    
    def add_alert(self, alert_type: str, message: str):
        """Add an alert to the state."""
        self.alerts.append((alert_type, message))
    
    def clear_alerts(self):
        """Clear all alerts."""
        self.alerts.clear()

@dataclass
class FacialMetrics:
    """Container for all facial metrics."""
    ear: float = 0.0
    mar: float = 0.0
    roll_angle: float = 0.0
    pitch_distance: float = 0.0
    timestamp: float = 0.0
    
    def to_dict(self) -> dict:
        """Convert metrics to dictionary."""
        return {
            'ear': self.ear,
            'mar': self.mar,
            'roll_angle': self.roll_angle,
            'pitch_distance': self.pitch_distance,
            'timestamp': self.timestamp
        }