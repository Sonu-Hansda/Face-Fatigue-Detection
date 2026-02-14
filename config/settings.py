import cv2 

MODEL_PATH = 'assets/face_landmarker_v2_with_blendshapes.task'

class DetectionThresholds:
    """Threshold values for fatigue detection."""
    EAR_THRESHOLD = 0.25      # Eye Aspect Ratio threshold
    MAR_THRESHOLD = 0.6       # Mouth Aspect Ratio threshold
    HEAD_TILT_THRESHOLD = 15  # Degrees of head tilt
    CLOSED_EYE_THRESHOLD = 15  # Frames before alert
    ALERT_COOLDOWN = 5         # Seconds between alerts

class FatigueLevels:
    """Thresholds for fatigue level categories."""
    LOW = 40
    MEDIUM = 70
    HIGH = 100

class DisplaySettings:
    """Display and UI settings."""
    WINDOW_NAME = "Fatigue Detection System"
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE_SMALL = 0.4
    FONT_SCALE_MEDIUM = 0.6
    FONT_SCALE_LARGE = 0.8
    FONT_SCALE_XLARGE = 1.0
    
    # Colors (BGR format)
    COLOR_WHITE = (255, 255, 255)
    COLOR_GREEN = (0, 255, 0)
    COLOR_RED = (0, 0, 255)
    COLOR_YELLOW = (0, 255, 255)
    COLOR_BLUE = (255, 0, 0)
    COLOR_CYAN = (255, 255, 0)
    COLOR_GRAY = (200, 200, 200)
    
    # Layout
    LEFT_PANEL_LABEL = "NORMAL VIEW"
    RIGHT_PANEL_LABEL = "ANALYZED VIEW"
    HEADER_Y_OFFSET = 60