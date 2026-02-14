import cv2
import time
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from config.settings import MODEL_PATH, DetectionThresholds
from models.fatigue_state import FatigueState, FacialMetrics
from detectors.fatigue_detector import SimpleFatigueDetector
from utils.landmark_utils import LandmarkCalculator, FacePointExtractor, LEFT_EYE, RIGHT_EYE, MOUTH
from utils.visualization import PanelDrawer, MeshDrawer, FrameComposer

class FatigueDetectionApp:
    """Main application class."""
    
    def __init__(self):
        self.detector = SimpleFatigueDetector()
        self.landmark_calculator = LandmarkCalculator()
        self.panel_drawer = PanelDrawer()
        self.mesh_drawer = MeshDrawer()
        self.frame_composer = FrameComposer()
        
        self.landmarker = self._load_model()
        self.cap = cv2.VideoCapture(0)
        self.global_start_time = time.time()
        
    def _load_model(self):
        """Load the MediaPipe face landmark model."""
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False
        )
        return vision.FaceLandmarker.create_from_options(options)
    
    def _extract_metrics(self, face_landmarks, frame_shape) -> FacialMetrics:
        """Extract facial metrics from landmarks."""
        h, w, _ = frame_shape
        point_extractor = FacePointExtractor(face_landmarks, frame_shape)
        
        # Get key points
        nose_pt = point_extractor.get_nose()
        left_eye_pt = point_extractor.get_left_eye_outer()
        right_eye_pt = point_extractor.get_right_eye_outer()
        chin_pt = point_extractor.get_chin()
        
        # Calculate metrics
        roll_angle = self.landmark_calculator.calculate_angle(left_eye_pt, right_eye_pt)
        pitch_distance = chin_pt[1] - nose_pt[1]
        
        # Extract eye and mouth points
        left_eye_pts = self.landmark_calculator.extract_points(face_landmarks, LEFT_EYE, w, h)
        right_eye_pts = self.landmark_calculator.extract_points(face_landmarks, RIGHT_EYE, w, h)
        mouth_pts = self.landmark_calculator.extract_points(face_landmarks, MOUTH, w, h)
        
        # Calculate EAR and MAR
        ear = (self.landmark_calculator.calculate_ear(left_eye_pts) + 
               self.landmark_calculator.calculate_ear(right_eye_pts)) / 2.0
        mar = self.landmark_calculator.calculate_mar(mouth_pts)
        
        return FacialMetrics(
            ear=ear,
            mar=mar,
            roll_angle=roll_angle,
            pitch_distance=pitch_distance,
            timestamp=time.time()
        )
    
    def _draw_left_panel(self, frame: np.ndarray, metrics: FacialMetrics, 
                         fatigue_state: FatigueState) -> np.ndarray:
        """Draw the left panel (normal view with analytics)."""
        view = frame.copy()
        y_offset = 80
        
        self.panel_drawer.draw_header(view, "LIVE FEED - FATIGUE ANALYTICS")
        
        # Draw metrics
        y_offset = self.panel_drawer.draw_metric(
            view, "EAR", metrics.ear, 
            "Measures eye openness (lower = more closed)", y_offset)
        y_offset = self.panel_drawer.draw_metric(
            view, "MAR", metrics.mar, 
            "Measures mouth openness (higher = yawning)", y_offset)
        y_offset = self.panel_drawer.draw_metric(
            view, "Head Tilt", metrics.roll_angle, 
            "Head angle (higher = drooping)", y_offset)
        
        # Draw fatigue bar
        y_offset = self.panel_drawer.draw_fatigue_bar(view, fatigue_state.fatigue_level, y_offset)
        y_offset += 30
        
        # Draw status indicators
        y_offset = self.panel_drawer.draw_status_indicator(
            view, "Eyes", "CLOSED" if metrics.ear < DetectionThresholds.EAR_THRESHOLD else "OPEN",
            metrics.ear < DetectionThresholds.EAR_THRESHOLD, y_offset)
        y_offset = self.panel_drawer.draw_status_indicator(
            view, "Mouth", "YAWNING" if metrics.mar > DetectionThresholds.MAR_THRESHOLD else "NORMAL",
            metrics.mar > DetectionThresholds.MAR_THRESHOLD, y_offset)
        y_offset = self.panel_drawer.draw_status_indicator(
            view, "Head", "TILTED" if abs(metrics.roll_angle) > DetectionThresholds.HEAD_TILT_THRESHOLD else "NORMAL",
            abs(metrics.roll_angle) > DetectionThresholds.HEAD_TILT_THRESHOLD, y_offset)
        
        # Draw alerts
        self.panel_drawer.draw_alerts(view, fatigue_state.alerts, y_offset + 40)
        
        return view
    
    def _draw_right_panel(self, frame: np.ndarray, face_landmarks, 
                          metrics: FacialMetrics, fatigue_state: FatigueState) -> np.ndarray:
        """Draw the right panel (mesh view with analysis)."""
        view = frame.copy()
        h, w, _ = frame.shape
        
        self.mesh_drawer.draw_header(view, "FACE MESH ANALYSIS")
        
        # Get points for drawing
        point_extractor = FacePointExtractor(face_landmarks, frame.shape)
        left_eye_pts = self.landmark_calculator.extract_points(face_landmarks, LEFT_EYE, w, h)
        right_eye_pts = self.landmark_calculator.extract_points(face_landmarks, RIGHT_EYE, w, h)
        mouth_pts = self.landmark_calculator.extract_points(face_landmarks, MOUTH, w, h)
        
        # Draw landmarks with color coding
        eye_color = (0, 255, 0) if metrics.ear > DetectionThresholds.EAR_THRESHOLD else (0, 0, 255)
        mouth_color = (0, 255, 0) if metrics.mar < DetectionThresholds.MAR_THRESHOLD else (0, 0, 255)
        
        self.mesh_drawer.draw_landmarks(view, left_eye_pts + right_eye_pts, eye_color, "Eye")
        self.mesh_drawer.draw_landmarks(view, mouth_pts, mouth_color, "Mouth")
        
        # Draw head pose lines
        self.mesh_drawer.draw_head_pose_lines(
            view,
            point_extractor.get_left_eye_outer(),
            point_extractor.get_right_eye_outer(),
            point_extractor.get_nose(),
            point_extractor.get_chin()
        )
        
        # Draw metrics overlay
        self.mesh_drawer.draw_metrics_overlay(view, metrics.ear, metrics.mar, 
                                              metrics.roll_angle, w, h)
        
        # Draw legend
        self.mesh_drawer.draw_legend(view, w, h)
        
        # Draw fatigue warning if needed
        if fatigue_state.is_fatigued:
            self.mesh_drawer.draw_fatigue_warning(view, w, h)
        
        return view
    
    def run(self):
        """Main application loop."""
        cv2.namedWindow("Fatigue Detection System", cv2.WINDOW_NORMAL)
        
        print("=" * 60)
        print("Fatigue Detection System Started")
        print("=" * 60)
        print(f"EAR threshold: {DetectionThresholds.EAR_THRESHOLD}")
        print(f"MAR threshold: {DetectionThresholds.MAR_THRESHOLD}")
        print(f"Head tilt threshold: {DetectionThresholds.HEAD_TILT_THRESHOLD}°")
        print("-" * 60)
        print("Press 'q' to quit")
        print("=" * 60)
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            # Process frame with MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp = int((time.time() - self.global_start_time) * 1000)
            
            result = self.landmarker.detect_for_video(mp_image, timestamp)
            
            if result.face_landmarks:
                face_landmarks = result.face_landmarks[0]
                
                # Extract metrics
                metrics = self._extract_metrics(face_landmarks, frame.shape)
                
                # Detect fatigue
                fatigue_state = self.detector.detect(metrics, time.time())
                
                # Draw panels
                left_view = self._draw_left_panel(frame, metrics, fatigue_state)
                right_view = self._draw_right_panel(frame, face_landmarks, metrics, fatigue_state)
            else:
                # No face detected
                left_view = frame.copy()
                right_view = frame.copy()
                cv2.putText(left_view, "NO FACE DETECTED", (30, frame.shape[0]//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                cv2.putText(right_view, "NO FACE DETECTED", (30, frame.shape[0]//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            
            # Combine and display
            combined = self.frame_composer.combine_views(left_view, right_view)
            cv2.imshow("Fatigue Detection System", combined)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = FatigueDetectionApp()
    app.run()