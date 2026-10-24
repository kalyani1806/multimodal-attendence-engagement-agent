"""
EPIC 2: Vision Module Development - Drowsiness, Eye Closure & Yawn Detection
Detects learner fatigue, prolonged eye closure, and yawning patterns.
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple

class DrowsinessDetector:
    """
    Analyzes eye openness and mouth state to detect learner drowsiness or fatigue.
    Computes an Alertness Score between 0.0 (completely asleep/drowsy) and 1.0 (fully alert).
    """
    def __init__(self, ear_threshold: float = 0.22, consecutive_frames_threshold: int = 12):
        self.ear_threshold = ear_threshold
        self.consecutive_frames_threshold = consecutive_frames_threshold
        
        # Track consecutive closed frames per participant ID
        self.closed_eye_counters: Dict[int, int] = {}
        self.yawn_counters: Dict[int, int] = {}

        eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
        self.eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
        
        smile_cascade_path = cv2.data.haarcascades + 'haarcascade_smile.xml'
        self.smile_cascade = cv2.CascadeClassifier(smile_cascade_path)

    def analyze(self, face_crop: np.ndarray, participant_id: int) -> Dict[str, Any]:
        """
        Analyzes face crop for eye closure, blink/drowsiness, and yawning.
        """
        if face_crop is None or face_crop.size == 0:
            return {
                "alertness_score": 0.5,
                "is_drowsy": False,
                "is_yawning": False,
                "status": "Unknown"
            }

        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY) if len(face_crop.shape) == 3 else face_crop
        h, w = gray.shape

        # Eye region: upper 60% of face
        eye_region = gray[int(h * 0.15):int(h * 0.60), :]
        eyes = self.eye_cascade.detectMultiScale(eye_region, scaleFactor=1.1, minNeighbors=3, minSize=(15, 15))

        # Mouth region: lower 45% of face
        mouth_region = gray[int(h * 0.55):h, :]

        # Determine eye openness
        eyes_detected = len(eyes)
        if participant_id not in self.closed_eye_counters:
            self.closed_eye_counters[participant_id] = 0
            self.yawn_counters[participant_id] = 0

        # If eyes are not detected inside the upper face for multiple frames, eyes may be closed
        if eyes_detected == 0:
            self.closed_eye_counters[participant_id] += 1
        else:
            self.closed_eye_counters[participant_id] = max(0, self.closed_eye_counters[participant_id] - 1)

        # Approximate Mouth Opening / Yawn using thresholding in mouth region
        _, thresh_mouth = cv2.threshold(mouth_region, 50, 255, cv2.THRESH_BINARY_INV)
        dark_pixels_ratio = np.sum(thresh_mouth == 255) / (mouth_region.size + 1e-6)
        is_yawning = dark_pixels_ratio > 0.40

        is_drowsy = self.closed_eye_counters[participant_id] >= self.consecutive_frames_threshold
        
        # Calculate Alertness Score (0.0 to 1.0)
        penalty = min(0.6, (self.closed_eye_counters[participant_id] / self.consecutive_frames_threshold) * 0.6)
        if is_yawning:
            penalty += 0.3
        
        alertness_score = max(0.1, min(1.0, 1.0 - penalty))

        status = "Active & Alert"
        if is_drowsy:
            status = "Drowsy / Eyes Closed"
        elif is_yawning:
            status = "Yawning / Fatigued"
        elif alertness_score < 0.7:
            status = "Mild Inattention"

        return {
            "participant_id": participant_id,
            "alertness_score": float(np.round(alertness_score, 2)),
            "is_drowsy": is_drowsy,
            "is_yawning": is_yawning,
            "closed_frames": self.closed_eye_counters[participant_id],
            "status": status
        }
