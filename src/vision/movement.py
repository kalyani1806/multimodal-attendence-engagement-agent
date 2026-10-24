"""
EPIC 2: Vision Module Development - Movement Pattern & Inactivity Dynamics
Measures optical motion, fidgeting, and sustained inactivity (frozen posture/sleeping).
"""

import cv2
import numpy as np
import time
from typing import Dict, Any, Tuple

class MovementAnalyzer:
    """
    Quantifies learner physical movement patterns and detects prolonged inactivity duration.
    """
    def __init__(self, inactivity_motion_threshold: float = 1.8, inactivity_timeout_sec: float = 10.0):
        self.inactivity_motion_threshold = inactivity_motion_threshold
        self.inactivity_timeout_sec = inactivity_timeout_sec
        
        # State per participant:
        # pid -> {'prev_crop_gray': np.ndarray, 'last_moved_time': float, 'inactivity_sec': float}
        self.participant_motion_state: Dict[int, Dict[str, Any]] = {}

    def analyze_movement(self, face_crop: np.ndarray, participant_id: int, current_timestamp: float) -> Dict[str, Any]:
        """
        Calculates motion magnitude between frames and tracks sustained inactivity duration.
        """
        if face_crop is None or face_crop.size == 0:
            return {
                "motion_magnitude": 0.0,
                "inactivity_duration_sec": 0.0,
                "movement_state": "Stationary",
                "stability_score": 0.8
            }

        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY) if len(face_crop.shape) == 3 else face_crop
        # Resize to standard small dimension for motion comparison
        gray_small = cv2.resize(gray, (64, 64))

        if participant_id not in self.participant_motion_state:
            self.participant_motion_state[participant_id] = {
                'prev_crop_gray': gray_small,
                'last_moved_time': current_timestamp,
                'inactivity_sec': 0.0
            }
            return {
                "motion_magnitude": 1.0,
                "inactivity_duration_sec": 0.0,
                "movement_state": "Calm & Attentive",
                "stability_score": 0.9
            }

        state = self.participant_motion_state[participant_id]
        prev_gray = state['prev_crop_gray']

        # Absolute frame difference
        diff = cv2.absdiff(prev_gray, gray_small)
        motion_mag = float(np.mean(diff))

        # Check if significant motion occurred
        if motion_mag > self.inactivity_motion_threshold:
            state['last_moved_time'] = current_timestamp
            state['inactivity_sec'] = 0.0
        else:
            state['inactivity_sec'] = current_timestamp - state['last_moved_time']

        state['prev_crop_gray'] = gray_small

        # Classify state
        # Moderate motion: 1.5 - 6.0 (normal attentive nodding/head movement)
        # High motion: > 8.0 (restless / fidgeting / looking around)
        # Prolonged zero motion: inactivity > inactivity_timeout_sec (frozen, heads down, asleep)
        if state['inactivity_sec'] > self.inactivity_timeout_sec:
            movement_state = "Prolonged Inactivity (Potential Sleep/Disengagement)"
            stability_score = 0.4
        elif motion_mag > 8.0:
            movement_state = "High Restlessness / Fidgeting"
            stability_score = 0.6
        else:
            movement_state = "Calm & Attentive"
            stability_score = 0.95

        return {
            "participant_id": participant_id,
            "motion_magnitude": float(np.round(motion_mag, 2)),
            "inactivity_duration_sec": float(np.round(state['inactivity_sec'], 1)),
            "movement_state": movement_state,
            "stability_score": float(np.round(stability_score, 2))
        }
