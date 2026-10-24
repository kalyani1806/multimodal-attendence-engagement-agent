"""
EPIC 2: Vision Module Development - Head Pose & Gaze Orientation Estimation
Estimates Yaw, Pitch, Roll angles to determine whether learner is attending to screen/instructor.
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Any

class HeadPoseEstimator:
    """
    Estimates 3D Head Pose (Yaw, Pitch, Roll) using 2D face geometry & perspective-n-point (PnP).
    Classifies gaze orientation: Forward (Attentive), Left, Right, Down (Distracted), Up.
    """
    def __init__(self):
        # 3D generic facial model points (in mm or arbitrary model coordinates)
        # Nose tip, Chin, Left eye left corner, Right eye right corner, Left mouth corner, Right mouth corner
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ], dtype=np.float64)

        eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
        self.eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

    def estimate_pose_from_crop(self, face_crop: np.ndarray, face_bbox: Tuple[int, int, int, int], frame_shape: Tuple[int, int]) -> Dict[str, Any]:
        """
        Estimates head pose angles and attentiveness from face bounding box and image crop.
        """
        h_frame, w_frame = frame_shape[:2]
        x, y, w, h = face_bbox

        # Approximate 2D facial landmarks from face crop geometry
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY) if len(face_crop.shape) == 3 else face_crop
        eyes = self.eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(15, 15))

        # Camera internals (focal length approximation)
        focal_length = w_frame
        center = (w_frame / 2, h_frame / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1)) # Assuming no lens distortion

        # Estimate eye centers or approximate landmarks
        if len(eyes) >= 2:
            # Sort eyes left-to-right
            eyes_sorted = sorted(eyes, key=lambda e: e[0])
            e1 = eyes_sorted[0]
            e2 = eyes_sorted[1]
            left_eye_center = (x + e1[0] + e1[2] // 2, y + e1[1] + e1[3] // 2)
            right_eye_center = (x + e2[0] + e2[2] // 2, y + e2[1] + e2[3] // 2)
            
            # Approximate nose & chin from eye line
            eye_dx = right_eye_center[0] - left_eye_center[0]
            eye_dy = right_eye_center[1] - left_eye_center[1]
            roll_angle = np.degrees(np.arctan2(eye_dy, eye_dx))

            mid_eyes_x = (left_eye_center[0] + right_eye_center[0]) / 2.0
            mid_eyes_y = (left_eye_center[1] + right_eye_center[1]) / 2.0

            # Nose tip approximation (slightly below midpoint between eyes)
            nose_tip = (mid_eyes_x, mid_eyes_y + h * 0.22)
            chin = (mid_eyes_x, y + h * 0.95)
            left_mouth = (mid_eyes_x - w * 0.2, y + h * 0.75)
            right_mouth = (mid_eyes_x + w * 0.2, y + h * 0.75)
            
            image_points = np.array([
                nose_tip,
                chin,
                left_eye_center,
                right_eye_center,
                left_mouth,
                right_mouth
            ], dtype=np.float64)

            # Solve PnP
            success, rotation_vector, translation_vector = cv2.solvePnP(
                self.model_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if success:
                # Convert rotation vector to rotation matrix
                rmat, _ = cv2.Rodrigues(rotation_vector)
                # Compute Euler angles (Pitch, Yaw, Roll)
                sy = np.sqrt(rmat[0, 0] * rmat[0, 0] + rmat[1, 0] * rmat[1, 0])
                singular = sy < 1e-6

                if not singular:
                    pitch = np.degrees(np.arctan2(rmat[2, 1], rmat[2, 2]))
                    yaw = np.degrees(np.arctan2(-rmat[2, 0], sy))
                    roll = np.degrees(np.arctan2(rmat[1, 0], rmat[0, 0]))
                else:
                    pitch = np.degrees(np.arctan2(-rmat[1, 2], rmat[1, 1]))
                    yaw = np.degrees(np.arctan2(-rmat[2, 0], sy))
                    roll = 0.0
            else:
                yaw, pitch, roll = 0.0, 0.0, 0.0
        else:
            # Fallback estimation based on face aspect & position
            face_center_x = x + w / 2.0
            face_center_y = y + h / 2.0
            yaw = float((face_center_x - (w_frame / 2.0)) / (w_frame / 2.0) * 20.0)
            pitch = float((face_center_y - (h_frame / 2.0)) / (h_frame / 2.0) * 15.0)
            roll = 0.0

        # Classify orientation
        orientation = "Forward (Attentive)"
        if abs(yaw) > 25.0:
            orientation = "Looking Right" if yaw > 0 else "Looking Left"
        elif pitch > 20.0:
            orientation = "Looking Down (Distracted/Notes)"
        elif pitch < -20.0:
            orientation = "Looking Up"

        # Calculate gaze engagement score (1.0 = ideal forward attention, decreases with deviation)
        total_deviation = (abs(yaw) / 45.0) * 0.5 + (abs(pitch) / 35.0) * 0.5
        gaze_score = max(0.0, min(1.0, 1.0 - total_deviation))

        return {
            "yaw": float(np.round(yaw, 1)),
            "pitch": float(np.round(pitch, 1)),
            "roll": float(np.round(roll, 1)),
            "orientation": orientation,
            "gaze_score": float(np.round(gaze_score, 2)),
            "is_attentive": gaze_score >= 0.55
        }
