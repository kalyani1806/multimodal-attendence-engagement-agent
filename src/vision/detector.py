"""
EPIC 2: Vision Module Development - Face Detection & Participant Tracking
Implements sustained presence tracking to eliminate transient walk-by false positives.
"""

import cv2
import numpy as np
import time
from typing import Dict, List, Tuple, Any, Optional

class ParticipantTracker:
    """
    Tracks participants across video frames using IoU & centroid matching.
    Maintains sustained presence counters to satisfy Scenario 1.
    """
    def __init__(self, sustained_sec_threshold: float = 10.0, max_disappeared: int = 40):
        self.sustained_sec_threshold = sustained_sec_threshold
        self.max_disappeared = max_disappeared
        
        # Internal state
        self.next_id = 1
        # id -> {'bbox': (x, y, w, h), 'centroid': (cx, cy), 'first_seen': float, 'last_seen': float,
        #        'active_frames': int, 'disappeared': int, 'sustained': bool, 'name': str}
        self.tracks: Dict[int, Dict[str, Any]] = {}
        
    def _compute_iou(self, boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]

        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return float(iou)

    def update(self, detected_boxes: List[Tuple[int, int, int, int]], current_timestamp: float, roster_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Updates participant tracks with newly detected bounding boxes.
        Returns active participants with sustained presence flags.
        """
        if not detected_boxes:
            # Mark disappeared
            for pid in list(self.tracks.keys()):
                self.tracks[pid]['disappeared'] += 1
                if self.tracks[pid]['disappeared'] > self.max_disappeared:
                    del self.tracks[pid]
            return []

        # If no existing tracks, register all
        if not self.tracks:
            for i, box in enumerate(detected_boxes):
                self._register(box, current_timestamp, roster_names)
            return self._get_active_tracks()

        # Match existing tracks to detected boxes using IoU & Distance
        track_ids = list(self.tracks.keys())
        track_boxes = [self.tracks[pid]['bbox'] for pid in track_ids]

        cost_matrix = np.zeros((len(track_ids), len(detected_boxes)), dtype=np.float32)
        for i, t_box in enumerate(track_boxes):
            for j, d_box in enumerate(detected_boxes):
                iou = self._compute_iou(t_box, d_box)
                cost_matrix[i, j] = 1.0 - iou

        # Simple greedy matching
        rows = cost_matrix.min(axis=1).argsort()
        cols = cost_matrix.argmin(axis=1)[rows]

        assigned_tracks = set()
        assigned_detections = set()

        for row, col in zip(rows, cols):
            if row in assigned_tracks or col in assigned_detections:
                continue
            if cost_matrix[row, col] > 0.8:  # IoU < 0.2
                continue

            pid = track_ids[row]
            box = detected_boxes[col]
            cx, cy = box[0] + box[2] // 2, box[1] + box[3] // 2
            
            self.tracks[pid]['bbox'] = box
            self.tracks[pid]['centroid'] = (cx, cy)
            self.tracks[pid]['last_seen'] = current_timestamp
            self.tracks[pid]['active_frames'] += 1
            self.tracks[pid]['disappeared'] = 0
            
            # Check sustained presence (Scenario 1)
            duration = self.tracks[pid]['last_seen'] - self.tracks[pid]['first_seen']
            if duration >= self.sustained_sec_threshold or self.tracks[pid]['active_frames'] >= 30:
                self.tracks[pid]['sustained'] = True

            assigned_tracks.add(row)
            assigned_detections.add(col)

        # Unmatched tracks
        unassigned_tracks = set(range(len(track_ids))) - assigned_tracks
        for row in unassigned_tracks:
            pid = track_ids[row]
            self.tracks[pid]['disappeared'] += 1
            if self.tracks[pid]['disappeared'] > self.max_disappeared:
                del self.tracks[pid]

        # Unmatched detections -> new participants
        unassigned_detections = set(range(len(detected_boxes))) - assigned_detections
        for col in unassigned_detections:
            self._register(detected_boxes[col], current_timestamp, roster_names)

        return self._get_active_tracks()

    def _register(self, bbox: Tuple[int, int, int, int], timestamp: float, roster_names: Optional[List[str]] = None):
        pid = self.next_id
        self.next_id += 1
        cx, cy = bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2
        
        name = f"Student {pid}"
        if roster_names and len(roster_names) >= pid:
            name = roster_names[pid - 1]

        self.tracks[pid] = {
            'id': pid,
            'name': name,
            'bbox': bbox,
            'centroid': (cx, cy),
            'first_seen': timestamp,
            'last_seen': timestamp,
            'active_frames': 1,
            'disappeared': 0,
            'sustained': False
        }

    def _get_active_tracks(self) -> List[Dict[str, Any]]:
        return [dict(data) for pid, data in self.tracks.items() if data['disappeared'] == 0]


class VisionDetector:
    """
    Offline Face & Participant Detection using OpenCV Haar / DNN.
    Guarantees privacy-preserving local offline execution.
    """
    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        
        # Load Haar cascades (natively packaged in OpenCV)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
        self.eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
        
        profile_cascade_path = cv2.data.haarcascades + 'haarcascade_profileface.xml'
        self.profile_cascade = cv2.CascadeClassifier(profile_cascade_path)

        self.tracker = ParticipantTracker()

    def detect_faces(self, frame_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detects frontal and profile faces in the frame.
        Returns list of (x, y, w, h) bounding boxes.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        # Frontal faces
        frontal_faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40)
        )
        
        boxes = [tuple(box) for box in frontal_faces]

        # Profile faces for detecting students turning their heads away
        profile_faces = self.profile_cascade.detectMultiScale(
            gray,
            scaleFactor=1.15,
            minNeighbors=4,
            minSize=(40, 40)
        )
        for pbox in profile_faces:
            # Avoid duplicate overlapping boxes
            is_dup = False
            for fbox in boxes:
                if self.tracker._compute_iou(fbox, tuple(pbox)) > 0.3:
                    is_dup = True
                    break
            if not is_dup:
                boxes.append(tuple(pbox))

        return boxes

    def process_frame(self, frame_bgr: np.ndarray, timestamp: float, roster_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Processes a frame: detects faces and updates participant tracks.
        """
        boxes = self.detect_faces(frame_bgr)
        active_participants = self.tracker.update(boxes, timestamp, roster_names)
        return active_participants
