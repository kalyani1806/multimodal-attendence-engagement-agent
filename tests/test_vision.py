"""
Unit tests for Vision Module (EPIC 2).
"""

import unittest
import numpy as np
from src.vision import VisionDetector, ParticipantTracker, HeadPoseEstimator, DrowsinessDetector, MovementAnalyzer

class TestVisionModule(unittest.TestCase):
    def setUp(self):
        self.tracker = ParticipantTracker(sustained_sec_threshold=5.0)
        self.pose_estimator = HeadPoseEstimator()
        self.drowsiness_detector = DrowsinessDetector(consecutive_frames_threshold=5)
        self.movement_analyzer = MovementAnalyzer()

    def test_participant_tracker_sustained_presence(self):
        """Tests that sustained presence requires cumulative time and rejects transient walk-bys."""
        box = (100, 100, 50, 50)
        
        # Step 1: First appearance (t=0s)
        tracks = self.tracker.update([box], current_timestamp=0.0)
        self.assertEqual(len(tracks), 1)
        self.assertFalse(tracks[0]['sustained'])

        # Step 2: Still present at t=2s (below 5s threshold)
        tracks = self.tracker.update([box], current_timestamp=2.0)
        self.assertFalse(tracks[0]['sustained'])

        # Step 3: Sustained at t=6s (above 5s threshold)
        tracks = self.tracker.update([box], current_timestamp=6.0)
        self.assertTrue(tracks[0]['sustained'])

    def test_head_pose_estimation(self):
        """Tests head pose angle calculation on synthetic crop."""
        synthetic_face = np.ones((100, 100, 3), dtype=np.uint8) * 128
        pose = self.pose_estimator.estimate_pose_from_crop(
            synthetic_face,
            (100, 100, 100, 100),
            (480, 640)
        )
        self.assertIn("yaw", pose)
        self.assertIn("pitch", pose)
        self.assertIn("orientation", pose)
        self.assertIn("gaze_score", pose)
        self.assertTrue(0.0 <= pose["gaze_score"] <= 1.0)

    def test_drowsiness_detection(self):
        """Tests alertness score and drowsiness tracking."""
        synthetic_face = np.ones((80, 80, 3), dtype=np.uint8) * 128
        # Feed 6 frames with no eyes to trigger threshold
        for _ in range(6):
            res = self.drowsiness_detector.analyze(synthetic_face, participant_id=1)
        
        self.assertTrue(res["is_drowsy"])
        self.assertTrue(res["alertness_score"] < 0.6)

    def test_movement_analyzer(self):
        """Tests movement stability and motion magnitude."""
        frame1 = np.zeros((80, 80, 3), dtype=np.uint8)
        res1 = self.movement_analyzer.analyze_movement(frame1, participant_id=1, current_timestamp=0.0)
        self.assertIn("stability_score", res1)

        # Subtle change
        frame2 = np.ones((80, 80, 3), dtype=np.uint8) * 20
        res2 = self.movement_analyzer.analyze_movement(frame2, participant_id=1, current_timestamp=1.0)
        self.assertTrue(res2["motion_magnitude"] > 0)

if __name__ == "__main__":
    unittest.main()
