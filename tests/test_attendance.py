"""
Unit tests for Sustained Attendance Tracking & Analytics (EPIC 5).
"""

import unittest
from pathlib import Path
from src.analytics import AttendanceTracker, SessionMetricsManager
from config.settings import ROSTERS_DIR

class TestAttendanceAnalytics(unittest.TestCase):
    def setUp(self):
        self.roster_path = ROSTERS_DIR / "student_roster.json"
        self.tracker = AttendanceTracker(roster_path=str(self.roster_path), min_sustained_seconds=10.0)
        self.metrics = SessionMetricsManager()

    def test_sustained_presence_walkby_rejection(self):
        """Walk-by participant with only 3 seconds presence must be marked Absent."""
        # Participant 1 present for 3 seconds only (dt=1)
        for t in range(3):
            self.tracker.update_presence([{"id": 1, "name": "Aarav Sharma"}], float(t), dt=1.0)
            
        df = self.tracker.finalize_session(total_session_sec=60.0)
        p1_row = df[df["Student ID"] == "STU-001"].iloc[0]
        self.assertIn("ABSENT", p1_row["Attendance Status"])

    def test_sustained_presence_marked_present(self):
        """Participant present for 40 seconds out of 60 seconds should be marked PRESENT."""
        for t in range(40):
            self.tracker.update_presence([{"id": 2, "name": "Priya Patel"}], float(t), dt=1.0)
            
        df = self.tracker.finalize_session(total_session_sec=60.0)
        p2_row = df[df["Student ID"] == "STU-002"].iloc[0]
        self.assertEqual(p2_row["Attendance Status"], "PRESENT")

    def test_session_metrics_attention_drops(self):
        """Metrics manager records attention drops correctly."""
        self.metrics.record_step(5.0, 38.0, 4, 0.01, "Silence", 1)
        summary = self.metrics.compute_summary_analytics()
        self.assertEqual(summary["total_attention_drops"], 1)

if __name__ == "__main__":
    unittest.main()
