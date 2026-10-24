"""
Unit tests for AI Implementation & Reasoning Agent (EPIC 4).
"""

import unittest
from src.agent import MultimodalFusionEngine, TeachingSuggestionAgent

class TestAgentModule(unittest.TestCase):
    def setUp(self):
        self.fusion = MultimodalFusionEngine()
        self.agent = TeachingSuggestionAgent(history_window_size=30)

    def test_multimodal_fusion_range(self):
        """Engagement score must stay strictly between 0 and 100."""
        res_high = self.fusion.compute_student_engagement(1.0, 1.0, 1.0, 1.0)
        self.assertAlmostEqual(res_high["engagement_score"], 100.0, places=1)
        self.assertEqual(res_high["classification"], "Highly Engaged")

        res_low = self.fusion.compute_student_engagement(0.0, 0.0, 0.0, 0.0)
        self.assertAlmostEqual(res_low["engagement_score"], 0.0, places=1)
        self.assertIn("Disengaged", res_low["classification"])

    def test_pedagogical_suggestions_generation(self):
        """Agent should trigger attention recovery suggestion on attention drops."""
        # Feed 5 low engagement timesteps
        for t in range(5):
            self.agent.record_timestep(
                timestamp=float(t),
                class_engagement={"class_engagement_score": 35.0, "attention_drop_count": 3, "active_students_count": 4},
                audio_metrics={"audio_state": "Silence", "is_speech": False, "silence_duration_sec": 16.0, "is_interaction_peak": False},
                student_details=[{"is_drowsy": False, "orientation": "Looking Down"}]
            )
        
        sug = self.agent.evaluate_and_suggest(current_time=5.0)
        self.assertIsNotNone(sug)
        self.assertIn("Pacing", sug["category"])
        self.assertEqual(sug["priority"], "HIGH")

if __name__ == "__main__":
    unittest.main()
