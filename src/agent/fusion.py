"""
EPIC 4: AI Implementation - Multimodal Signal Fusion Engine
Fuses Computer Vision (gaze, alertness, movement) and Audio (speech, energy)
into a unified engagement metric (0-100%).
"""

import numpy as np
from typing import Dict, Any, List
from config.settings import FUSION_WEIGHTS

class MultimodalFusionEngine:
    """
    Fuses vision-derived features with acoustic signals using calibrated weighting.
    Guarantees privacy-preserving offline mathematical inference.
    """
    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or FUSION_WEIGHTS
        # Normalize weights to ensure sum == 1.0
        total_w = sum(self.weights.values())
        self.weights = {k: v / total_w for k, v in self.weights.items()}

    def compute_student_engagement(
        self,
        gaze_score: float,
        alertness_score: float,
        stability_score: float,
        audio_score: float
    ) -> Dict[str, Any]:
        """
        Computes composite engagement score for an individual student.
        Returns score (0-100) and qualitative classification.
        """
        # Weighted linear fusion
        composite_score = (
            self.weights["gaze_head_pose"] * gaze_score +
            self.weights["eye_openness"] * alertness_score +
            self.weights["body_stability"] * stability_score +
            self.weights["audio_participation"] * audio_score
        ) * 100.0

        composite_score = max(0.0, min(100.0, composite_score))

        # Qualitative classification
        if composite_score >= 80.0:
            classification = "Highly Engaged"
            badge = "success"
        elif composite_score >= 60.0:
            classification = "Attentive & Active"
            badge = "primary"
        elif composite_score >= 45.0:
            classification = "Moderate Attention"
            badge = "warning"
        elif composite_score >= 30.0:
            classification = "Attention Drop"
            badge = "danger"
        else:
            classification = "Disengaged / Sleeping"
            badge = "critical"

        return {
            "engagement_score": float(np.round(composite_score, 1)),
            "classification": classification,
            "badge": badge,
            "components": {
                "gaze": float(np.round(gaze_score * 100, 1)),
                "alertness": float(np.round(alertness_score * 100, 1)),
                "stability": float(np.round(stability_score * 100, 1)),
                "audio": float(np.round(audio_score * 100, 1))
            }
        }

    def compute_classroom_engagement(
        self,
        student_engagement_list: List[Dict[str, Any]],
        audio_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Aggregates individual scores to generate classroom-level engagement dynamics.
        """
        if not student_engagement_list:
            # When no participants are visible
            baseline_score = 40.0 if audio_metrics.get("is_speech", False) else 25.0
            return {
                "class_engagement_score": baseline_score,
                "active_students_count": 0,
                "attention_drop_count": 0,
                "engaged_ratio": 0.0,
                "overall_mood": "No Participants Detected"
            }

        scores = [s["engagement_score"] for s in student_engagement_list]
        avg_score = float(np.mean(scores))
        active_count = len(scores)

        # Count how many students have attention drop (<45%)
        drop_count = sum(1 for s in scores if s < 45.0)
        engaged_ratio = (active_count - drop_count) / active_count if active_count > 0 else 0.0

        if avg_score >= 75.0:
            overall_mood = "High Interaction & Focus"
        elif avg_score >= 60.0:
            overall_mood = "Attentive Learning"
        elif avg_score >= 45.0:
            overall_mood = "Mixed Attention (Pacing Warning)"
        else:
            overall_mood = "Widespread Inattention Detected"

        return {
            "class_engagement_score": float(np.round(avg_score, 1)),
            "active_students_count": active_count,
            "attention_drop_count": drop_count,
            "engaged_ratio": float(np.round(engaged_ratio * 100, 1)),
            "overall_mood": overall_mood
        }
