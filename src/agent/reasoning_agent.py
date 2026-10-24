"""
EPIC 4: AI Implementation - Agent-Based Reasoning & AI-Driven Teaching Suggestions
Satisfies Scenario 3: Generates context-aware, assistive teaching recommendations
to help instructors adapt pacing, introduce interactive activities, and manage classroom dynamics.
"""

import time
import numpy as np
from typing import List, Dict, Any, Optional

class TeachingSuggestionAgent:
    """
    Offline AI Agent that observes session dynamics and provides assistive decision support
    to educators in real time.
    """
    def __init__(self, history_window_size: int = 60, suggestion_cooldown_sec: float = 15.0):
        self.history_window_size = history_window_size
        self.session_timeline: List[Dict[str, Any]] = []
        self.generated_suggestions: List[Dict[str, Any]] = []
        self.suggestion_cooldown_sec = suggestion_cooldown_sec  # Prevent overwhelming educator with alerts
        self.last_suggestion_time = -999.0  # Allow immediate first evaluation

    def record_timestep(
        self,
        timestamp: float,
        class_engagement: Dict[str, Any],
        audio_metrics: Dict[str, Any],
        student_details: List[Dict[str, Any]]
    ):
        """
        Records a timestamped snapshot of classroom multimodal state.
        """
        snapshot = {
            "timestamp": timestamp,
            "engagement_score": class_engagement.get("class_engagement_score", 50.0),
            "drop_count": class_engagement.get("attention_drop_count", 0),
            "active_students": class_engagement.get("active_students_count", 0),
            "audio_state": audio_metrics.get("audio_state", "Silence"),
            "is_speech": audio_metrics.get("is_speech", False),
            "is_interaction_peak": audio_metrics.get("is_interaction_peak", False),
            "silence_sec": audio_metrics.get("silence_duration_sec", 0.0),
            "drowsy_count": sum(1 for s in student_details if s.get("is_drowsy", False) or s.get("is_yawning", False)),
            "looking_away_count": sum(1 for s in student_details if "Looking" in s.get("orientation", "") and "Forward" not in s.get("orientation", ""))
        }
        self.session_timeline.append(snapshot)
        if len(self.session_timeline) > self.history_window_size:
            self.session_timeline.pop(0)

    def evaluate_and_suggest(self, current_time: float) -> Optional[Dict[str, Any]]:
        """
        Evaluates recent history trends and formulates context-aware pedagogical suggestions.
        """
        if len(self.session_timeline) < 3:
            return None

        # Check cooldown to avoid cognitive overload
        if (current_time - self.last_suggestion_time) < self.suggestion_cooldown_sec:
            return None

        latest = self.session_timeline[-1]
        recent_scores = [entry["engagement_score"] for entry in self.session_timeline[-5:]]
        avg_recent = float(np.mean(recent_scores))
        
        # Calculate trend (delta over last 5-10 timesteps)
        if len(self.session_timeline) >= 8:
            older_avg = float(np.mean([entry["engagement_score"] for entry in self.session_timeline[-10:-5]]))
            engagement_delta = avg_recent - older_avg
        else:
            engagement_delta = 0.0

        suggestion = None

        # Condition 1: Severe Attention Drop (Scenario 2 & 3)
        if avg_recent < 42.0 or engagement_delta <= -18.0:
            suggestion = {
                "id": f"sug_{len(self.generated_suggestions) + 1}",
                "timestamp": current_time,
                "category": "Pacing & Engagement Recovery",
                "priority": "HIGH",
                "badge_color": "#FF4B4B",
                "observation": f"Class engagement fell to {avg_recent:.1f}% ({engagement_delta:.1f}% change). {latest['looking_away_count']} students looking away.",
                "action": "Pause lecture monologue. Pose a provocative concept check question or launch a 60-second peer discussion.",
                "pedagogical_reason": "Prolonged passive reception leads to cognitive saturation. A switch to active retrieval restores working memory focus."
            }

        # Condition 2: Fatigue & Drowsiness Cluster
        elif latest["drowsy_count"] >= 2 or (latest["active_students"] > 0 and (latest["drowsy_count"] / latest["active_students"]) >= 0.3):
            suggestion = {
                "id": f"sug_{len(self.generated_suggestions) + 1}",
                "timestamp": current_time,
                "category": "Classroom Management & Fatigue",
                "priority": "HIGH",
                "badge_color": "#FFA726",
                "observation": f"Fatigue indicators detected in {latest['drowsy_count']} participants (eye closure / yawning / head drooping).",
                "action": "Introduce a 90-second physical energizer: request students to stand, stretch, take water, or change seat posture.",
                "pedagogical_reason": "Physical movement increases cerebral blood flow and immediately mitigates circadian fatigue slumps."
            }

        # Condition 3: Prolonged Silence Post-Explanation
        elif latest["silence_sec"] >= 15.0 and not latest["is_speech"]:
            suggestion = {
                "id": f"sug_{len(self.generated_suggestions) + 1}",
                "timestamp": current_time,
                "category": "Interactive Activity",
                "priority": "MEDIUM",
                "badge_color": "#29B6F6",
                "observation": f"Room quiet for {latest['silence_sec']:.0f} seconds following instruction block.",
                "action": "Prompt the room with a warm-up query: 'Who can rephrase the core trade-off we just analyzed in one sentence?'",
                "pedagogical_reason": "Low-stakes open prompts break hesitation barriers and stimulate collaborative dialogue."
            }

        # Condition 4: High Engagement / Interaction Peak
        elif latest["is_interaction_peak"] or avg_recent >= 78.0:
            suggestion = {
                "id": f"sug_{len(self.generated_suggestions) + 1}",
                "timestamp": current_time,
                "category": "Pacing & Discussion Momentum",
                "priority": "POSITIVE",
                "badge_color": "#66BB6A",
                "observation": f"Interaction peak detected! Engagement at {avg_recent:.1f}% with lively vocal exchange.",
                "action": "Synthesize student inputs on the board. Deepen the debate by asking: 'What is a counter-example to this logic?'",
                "pedagogical_reason": "Capitalizing on peer discourse momentum locks in deeper conceptual understanding."
            }

        # Condition 5: Inactivity / Pacing Bottleneck (Students looking down constantly, possible note-taking or confusion)
        elif latest["looking_away_count"] >= (latest["active_students"] * 0.5) and latest["active_students"] > 0:
            suggestion = {
                "id": f"sug_{len(self.generated_suggestions) + 1}",
                "timestamp": current_time,
                "category": "Pacing Adjustment",
                "priority": "MEDIUM",
                "badge_color": "#AB47BC",
                "observation": "Over 50% of active learners are looking down. Pacing may be too fast for simultaneous transcription.",
                "action": "Slow down delivery by 20%. Give learners 30 seconds of quiet time to finish jotting down key diagrams before proceeding.",
                "pedagogical_reason": "Cognitive split-attention effect occurs when students struggle to process spoken words while rapidly copying slides."
            }

        if suggestion:
            self.last_suggestion_time = current_time
            self.generated_suggestions.append(suggestion)

        return suggestion

    def get_all_suggestions(self) -> List[Dict[str, Any]]:
        return self.generated_suggestions
