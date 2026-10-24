"""
EPIC 5: Analytics & Recommendation Engine - Session Dynamics & Timeline Metrics
Satisfies Scenario 2: Aggregates time-series engagement trends, attention drops, and interaction peaks.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional

class SessionMetricsManager:
    """
    Maintains session time-series data and extracts macro dynamics:
    - Continuous engagement trends
    - Attention drop intervals
    - Interaction peak timestamps
    - Pacing alerts
    """
    def __init__(self):
        self.time_series: List[Dict[str, Any]] = []
        self.attention_drop_events: List[Dict[str, Any]] = []
        self.interaction_peak_events: List[Dict[str, Any]] = []

    def record_step(
        self,
        time_elapsed_sec: float,
        engagement_score: float,
        active_participants: int,
        audio_energy: float,
        audio_state: str,
        drowsy_count: int,
        suggestion_active: Optional[str] = None
    ):
        """
        Records a discrete time-step into the analytics buffer.
        """
        entry = {
            "time_sec": time_elapsed_sec,
            "engagement_score": float(np.round(engagement_score, 1)),
            "active_participants": active_participants,
            "audio_energy": float(np.round(audio_energy, 4)),
            "audio_state": audio_state,
            "drowsy_count": drowsy_count,
            "suggestion": suggestion_active or ""
        }
        self.time_series.append(entry)

        # Detect Attention Drop Event (Scenario 2)
        if engagement_score < 45.0:
            if not self.attention_drop_events or (time_elapsed_sec - self.attention_drop_events[-1]["time_sec"]) > 10.0:
                self.attention_drop_events.append({
                    "time_sec": time_elapsed_sec,
                    "engagement_score": engagement_score,
                    "reason": "Gaze deflection / prolonged low response"
                })

        # Detect Interaction Peak Event (Scenario 2)
        if engagement_score > 75.0 and audio_energy > 0.04:
            if not self.interaction_peak_events or (time_elapsed_sec - self.interaction_peak_events[-1]["time_sec"]) > 10.0:
                self.interaction_peak_events.append({
                    "time_sec": time_elapsed_sec,
                    "engagement_score": engagement_score,
                    "reason": "Spontaneous Q&A / lively peer discussion"
                })

    def get_dataframe(self) -> pd.DataFrame:
        """Returns time series as Pandas DataFrame for plotting."""
        if not self.time_series:
            return pd.DataFrame(columns=["time_sec", "engagement_score", "active_participants", "audio_energy", "audio_state"])
        return pd.DataFrame(self.time_series)

    def compute_summary_analytics(self) -> Dict[str, Any]:
        """Calculates macro session indices."""
        if not self.time_series:
            return {
                "mean_engagement": 0.0,
                "peak_engagement": 0.0,
                "min_engagement": 0.0,
                "total_attention_drops": 0,
                "total_interaction_peaks": 0,
                "pacing_index": "N/A"
            }

        scores = [t["engagement_score"] for t in self.time_series]
        mean_eng = float(np.mean(scores))
        peak_eng = float(np.max(scores))
        min_eng = float(np.min(scores))

        # Pacing Index Assessment
        drop_count = len(self.attention_drop_events)
        if drop_count == 0:
            pacing = "Optimal Pacing (Sustained Focus)"
        elif drop_count <= 2:
            pacing = "Good Pacing (Minor Fluctuations)"
        else:
            pacing = "Inconsistent Pacing (Frequent Attention Lapses)"

        return {
            "mean_engagement": float(np.round(mean_eng, 1)),
            "peak_engagement": float(np.round(peak_eng, 1)),
            "min_engagement": float(np.round(min_eng, 1)),
            "total_attention_drops": drop_count,
            "total_interaction_peaks": len(self.interaction_peak_events),
            "pacing_index": pacing
        }
