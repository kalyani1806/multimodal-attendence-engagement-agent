"""
Configuration settings for Multimodal Attendance & Engagement Agent.
Follows privacy-preserving, 100% offline edge processing requirements.
"""

from pathlib import Path
import os

# Root directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = DATA_DIR / "logs"
ROSTERS_DIR = DATA_DIR / "rosters"
SAMPLES_DIR = DATA_DIR / "sample_sessions"

# Create required directories
for directory in [DATA_DIR, LOGS_DIR, ROSTERS_DIR, SAMPLES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Vision Configuration
VISION_CONFIG = {
    # Face Detection & Tracking
    "detection_confidence": 0.5,
    "max_faces": 20,
    "track_timeout_seconds": 3.0,
    
    # Attendance Sustained Presence Thresholds (Scenario 1)
    # Student must be present for at least sustained_presence_ratio of total session time
    # or sustained_window_seconds continuously to avoid transient walk-by false positives.
    "sustained_window_seconds": 15.0,  # Minimum seconds of continuous presence to mark as active
    "min_presence_ratio_for_attendance": 0.60, # 60% session presence = "Present", 40-60% = "Late/Partial", <40% = "Absent"
    
    # Head Pose Thresholds (Scenario 2)
    # Degrees deviation from center
    "yaw_threshold_degrees": 25.0,     # Looking left/right
    "pitch_threshold_degrees": 20.0,   # Looking up/down
    "roll_threshold_degrees": 20.0,    # Tilting head
    
    # Drowsiness & Inactivity (Scenario 2)
    "ear_threshold": 0.21,             # Eye Aspect Ratio below which eye is considered closed
    "ear_consecutive_frames": 15,      # Frames with closed eyes to trigger drowsiness alert (~1.5s at 10fps)
    "mar_yawn_threshold": 0.65,        # Mouth Aspect Ratio above which yawn is detected
    "inactivity_movement_threshold": 3.5, # Optical flow / pixel displacement threshold
    "inactivity_timeout_seconds": 8.0  # Prolonged zero movement duration
}

# Audio Configuration (EPIC 3)
AUDIO_CONFIG = {
    "sample_rate": 16000,
    "chunk_duration_seconds": 1.0,
    "energy_threshold": 0.015,         # RMS energy threshold for speech vs silence
    "silence_drop_seconds": 12.0,      # Silence interval triggering agent intervention suggestion
    "interaction_peak_volume": 0.08    # High audio energy threshold indicating lively discussion
}

# Multimodal Fusion Weights (EPIC 4)
FUSION_WEIGHTS = {
    "gaze_head_pose": 0.35,            # Attentive facing forward vs looking away
    "eye_openness": 0.25,              # Alertness vs drowsiness
    "body_stability": 0.15,            # Calm posture vs excessive restless movement / total freeze
    "audio_participation": 0.25        # Active verbal interaction / responsive session audio
}

# AI Agent Reasoning Thresholds (Scenario 3 & EPIC 5)
AGENT_CONFIG = {
    "attention_drop_threshold": 45.0,  # Composite engagement score below 45% = Attention Drop
    "attention_critical_threshold": 30.0, # Composite score below 30% = Critical Disengagement
    "interaction_peak_threshold": 75.0, # Composite score above 75% = High Engagement Peak
    "evaluation_window_seconds": 10.0, # Rolling evaluation window for trend analysis
    "max_history_records": 1000
}
