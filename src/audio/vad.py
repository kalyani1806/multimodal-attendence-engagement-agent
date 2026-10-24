"""
EPIC 3: Audio Module Development - Voice Activity Detection & Signal Dynamics
Analyzes RMS energy, vocal activity, and classroom audio interaction dynamics.
"""

import numpy as np
from typing import Dict, Any, List

class VoiceActivityDetector:
    """
    Offline Voice Activity Detector (VAD) analyzing energy, spectral characteristics,
    and classifying speech vs ambient noise vs silence.
    """
    def __init__(self, energy_threshold: float = 0.02, sample_rate: int = 16000):
        self.energy_threshold = energy_threshold
        self.sample_rate = sample_rate
        self.silence_seconds_counter = 0.0

    def calculate_rms_energy(self, audio_chunk: np.ndarray) -> float:
        """
        Calculates Root-Mean-Square (RMS) energy of audio signal.
        """
        if audio_chunk is None or len(audio_chunk) == 0:
            return 0.0
        audio_chunk = np.asarray(audio_chunk, dtype=np.float32)
        # Handle stereo by taking mean of channels
        if len(audio_chunk.shape) > 1:
            audio_chunk = np.mean(audio_chunk, axis=1)
        rms = np.sqrt(np.mean(audio_chunk ** 2) + 1e-9)
        return float(rms)

    def calculate_zero_crossing_rate(self, audio_chunk: np.ndarray) -> float:
        """
        Calculates Zero Crossing Rate (ZCR) useful for unvoiced speech / noise discrimination.
        """
        if audio_chunk is None or len(audio_chunk) < 2:
            return 0.0
        if len(audio_chunk.shape) > 1:
            audio_chunk = np.mean(audio_chunk, axis=1)
        zcr = np.mean(np.abs(np.diff(np.sign(audio_chunk)))) / 2.0
        return float(zcr)

    def process_chunk(self, audio_chunk: np.ndarray, duration_sec: float = 1.0) -> Dict[str, Any]:
        """
        Processes audio buffer chunk and returns speech classification and engagement metrics.
        """
        rms = self.calculate_rms_energy(audio_chunk)
        zcr = self.calculate_zero_crossing_rate(audio_chunk)

        is_speech = rms > self.energy_threshold
        is_interaction_peak = rms > (self.energy_threshold * 3.5)

        if not is_speech:
            self.silence_seconds_counter += duration_sec
            audio_state = "Silence / Quiet Room" if rms < (self.energy_threshold * 0.4) else "Ambient Noise"
            participation_score = 0.35  # Baseline listening score
        else:
            self.silence_seconds_counter = 0.0
            if is_interaction_peak:
                audio_state = "Interaction Peak (Active Q&A / Discussion)"
                participation_score = 1.0
            else:
                audio_state = "Lecture / Active Voice"
                participation_score = 0.85

        return {
            "rms_energy": float(np.round(rms, 4)),
            "zero_crossing_rate": float(np.round(zcr, 4)),
            "is_speech": is_speech,
            "is_interaction_peak": is_interaction_peak,
            "silence_duration_sec": float(np.round(self.silence_seconds_counter, 1)),
            "audio_state": audio_state,
            "participation_score": float(np.round(participation_score, 2))
        }
