"""
EPIC 3: Audio Module Development - Audio Stream & File Processing
Captures live audio from microphone or processes audio tracks from classroom recordings.
"""

import numpy as np
import time
from typing import Dict, Any, Optional
from .vad import VoiceActivityDetector

class AudioProcessor:
    """
    Manages live microphone streams, recorded audio file reading, and offline synthetic test audio.
    """
    def __init__(self, sample_rate: int = 16000, chunk_duration_sec: float = 1.0):
        self.sample_rate = sample_rate
        self.chunk_duration_sec = chunk_duration_sec
        self.chunk_size = int(sample_rate * chunk_duration_sec)
        self.vad = VoiceActivityDetector(sample_rate=sample_rate)
        
        self.mic_stream = None
        self.is_mic_running = False

    def start_mic_stream(self) -> bool:
        """
        Attempts to initialize a live microphone input stream via sounddevice.
        Falls back gracefully if no microphone hardware or permission is available.
        """
        try:
            import sounddevice as sd
            self.mic_stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                blocksize=self.chunk_size,
                dtype='float32'
            )
            self.mic_stream.start()
            self.is_mic_running = True
            return True
        except Exception as e:
            self.is_mic_running = False
            return False

    def read_mic_chunk(self) -> Optional[np.ndarray]:
        """
        Reads one audio chunk from the active microphone stream.
        """
        if not self.is_mic_running or self.mic_stream is None:
            return None
        try:
            chunk, overflowed = self.mic_stream.read(self.chunk_size)
            return chunk.flatten()
        except Exception:
            return None

    def stop_mic_stream(self):
        """Stops the live microphone stream."""
        if self.mic_stream is not None:
            try:
                self.mic_stream.stop()
                self.mic_stream.close()
            except Exception:
                pass
            self.mic_stream = None
            self.is_mic_running = False

    def generate_synthetic_chunk(self, scenario: str = "lecture", session_time_sec: float = 0.0) -> np.ndarray:
        """
        Generates realistic classroom audio waveforms for offline demonstration:
        - 'lecture': steady instructor voice with natural pauses
        - 'discussion': lively multi-voice exchanges (interaction peaks)
        - 'silence': quiet classroom thinking / problem solving
        - 'fidget': ambient background shuffle
        """
        t = np.linspace(0, self.chunk_duration_sec, self.chunk_size, endpoint=False)
        np.random.seed(int(session_time_sec * 10) % 100000)

        if scenario == "lecture":
            # Formant-like frequencies for speech (200Hz fundamental + harmonics) + natural noise
            speech = 0.04 * np.sin(2 * np.pi * 220 * t) + 0.02 * np.sin(2 * np.pi * 440 * t)
            noise = np.random.normal(0, 0.005, self.chunk_size)
            # Periodic pauses
            pause = 0.2 if (int(session_time_sec) % 6 == 0) else 1.0
            return (speech * pause + noise).astype(np.float32)

        elif scenario == "discussion":
            # Higher energy, mixed frequencies representing lively questions/responses
            speech = 0.09 * np.sin(2 * np.pi * 320 * t) + 0.05 * np.sin(2 * np.pi * 600 * t)
            noise = np.random.normal(0, 0.012, self.chunk_size)
            return (speech + noise).astype(np.float32)

        elif scenario == "silence":
            # Pure quiet room background noise
            noise = np.random.normal(0, 0.003, self.chunk_size)
            return noise.astype(np.float32)

        else: # ambient
            noise = np.random.normal(0, 0.008, self.chunk_size)
            return noise.astype(np.float32)

    def process_time_slice(self, raw_audio: Optional[np.ndarray], fallback_scenario: str = "lecture", session_time_sec: float = 0.0) -> Dict[str, Any]:
        """
        Processes real audio chunk or generates simulated chunk if real chunk is absent.
        """
        if raw_audio is None or len(raw_audio) == 0:
            raw_audio = self.generate_synthetic_chunk(scenario=fallback_scenario, session_time_sec=session_time_sec)

        return self.vad.process_chunk(raw_audio, duration_sec=self.chunk_duration_sec)
