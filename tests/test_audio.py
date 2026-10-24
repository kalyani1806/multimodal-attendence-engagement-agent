"""
Unit tests for Audio Module (EPIC 3).
"""

import unittest
import numpy as np
from src.audio import VoiceActivityDetector, AudioProcessor

class TestAudioModule(unittest.TestCase):
    def setUp(self):
        self.vad = VoiceActivityDetector(energy_threshold=0.015, sample_rate=16000)
        self.audio_proc = AudioProcessor()

    def test_silence_detection(self):
        """Pure quiet signal should register as silence and accumulate duration."""
        silence_chunk = np.zeros(16000, dtype=np.float32)
        res = self.vad.process_chunk(silence_chunk, duration_sec=1.0)
        
        self.assertFalse(res["is_speech"])
        self.assertEqual(res["silence_duration_sec"], 1.0)
        self.assertTrue(res["rms_energy"] < 0.001)

    def test_speech_detection(self):
        """Active sine wave signal should register as speech."""
        t = np.linspace(0, 1, 16000, endpoint=False)
        speech_chunk = 0.08 * np.sin(2 * np.pi * 300 * t).astype(np.float32)
        res = self.vad.process_chunk(speech_chunk, duration_sec=1.0)
        
        self.assertTrue(res["is_speech"])
        self.assertEqual(res["silence_duration_sec"], 0.0)

    def test_synthetic_audio_generator(self):
        """Processor generates valid waveforms for simulation."""
        chunk = self.audio_proc.generate_synthetic_chunk("lecture", 10.0)
        self.assertEqual(len(chunk), 16000)
        self.assertIsInstance(chunk, np.ndarray)

if __name__ == "__main__":
    unittest.main()
