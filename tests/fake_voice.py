"""Shared fake voice engines for tests (no faster-whisper/kokoro imports)."""
import numpy as np


class FakeSTT:
    """Fake STT engine: reports sample count instead of real transcription."""

    def transcribe(self, audio_16k: np.ndarray) -> str:
        return f"FAKE({len(audio_16k)} samples)"


class FakeTTS:
    """Fake TTS engine: returns a 0.1 s 440 Hz sine at 24 kHz."""

    RATE = 24000

    def synthesize(self, text: str) -> np.ndarray:
        n = int(self.RATE * 0.1)
        t = np.arange(n, dtype=np.float32) / self.RATE
        return (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
