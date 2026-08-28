"""Shared fake voice engines for tests (no faster-whisper/kokoro imports)."""
import numpy as np


class FakeSTT:
    """Fake STT engine: reports sample count instead of real transcription."""

    def transcribe(self, audio_16k: np.ndarray) -> str:
        return f"FAKE({len(audio_16k)} samples)"
