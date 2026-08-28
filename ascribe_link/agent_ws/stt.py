"""Speech-to-text engine (injectable) and utterance accumulator.

``faster_whisper`` is imported lazily inside ``_ensure_model`` so this module
(and the server) imports cleanly without the ``voice`` extra installed.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

from . import audio


class STTEngine(Protocol):
    """Blocking transcription engine. Called via ``asyncio.to_thread``."""

    def transcribe(self, audio_16k: np.ndarray) -> str:
        """Transcribe float32 mono 16 kHz audio to text."""
        ...


class FasterWhisperSTT:
    """STT engine backed by faster-whisper, lazily loaded on first use."""

    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None

    def warmup(self) -> None:
        """Download/load the model now (blocking) so the first real
        conversation turn doesn't pay the cost mid-dialogue."""
        self._ensure_model()

    def _ensure_model(self):
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise RuntimeError(
                    "faster-whisper is not installed; run "
                    "'pip install -e .[voice]' to enable speech-to-text."
                ) from exc
            self._model = WhisperModel(
                self.model_size, device=self.device, compute_type=self.compute_type
            )
        return self._model

    def transcribe(self, audio_16k: np.ndarray) -> str:
        model = self._ensure_model()
        segments, _info = model.transcribe(audio_16k, vad_filter=True)
        return "".join(segment.text for segment in segments).strip()


class UtteranceBuffer:
    """Accumulates incoming audio chunks into a single 16 kHz utterance."""

    RATE = 16000

    def __init__(self, rate_hint: int = 48000) -> None:
        self.rate_hint = rate_hint
        self._chunks: list[np.ndarray] = []
        self._samples = 0

    def add(self, payload: bytes, rate: int) -> None:
        arr = audio.pcm16_to_float32(payload)
        arr16k = audio.resample(arr, rate, self.RATE)
        self._chunks.append(arr16k)
        self._samples += len(arr16k)

    @property
    def duration_s(self) -> float:
        return self._samples / self.RATE

    def should_finalize(self, silence_s: float = 2.0, max_s: float = 60.0) -> bool:
        if self.duration_s < 0.5:
            return False
        if self.duration_s >= max_s:
            return True
        combined = np.concatenate(self._chunks) if self._chunks else np.array(
            [], dtype=np.float32
        )
        return audio.trailing_silence_s(combined, self.RATE) >= silence_s

    def take(self) -> np.ndarray:
        combined = (
            np.concatenate(self._chunks) if self._chunks else np.array([], dtype=np.float32)
        )
        self._chunks = []
        self._samples = 0
        return combined
