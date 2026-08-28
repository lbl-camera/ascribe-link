"""Text-to-speech engine (injectable) and streaming sentence chunker.

``kokoro_onnx`` is imported lazily inside ``_ensure_model`` so this module
(and the server) imports cleanly without the ``voice`` extra installed.
"""
from __future__ import annotations

import logging
import os
import urllib.request
from typing import Protocol

import numpy as np

from . import audio

log = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = os.path.expanduser("~/.cache/ascribe-link/kokoro")

KOKORO_MODEL_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
    "model-files-v1.0/kokoro-v1.0.onnx"
)
KOKORO_VOICES_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
    "model-files-v1.0/voices-v1.0.bin"
)

KOKORO_RATE = 24000


class TTSEngine(Protocol):
    """Blocking synthesis engine. Called via ``asyncio.to_thread``."""

    def synthesize(self, text: str) -> np.ndarray:
        """Synthesize text to float32 mono 24 kHz audio."""
        ...


class KokoroTTS:
    """TTS engine backed by kokoro-onnx, lazily loaded on first use."""

    def __init__(self, voice: str = "af_heart", model_dir: str | None = None) -> None:
        self.voice = voice
        self.model_dir = model_dir or DEFAULT_CACHE_DIR
        self._kokoro = None

    def warmup(self) -> None:
        """Download/load the model now (blocking) so the first real
        conversation turn doesn't pay the cost mid-dialogue."""
        self._ensure_model()

    def _ensure_model(self):
        if self._kokoro is None:
            try:
                from kokoro_onnx import Kokoro
            except ImportError as exc:
                raise RuntimeError(
                    "kokoro-onnx is not installed; run "
                    "'pip install -e .[voice]' to enable text-to-speech."
                ) from exc

            os.makedirs(self.model_dir, exist_ok=True)
            model_path = os.path.join(self.model_dir, "kokoro-v1.0.onnx")
            voices_path = os.path.join(self.model_dir, "voices-v1.0.bin")

            if not os.path.exists(model_path):
                log.info("Downloading kokoro model to %s", model_path)
                urllib.request.urlretrieve(KOKORO_MODEL_URL, model_path)
            if not os.path.exists(voices_path):
                log.info("Downloading kokoro voices to %s", voices_path)
                urllib.request.urlretrieve(KOKORO_VOICES_URL, voices_path)

            self._kokoro = Kokoro(model_path, voices_path)
        return self._kokoro

    def synthesize(self, text: str) -> np.ndarray:
        kokoro = self._ensure_model()
        samples, sample_rate = kokoro.create(text, voice=self.voice)
        samples = np.asarray(samples, dtype=np.float32)
        if sample_rate != KOKORO_RATE:
            samples = audio.resample(samples, sample_rate, KOKORO_RATE)
        return samples


_TERMINATORS = (".", "!", "?", "\n")
_MIN_CHARS_SINCE_BREAK = 3


class SentenceChunker:
    """Accumulates streamed text deltas and emits complete sentences.

    A break occurs at a terminator (``.``, ``!``, ``?``, ``\\n``) followed by
    a space or end-of-buffer, provided at least ``_MIN_CHARS_SINCE_BREAK``
    characters have accumulated since the last break.
    """

    def __init__(self) -> None:
        self._buf = ""

    def feed(self, text_delta: str) -> list[str]:
        self._buf += text_delta
        sentences: list[str] = []

        while True:
            break_at = None
            for i, ch in enumerate(self._buf):
                if ch in _TERMINATORS:
                    is_end = i == len(self._buf) - 1
                    is_space_after = not is_end and self._buf[i + 1] == " "
                    if (is_end or is_space_after) and (i + 1) >= _MIN_CHARS_SINCE_BREAK:
                        break_at = i
                        break
            if break_at is None:
                break
            # Don't emit on end-of-buffer terminator unless we know no more
            # text is coming right after it (i.e. it's truly the last char
            # fed so far) -- since feed() is called per-delta, end-of-buffer
            # is a valid break point (we can't look ahead further).
            sentence = self._buf[: break_at + 1].strip()
            rest = self._buf[break_at + 1 :].lstrip(" ")
            if sentence:
                sentences.append(sentence)
            self._buf = rest

        return sentences

    def flush(self) -> str:
        remainder = self._buf.strip()
        self._buf = ""
        return remainder
