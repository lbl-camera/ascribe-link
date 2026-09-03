"""Text-to-speech engine (injectable) and streaming sentence chunker.

``kokoro_onnx`` is imported lazily inside ``_ensure_model`` so this module
(and the server) imports cleanly without the ``voice`` extra installed.
"""
from __future__ import annotations

import logging
import re
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
            # kokoro-onnx phonemizes through phonemizer's espeak backend,
            # which warns "words count mismatch on N% of the lines" whenever
            # espeak merges or splits tokens (numbers, hyphens, symbols such
            # as "[::4]"). It is informational -- the audio is unaffected --
            # and it fires on most sentences, so drop it below WARNING.
            logging.getLogger("phonemizer").setLevel(logging.ERROR)
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

# --- speakable text -----------------------------------------------------------
# The agent writes markdown; Kokoro reads it literally ("asterisk asterisk
# Done asterisk asterisk"). Strip formatting and turn the symbols people say
# differently from how they're written into words, per sentence.

_RE_FENCE = re.compile(r"```[\w-]*")                     # fence markers (content kept)
_RE_INLINE_CODE = re.compile(r"`([^`]*)`")
_RE_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")           # [text](url) -> text
_RE_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")           # ![alt](url) -> ""
_RE_URL = re.compile(r"https?://\S+|www\.\S+")
_RE_HEADER = re.compile(r"^\s{0,3}#{1,6}\s+", re.MULTILINE)
_RE_BULLET = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s+", re.MULTILINE)
_RE_BLOCKQUOTE = re.compile(r"^\s*>\s?", re.MULTILINE)
_RE_HRULE = re.compile(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$", re.MULTILINE)
_RE_EMPHASIS = re.compile(r"(\*{1,3}|_{1,3}|~~)(?=\S)(.+?)(?<=\S)\1")
_RE_DIMENSIONS = re.compile(r"(?<=\d)\s*[x×]\s*(?=\d)")   # 251x131 -> 251 by 131
_RE_SNAKE = re.compile(r"(?<=\w)_(?=\w)")                 # plant_sub -> plant sub
_RE_ARROW = re.compile(r"\s*(?:->|=>|→|⇒)\s*")
_RE_EMOJI = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F900-\U0001F9FF️‍]"
)
_RE_LEFTOVER_SYMBOLS = re.compile(r"[*_`#|\\<>{}\[\]^~]")
_RE_SPACE = re.compile(r"[ \t]+")
_RE_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?])")


def speakable_text(text: str) -> str:
    """Rewrite one sentence of agent markdown into something worth voicing.

    Formatting is removed (bold/italic/headers/bullets/code/links); symbols
    that are read aloud differently from how they're typed become words
    (``251x131`` -> ``251 by 131``, ``a -> b`` -> ``a to b``, ``&`` -> ``and``,
    ``plant_sub`` -> ``plant sub``); emoji and stray markup characters are
    dropped. Returns "" when nothing speakable remains (e.g. a bare ``---``
    or a fence line), which the caller should skip.
    """
    s = text
    s = _RE_IMAGE.sub("", s)
    s = _RE_LINK.sub(r"\1", s)
    s = _RE_URL.sub("link", s)
    s = _RE_FENCE.sub("", s)
    s = _RE_INLINE_CODE.sub(r"\1", s)
    s = _RE_HRULE.sub("", s)
    s = _RE_HEADER.sub("", s)
    s = _RE_BLOCKQUOTE.sub("", s)
    s = _RE_BULLET.sub("", s)
    # snake_case before emphasis, so `plant_sub` ... `mesh_out` can't be
    # mistaken for an _italic_ span.
    s = _RE_SNAKE.sub(" ", s)
    for _ in range(3):  # nested emphasis: ***x***, **_x_**
        s = _RE_EMPHASIS.sub(r"\2", s)
    s = _RE_DIMENSIONS.sub(" by ", s)
    s = _RE_ARROW.sub(" to ", s)
    s = s.replace("&", " and ")
    s = _RE_EMOJI.sub("", s)
    s = _RE_LEFTOVER_SYMBOLS.sub(" ", s)
    s = _RE_SPACE.sub(" ", s)
    s = _RE_SPACE_BEFORE_PUNCT.sub(r"\1", s)
    s = s.strip()
    # Nothing but punctuation left (e.g. a fence or rule line) -> nothing to say.
    if not re.search(r"[A-Za-z0-9À-￿]", s):
        return ""
    return s


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
