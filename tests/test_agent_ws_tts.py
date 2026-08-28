"""Tests for the injectable TTS engine and sentence chunker."""
import sys

import numpy as np

from ascribe_link.agent_ws import tts as t
from tests.fake_voice import FakeTTS


class TestSentenceChunker:
    def test_basic_two_feeds(self):
        chunker = t.SentenceChunker()
        assert chunker.feed("Hello the") == []
        assert chunker.feed("re. How") == ["Hello there."]

    def test_continues_and_flush(self):
        chunker = t.SentenceChunker()
        chunker.feed("Hello the")
        chunker.feed("re. How")
        assert chunker.feed(" are you? I") == ["How are you?"]
        assert chunker.flush() == "I"

    def test_decimal_not_split(self):
        chunker = t.SentenceChunker()
        out = chunker.feed("3.14 is pi. ")
        assert out == ["3.14 is pi."]

    def test_flush_empty(self):
        chunker = t.SentenceChunker()
        assert chunker.flush() == ""


class TestKokoroTTS:
    def test_construction_does_not_import_kokoro(self):
        sys.modules.pop("kokoro_onnx", None)
        t.KokoroTTS()
        assert "kokoro_onnx" not in sys.modules

    def test_construction_does_not_create_cache_dir(self, tmp_path):
        cache_dir = tmp_path / "kokoro_cache"
        t.KokoroTTS(model_dir=str(cache_dir))
        assert not cache_dir.exists()

    def test_synthesize_raises_clear_error_without_dependency(self, monkeypatch):
        engine = t.KokoroTTS()
        monkeypatch.setitem(sys.modules, "kokoro_onnx", None)
        try:
            engine.synthesize("hello")
            assert False, "expected RuntimeError"
        except RuntimeError as exc:
            assert "pip install -e .[voice]" in str(exc)


class TestFakeTTS:
    def test_returns_expected_samples(self):
        engine = FakeTTS()
        out = engine.synthesize("hello there")
        assert out.dtype == np.float32
        assert len(out) == 2400
        assert np.abs(out).max() > 0.0
