"""Tests for STT engine (injectable) and utterance accumulator."""
import sys

import numpy as np
import pytest

from ascribe_link.agent_ws import stt
from tests.fake_voice import FakeSTT


def _tone(rate: int, duration_s: float, freq: float = 440.0) -> np.ndarray:
    t = np.arange(int(rate * duration_s)) / rate
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


class TestUtteranceBuffer:
    def test_duration_after_half_second_tone(self):
        buf = stt.UtteranceBuffer(rate_hint=48000)
        tone = _tone(48000, 0.5)
        from ascribe_link.agent_ws.audio import float32_to_pcm16

        buf.add(float32_to_pcm16(tone), 48000)
        assert abs(buf.duration_s - 0.5) <= 0.02

    def test_should_finalize_true_after_trailing_silence(self):
        buf = stt.UtteranceBuffer(rate_hint=48000)
        from ascribe_link.agent_ws.audio import float32_to_pcm16

        tone = _tone(48000, 0.5)
        silence = np.zeros(int(48000 * 2.1), dtype=np.float32)
        buf.add(float32_to_pcm16(tone), 48000)
        buf.add(float32_to_pcm16(silence), 48000)
        assert buf.should_finalize() is True

    def test_should_finalize_false_for_tone_only(self):
        buf = stt.UtteranceBuffer(rate_hint=48000)
        from ascribe_link.agent_ws.audio import float32_to_pcm16

        tone = _tone(48000, 0.5)
        buf.add(float32_to_pcm16(tone), 48000)
        assert buf.should_finalize() is False

    def test_take_resets_buffer(self):
        buf = stt.UtteranceBuffer(rate_hint=48000)
        from ascribe_link.agent_ws.audio import float32_to_pcm16

        tone = _tone(48000, 0.5)
        buf.add(float32_to_pcm16(tone), 48000)
        result = buf.take()
        assert len(result) > 0
        assert buf.duration_s == 0


class TestFasterWhisperSTT:
    def test_construction_does_not_import_faster_whisper(self):
        sys.modules.pop("faster_whisper", None)
        stt.FasterWhisperSTT()
        assert "faster_whisper" not in sys.modules

    def test_transcribe_without_package_raises_runtime_error(self, monkeypatch):
        engine = stt.FasterWhisperSTT()
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "faster_whisper":
                raise ImportError("no module named faster_whisper")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(RuntimeError, match=r"pip install -e \.\[voice\]"):
            engine.transcribe(np.zeros(100, dtype=np.float32))


class TestFakeSTT:
    def test_fake_stt_reports_sample_count(self):
        fake = FakeSTT()
        audio = np.zeros(1234, dtype=np.float32)
        assert fake.transcribe(audio) == "FAKE(1234 samples)"
