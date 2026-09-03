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


def test_silence_only_utterance_gets_grace_period():
    from ascribe_link.agent_ws.stt import UtteranceBuffer
    import numpy as np
    """Mic warm-up streams zeros; an all-silence buffer must NOT finalize at
    the normal 2s endpoint (regression: first Talk press returned '(silence)'
    before the user's first word arrived)."""
    from ascribe_link.agent_ws.audio import float32_to_pcm16

    buf = UtteranceBuffer()
    silence = np.zeros(16000 * 3, dtype=np.float32)  # 3s of zeros @16k
    buf.add(float32_to_pcm16(silence), 16000)
    assert not buf.should_finalize()  # 3s all-silence: still waiting

    buf.add(float32_to_pcm16(np.zeros(16000 * 6, dtype=np.float32)), 16000)
    assert buf.should_finalize()  # 9s all-silence: grace period exceeded


def test_voiced_then_silence_still_finalizes_at_2s():
    from ascribe_link.agent_ws.stt import UtteranceBuffer
    import numpy as np
    from ascribe_link.agent_ws.audio import float32_to_pcm16

    buf = UtteranceBuffer()
    t = np.arange(16000, dtype=np.float32) / 16000.0
    tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    buf.add(float32_to_pcm16(tone), 16000)
    buf.add(float32_to_pcm16(np.zeros(16000 * 2 + 1600, dtype=np.float32)), 16000)
    assert buf.should_finalize()


def test_end_silence_is_one_second():
    """Endpointing after speech: 1.0 s of trailing silence (END_SILENCE_S),
    not 2.0 -- the perceived wait also includes capture buffering and
    STT/LLM/TTS latency, and 2.0 felt like ~4 s end to end."""
    from ascribe_link.agent_ws.stt import UtteranceBuffer
    import numpy as np
    from ascribe_link.agent_ws.audio import float32_to_pcm16

    assert UtteranceBuffer.END_SILENCE_S == 1.0

    buf = UtteranceBuffer()
    t = np.arange(16000, dtype=np.float32) / 16000.0
    tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    buf.add(float32_to_pcm16(tone), 16000)
    buf.add(float32_to_pcm16(np.zeros(int(16000 * 0.6), dtype=np.float32)), 16000)
    assert not buf.should_finalize()  # 0.6 s: a thinking pause, keep listening
    buf.add(float32_to_pcm16(np.zeros(int(16000 * 0.5), dtype=np.float32)), 16000)
    assert buf.should_finalize()  # 1.1 s: done
