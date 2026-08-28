"""Tests for audio utilities: PCM conversion, resampling, silence detection."""
import numpy as np
import pytest

from ascribe_link.agent_ws import audio


class TestPCMConversion:
    """Test PCM16 <-> float32 conversion."""

    def test_round_trip_pcm16_to_float_to_pcm16(self):
        """Round-trip conversion stays within ±1 LSB."""
        # Create test data with various PCM16 values
        original_pcm16 = np.array(
            [0, 1, 100, 1000, 16384, 32767, -1, -100, -1000, -16384, -32768],
            dtype=np.int16,
        )
        original_bytes = original_pcm16.tobytes()

        # Convert to float and back
        float_arr = audio.pcm16_to_float32(original_bytes)
        recovered_bytes = audio.float32_to_pcm16(float_arr)
        recovered_pcm16 = np.frombuffer(recovered_bytes, dtype=np.int16)

        # Check difference is within ±1 LSB
        diff = np.abs(original_pcm16.astype(np.int32) - recovered_pcm16.astype(np.int32))
        assert np.all(diff <= 1), f"LSB error too large: max diff = {np.max(diff)}"

    def test_float_one_clamps_to_32767(self):
        """Float 1.0 should map to PCM16 32767."""
        float_arr = np.array([1.0], dtype=np.float32)
        pcm_bytes = audio.float32_to_pcm16(float_arr)
        pcm_val = np.frombuffer(pcm_bytes, dtype=np.int16)[0]
        assert pcm_val == 32767

    def test_float_minus_one_clamps_to_minus_32768(self):
        """Float -1.0 should map to PCM16 -32768 (or -32767; test accepts either)."""
        float_arr = np.array([-1.0], dtype=np.float32)
        pcm_bytes = audio.float32_to_pcm16(float_arr)
        pcm_val = np.frombuffer(pcm_bytes, dtype=np.int16)[0]
        # Accept both -32768 and -32767
        assert pcm_val in (-32768, -32767)

    def test_pcm16_to_float32_output_range(self):
        """Converted float32 should be in [-1, 1] range."""
        # Test with min and max PCM16 values
        pcm16_vals = np.array([32767, -32768, 16384, -16384, 0], dtype=np.int16)
        pcm_bytes = pcm16_vals.tobytes()
        float_arr = audio.pcm16_to_float32(pcm_bytes)

        assert np.all(float_arr >= -1.0)
        assert np.all(float_arr <= 1.0)
        assert float_arr.dtype == np.float32

    def test_float32_to_pcm16_clips_out_of_range(self):
        """Values outside [-1, 1] should be clipped."""
        float_arr = np.array([1.5, -1.5, 2.0, -2.0], dtype=np.float32)
        pcm_bytes = audio.float32_to_pcm16(float_arr)
        pcm_vals = np.frombuffer(pcm_bytes, dtype=np.int16)

        assert np.all(pcm_vals <= 32767)
        assert np.all(pcm_vals >= -32768)

    def test_pcm16_to_float32_odd_length_bytes_truncates(self):
        """Odd-length bytes should truncate to largest even length (drop trailing byte)."""
        # 3 bytes = 1 sample + 1 dropped byte
        odd_bytes = b"\x01\x02\x03"
        float_arr = audio.pcm16_to_float32(odd_bytes)

        # Should get 1 sample (3 bytes → 2 bytes → 1 int16)
        assert len(float_arr) == 1
        assert float_arr.dtype == np.float32

    def test_pcm16_to_float32_empty_bytes_returns_empty(self):
        """Empty bytes should return empty float32 array."""
        empty_bytes = b""
        float_arr = audio.pcm16_to_float32(empty_bytes)

        assert len(float_arr) == 0
        assert float_arr.dtype == np.float32


class TestResample:
    """Test audio resampling."""

    def test_resample_48000_to_16000_sine_wave(self):
        """Resample 1 kHz sine from 48 kHz to 16 kHz.

        Expected: length ratio 1/3 (±1 sample) and stays a sine wave
        (zero crossings within ±2 of expected).
        """
        src_rate = 48000
        dst_rate = 16000
        duration_s = 0.1  # 100 ms for stable sine
        freq = 1000  # 1 kHz

        # Generate sine wave
        t = np.arange(src_rate * duration_s) / src_rate
        sine = np.sin(2 * np.pi * freq * t).astype(np.float32)

        resampled = audio.resample(sine, src_rate, dst_rate)

        # Check length ratio (expect src_rate / dst_rate = 3)
        expected_len = len(sine) * dst_rate // src_rate
        assert abs(len(resampled) - expected_len) <= 1, (
            f"Length ratio wrong: got {len(resampled)}, expected ~{expected_len}"
        )

        # Check zero crossings to verify it's still a sine
        # In 100 ms at 1 kHz, expect ~200 zero crossings
        zero_crossings = np.sum(np.diff(np.sign(resampled)) != 0)
        expected_crossings = int(2 * freq * duration_s)  # 200 for 100 ms
        assert abs(zero_crossings - expected_crossings) <= 2, (
            f"Zero crossings suggest not a sine: got {zero_crossings}, "
            f"expected ~{expected_crossings}"
        )

    def test_identity_resample_returns_same_array(self):
        """Resampling at same rate should return identical array."""
        arr = np.sin(np.linspace(0, 2 * np.pi, 1000)).astype(np.float32)
        resampled = audio.resample(arr, 16000, 16000)

        # Check it's the same
        np.testing.assert_array_equal(resampled, arr)

    def test_resample_output_is_float32(self):
        """Resampled output should be float32."""
        arr = np.array([0.5, -0.3, 0.8, -0.1, 0.2], dtype=np.float32)
        resampled = audio.resample(arr, 16000, 8000)
        assert resampled.dtype == np.float32

    def test_resample_empty_array_different_rates(self):
        """Empty array with different rates should return empty float32 array."""
        empty_arr = np.array([], dtype=np.float32)
        resampled = audio.resample(empty_arr, 48000, 16000)

        assert len(resampled) == 0
        assert resampled.dtype == np.float32

    def test_resample_empty_array_same_rate(self):
        """Empty array with same rate should return empty float32 array."""
        empty_arr = np.array([], dtype=np.float32)
        resampled = audio.resample(empty_arr, 16000, 16000)

        assert len(resampled) == 0
        assert resampled.dtype == np.float32


class TestTrailingSilence:
    """Test trailing silence detection."""

    def test_trailing_silence_tone_then_zeros(self):
        """Trailing silence after 0.5s tone + 1.0s zeros should return ~1.0s (±0.06)."""
        rate = 16000
        tone_duration = 0.5
        silence_duration = 1.0
        threshold = 0.01

        # Generate tone (loud enough to be clearly above threshold)
        tone_samples = int(rate * tone_duration)
        tone = np.sin(np.linspace(0, 2 * np.pi * 10, tone_samples)).astype(
            np.float32
        )  # 10 Hz tone
        # Generate silence (zeros)
        silence_samples = int(rate * silence_duration)
        silence = np.zeros(silence_samples, dtype=np.float32)

        combined = np.concatenate([tone, silence]).astype(np.float32)

        trailing_silence = audio.trailing_silence_s(combined, rate, threshold)

        # Should be approximately 1.0 second, within ±0.06s
        assert abs(trailing_silence - silence_duration) <= 0.06, (
            f"Expected ~{silence_duration}s silence, got {trailing_silence}s"
        )

    def test_all_silence_returns_full_duration(self):
        """All-silence array should return full duration."""
        rate = 16000
        duration = 2.0
        samples = int(rate * duration)
        silence = np.zeros(samples, dtype=np.float32)

        trailing_silence = audio.trailing_silence_s(silence, rate, threshold=0.01)

        assert abs(trailing_silence - duration) <= 0.001, (
            f"Expected ~{duration}s silence, got {trailing_silence}s"
        )

    def test_trailing_silence_no_silence_at_end(self):
        """Array ending with loud signal should return ~0s silence."""
        rate = 16000
        duration = 1.0
        samples = int(rate * duration)
        loud_signal = np.ones(samples, dtype=np.float32) * 0.5  # Loud throughout

        trailing_silence = audio.trailing_silence_s(loud_signal, rate, threshold=0.01)

        # Should be very small, essentially 0
        assert trailing_silence < 0.05, f"Expected ~0s silence, got {trailing_silence}s"

    def test_trailing_silence_window_is_50ms(self):
        """Trailing silence uses 50ms windows for RMS calculation."""
        rate = 16000
        # 1 window = 50ms = 0.05s at 16kHz = 800 samples
        window_samples = rate // 20  # 50ms window = 800 samples at 16kHz

        # Create signal that's loud for 2 windows, then quiet for exactly 1 window
        loud = np.ones(2 * window_samples, dtype=np.float32) * 0.5
        quiet = np.zeros(window_samples, dtype=np.float32)
        signal = np.concatenate([loud, quiet]).astype(np.float32)

        trailing_silence = audio.trailing_silence_s(signal, rate, threshold=0.01)

        # Should be approximately 1 window = 50ms = 0.05s
        expected = 0.05
        assert abs(trailing_silence - expected) <= 0.01, (
            f"Expected ~{expected}s trailing silence (1 window), got {trailing_silence}s"
        )

    def test_trailing_silence_empty_array_returns_zero(self):
        """Empty array should return 0.0 seconds of trailing silence."""
        empty_arr = np.array([], dtype=np.float32)
        trailing_silence = audio.trailing_silence_s(empty_arr, 16000, threshold=0.01)

        assert trailing_silence == 0.0
