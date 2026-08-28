"""Audio utilities: PCM16/float32 conversion, resampling, silence detection.

Pure numpy implementation for STT/TTS pipeline.
"""
import numpy as np


def pcm16_to_float32(data: bytes) -> np.ndarray:
    """Convert PCM16 (little-endian signed 16-bit) bytes to float32 in [-1, 1].

    Args:
        data: Raw PCM16 bytes (little-endian, signed).

    Returns:
        Float32 array in [-1, 1] range.
    """
    # Interpret bytes as int16 (little-endian is numpy's default)
    pcm16 = np.frombuffer(data, dtype=np.int16)
    # Convert to float32: divide by 32768.0 to get [-1, 1]
    float32 = pcm16.astype(np.float32) / 32768.0
    return float32


def float32_to_pcm16(arr: np.ndarray) -> bytes:
    """Convert float32 in [-1, 1] to PCM16 (little-endian signed 16-bit) bytes.

    Args:
        arr: Float32 array (should be in [-1, 1] range; values outside are clipped).

    Returns:
        Raw PCM16 bytes (little-endian, signed).
    """
    # Clip to [-1, 1]
    clipped = np.clip(arr, -1.0, 1.0)
    # Scale: 1.0 -> 32767, -1.0 -> -32768
    # We scale by 32767 then use int16 which wraps -32768 correctly
    scaled = (clipped * 32767).astype(np.int16)
    return scaled.tobytes()


def resample(arr: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample audio using linear interpolation (np.interp).

    Args:
        arr: Input float32 audio array.
        src_rate: Source sample rate (Hz).
        dst_rate: Destination sample rate (Hz).

    Returns:
        Resampled float32 array. If src_rate == dst_rate, returns the same array.
    """
    # Identity case: same rate
    if src_rate == dst_rate:
        return arr

    # Create time grid for source and destination
    src_time = np.arange(len(arr))
    # Map to seconds-equivalent scale
    dst_time = np.arange(len(arr) * dst_rate / src_rate) * (src_rate / dst_rate)

    # Linear interpolation
    resampled = np.interp(dst_time, src_time, arr)
    return resampled.astype(np.float32)


def trailing_silence_s(
    arr: np.ndarray, rate: int, threshold: float = 0.01
) -> float:
    """Measure trailing silence duration in seconds.

    Computes RMS on 50 ms windows and counts trailing windows below threshold.

    Args:
        arr: Input float32 audio array.
        rate: Sample rate (Hz).
        threshold: RMS threshold; windows with RMS < threshold count as silent.

    Returns:
        Duration of trailing silence in seconds.
    """
    # 50 ms window
    window_samples = int(rate * 0.05)

    if window_samples <= 0:
        return 0.0

    # Pad array to make it divisible by window_samples
    total_samples = len(arr)
    padded_len = ((total_samples + window_samples - 1) // window_samples) * window_samples
    padded = np.pad(arr, (0, padded_len - total_samples), mode="constant")

    # Compute RMS for each window
    windows = padded.reshape(-1, window_samples)
    rms = np.sqrt(np.mean(windows**2, axis=1))

    # Count trailing silent windows (RMS < threshold)
    # Start from the end and count backwards
    silent_count = 0
    for i in range(len(rms) - 1, -1, -1):
        if rms[i] < threshold:
            silent_count += 1
        else:
            break

    # Convert to seconds
    trailing_silence_duration = silent_count * 0.05
    return trailing_silence_duration
