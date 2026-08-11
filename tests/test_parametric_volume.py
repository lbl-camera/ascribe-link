"""Tests for parametric volume specimens."""
from __future__ import annotations

import pytest

from ascribe_link.models import VolumeResult
from ascribe_link.parametric import generate_gaussian_volume


def test_default_gaussian_volume():
    result = generate_gaussian_volume()
    assert isinstance(result, VolumeResult)
    assert result.shape == [64, 64, 64]
    assert result.dtype == "float32"
    arr = result.to_numpy()
    # Peak at the center, decays outward.
    center = (32, 32, 32)
    assert arr[center] == pytest.approx(1.0, abs=1e-3)
    # A corner should be much smaller than the center.
    assert arr[0, 0, 0] < arr[center] * 0.5


def test_gaussian_volume_resolution():
    result = generate_gaussian_volume(resolution=32)
    assert result.shape == [32, 32, 32]


def test_gaussian_volume_sigma_affects_spread():
    narrow = generate_gaussian_volume(resolution=32, sigma=0.1).to_numpy()
    wide = generate_gaussian_volume(resolution=32, sigma=0.5).to_numpy()
    # Wider sigma -> more mass away from center.
    off_center_narrow = narrow[0, 0, 0]
    off_center_wide = wide[0, 0, 0]
    assert off_center_wide > off_center_narrow


def test_gaussian_volume_clamps_resolution():
    # Below minimum
    r = generate_gaussian_volume(resolution=8)
    assert r.shape == [32, 32, 32]
    # Above maximum
    r = generate_gaussian_volume(resolution=999)
    assert r.shape == [256, 256, 256]
