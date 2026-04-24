"""Tests for AI-agent output dispatch (mesh vs. volume)."""
from __future__ import annotations

import numpy as np
import pytest

import ascribe_link.agent_generator as agent_gen
from ascribe_link.models import MeshResult, VolumeResult


def test_dispatch_3d_ndarray_to_volume():
    arr = np.ones((8, 8, 8), dtype=np.float32)
    result = agent_gen.wrap_agent_output(arr)
    assert isinstance(result, VolumeResult)
    assert result.shape == [8, 8, 8]


def test_dispatch_volume_result_passthrough():
    original = VolumeResult.from_numpy(np.zeros((2, 3, 4), dtype=np.float32))
    result = agent_gen.wrap_agent_output(original)
    assert result is original


def test_dispatch_mesh_result_passthrough():
    original = MeshResult(vertices=[0.0, 0.0, 0.0], indices=[0], normals=None)
    result = agent_gen.wrap_agent_output(original)
    assert result is original


def test_dispatch_pyvista_mesh_to_mesh_result():
    import pyvista as pv
    mesh = pv.Sphere(radius=1.0, theta_resolution=8, phi_resolution=8)
    result = agent_gen.wrap_agent_output(mesh)
    assert isinstance(result, MeshResult)
    assert len(result.vertices) > 0
    assert len(result.indices) > 0


def test_dispatch_unknown_raises():
    with pytest.raises(TypeError, match="cannot wrap agent output"):
        agent_gen.wrap_agent_output("not a mesh or volume")
