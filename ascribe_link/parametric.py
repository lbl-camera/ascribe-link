"""Parametric specimen generation functions.

These functions generate 3D data from parameters, suitable for dynamic specimens.
"""

from __future__ import annotations

import numpy as np
import pyvista as pv

from ascribe_link.models import MeshResult, VolumeResult


def generate_sphere(radius: float = 1.0, resolution: int = 32) -> MeshResult:
    """Generate a parametric sphere mesh.

    Parameters
    ----------
    radius : float
        Sphere radius (0.5 to 5.0)
    resolution : int
        Number of divisions (8 to 128)

    Returns
    -------
    MeshResult
        Sphere mesh with vertices, indices, and normals
    """
    sphere = pv.Sphere(
        radius=radius,
        theta_resolution=resolution,
        phi_resolution=resolution,
    )
    return MeshResult.from_pyvista(sphere)


def generate_torus(
    major_radius: float = 1.0,
    minor_radius: float = 0.3,
    segments: int = 32,
) -> MeshResult:
    """Generate a parametric torus mesh.

    Parameters
    ----------
    major_radius : float
        Distance from center to tube center (0.5 to 3.0)
    minor_radius : float
        Tube radius (0.1 to 1.0)
    segments : int
        Number of divisions (8 to 128)

    Returns
    -------
    MeshResult
        Torus mesh with vertices, indices, and normals
    """
    torus = pv.ParametricTorus(
        ringradius=major_radius,
        crosssectionradius=minor_radius,
    )
    # Resample to target resolution
    torus = torus.subdivide(segments // 16, subfilter="butterfly")
    return MeshResult.from_pyvista(torus)


def generate_gaussian_volume(resolution: int = 64, sigma: float = 0.3) -> VolumeResult:
    """Generate a 3D Gaussian blob centered in a unit cube.

    Parameters
    ----------
    resolution : int
        Number of voxels per axis (clamped to [32, 256]).
    sigma : float
        Standard deviation of the Gaussian relative to the cube edge
        (clamped to [0.05, 1.0]).

    Returns
    -------
    VolumeResult
        float32 volume, shape [resolution]*3, normalized so peak == 1.0.
    """
    resolution = max(32, min(256, int(resolution)))
    sigma = max(0.05, min(1.0, float(sigma)))

    axis = np.linspace(-0.5, 0.5, resolution, dtype=np.float32)
    z, y, x = np.meshgrid(axis, axis, axis, indexing="ij")
    r2 = x * x + y * y + z * z
    volume = np.exp(-r2 / (2.0 * sigma * sigma)).astype(np.float32)
    # Normalize so peak == 1.0 exactly (the discrete grid may not sample
    # the true continuous peak at r=0, especially for even resolutions).
    volume /= volume.max()
    return VolumeResult.from_numpy(
        volume,
        spacing=[1.0 / resolution, 1.0 / resolution, 1.0 / resolution],
        origin=[0.0, 0.0, 0.0],
    )
