"""Parametric specimen generation functions.

These functions generate 3D data from parameters, suitable for dynamic specimens.
"""

from __future__ import annotations

import pyvista as pv

from ascribe_link.models import MeshResult


def generate_sphere(radius: float = 1.0, resolution: int = 32, name: str | None = None, fortnite: int | None = None) -> MeshResult:
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
