"""Mesh utilities for agent-generated data.

These utilities are designed to be used by the AI agent when generating
meshes with PyVista. They handle the common conversions needed to
submit mesh data via the submit_mesh tool.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pyvista as pv


def extract_mesh_data(mesh: "pv.PolyData") -> tuple[list[list[float]], list[int]]:
    """Extract vertices and indices from a PyVista mesh for submit_mesh.

    Automatically triangulates the mesh if needed (converts quads to triangles)
    and converts numpy arrays to plain Python lists.

    Parameters
    ----------
    mesh : pv.PolyData
        A PyVista mesh object (sphere, box, cylinder, or any PolyData).

    Returns
    -------
    vertices : list[list[float]]
        List of [x, y, z] coordinates for each vertex.
    indices : list[int]
        Flat list of vertex indices (every 3 = one triangle).

    Example
    -------
    >>> import pyvista as pv
    >>> from ascribe_link.mesh_utils import extract_mesh_data
    >>>
    >>> sphere = pv.Sphere(radius=1.0)
    >>> vertices, indices = extract_mesh_data(sphere)
    >>> # Now call submit_mesh(vertices=vertices, indices=indices)
    """
    # Triangulate to ensure all faces are triangles (not quads)
    triangulated = mesh.triangulate()

    # Extract vertices as list of [x, y, z]
    vertices = triangulated.points.tolist()

    # Extract face indices
    # PyVista faces format: [n, i0, i1, i2, n, i0, i1, i2, ...]
    # where n is the number of vertices per face (3 for triangles)
    faces = triangulated.faces
    if len(faces) > 0:
        # Reshape to [n_faces, 4] and take columns 1:4 (skip the count)
        indices = faces.reshape(-1, 4)[:, 1:].flatten().tolist()
    else:
        indices = []

    return vertices, indices


def create_cube(
    size: float = 1.0, center: tuple[float, float, float] = (0, 0, 0)
) -> tuple[list[list[float]], list[int]]:
    """Create a cube mesh and return vertices/indices ready for submit_mesh.

    Parameters
    ----------
    size : float
        Side length of the cube. Default 1.0.
    center : tuple
        Center point (x, y, z). Default (0, 0, 0).

    Returns
    -------
    vertices, indices : tuple
        Ready to pass to submit_mesh.
    """
    import pyvista as pv

    half = size / 2
    cx, cy, cz = center
    box = pv.Box(
        bounds=(cx - half, cx + half, cy - half, cy + half, cz - half, cz + half)
    )
    return extract_mesh_data(box)


def create_sphere(
    radius: float = 1.0,
    center: tuple[float, float, float] = (0, 0, 0),
    resolution: int = 30,
) -> tuple[list[list[float]], list[int]]:
    """Create a sphere mesh and return vertices/indices ready for submit_mesh.

    Parameters
    ----------
    radius : float
        Sphere radius. Default 1.0.
    center : tuple
        Center point (x, y, z). Default (0, 0, 0).
    resolution : int
        Angular resolution. Default 30.

    Returns
    -------
    vertices, indices : tuple
        Ready to pass to submit_mesh.
    """
    import pyvista as pv

    sphere = pv.Sphere(
        radius=radius,
        center=center,
        theta_resolution=resolution,
        phi_resolution=resolution,
    )
    return extract_mesh_data(sphere)


def create_cylinder(
    radius: float = 0.5,
    height: float = 1.0,
    center: tuple[float, float, float] = (0, 0, 0),
    resolution: int = 30,
) -> tuple[list[list[float]], list[int]]:
    """Create a cylinder mesh and return vertices/indices ready for submit_mesh.

    Parameters
    ----------
    radius : float
        Cylinder radius. Default 0.5.
    height : float
        Cylinder height. Default 1.0.
    center : tuple
        Center point (x, y, z). Default (0, 0, 0).
    resolution : int
        Angular resolution. Default 30.

    Returns
    -------
    vertices, indices : tuple
        Ready to pass to submit_mesh.
    """
    import pyvista as pv

    cylinder = pv.Cylinder(
        radius=radius, height=height, center=center, resolution=resolution
    )
    return extract_mesh_data(cylinder)


def create_torus(
    ring_radius: float = 1.0,
    cross_section_radius: float = 0.3,
    center: tuple[float, float, float] = (0, 0, 0),
) -> tuple[list[list[float]], list[int]]:
    """Create a torus mesh and return vertices/indices ready for submit_mesh.

    Parameters
    ----------
    ring_radius : float
        Distance from center to tube center. Default 1.0.
    cross_section_radius : float
        Radius of the tube. Default 0.3.
    center : tuple
        Center point (x, y, z). Default (0, 0, 0).

    Returns
    -------
    vertices, indices : tuple
        Ready to pass to submit_mesh.
    """
    import pyvista as pv

    torus = pv.ParametricTorus(
        ringradius=ring_radius, crosssectionradius=cross_section_radius
    )
    if center != (0, 0, 0):
        torus = torus.translate(center, inplace=False)
    return extract_mesh_data(torus)
