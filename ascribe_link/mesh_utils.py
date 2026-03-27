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
