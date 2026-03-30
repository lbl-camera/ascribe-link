"""Mesh utilities for agent-generated data.

These utilities are designed to be used by the AI agent when generating
meshes with PyVista. They handle the common conversions needed to
submit mesh data via the submit_mesh tool.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pyvista as pv


def extract_mesh_data(
    mesh: "pv.PolyData", include_normals: bool = True
) -> tuple[list[list[float]], list[int], list[float] | None]:
    """Extract vertices, indices, and normals from a PyVista mesh.

    Automatically triangulates the mesh if needed (converts quads to triangles)
    and converts numpy arrays to plain Python lists.

    Parameters
    ----------
    mesh : pv.PolyData
        A PyVista mesh object (sphere, box, cylinder, or any PolyData).
    include_normals : bool
        If True, compute and return normals.

    Returns
    -------
    vertices : list[float]
        Flat list of vertex coordinates [x, y, z, x, y, z, ...].
    indices : list[int]
        Flat list of vertex indices (every 3 = one triangle).
    normals : list[float] | None
        Flat list of normal components [nx, ny, nz, nx, ny, nz, ...] or None.

    Example
    -------
    >>> import pyvista as pv
    >>> from ascribe_link.mesh_utils import extract_mesh_data
    >>>
    >>> sphere = pv.Sphere(radius=1.0)
    >>> vertices, indices, normals = extract_mesh_data(sphere)
    """
    # Triangulate to ensure all faces are triangles (not quads)
    triangulated = mesh.triangulate()

    # Compute normals if requested
    if include_normals:
        triangulated = triangulated.compute_normals(
            point_normals=True, cell_normals=False
        )

    # Extract vertices as flat list [x, y, z, x, y, z, ...]
    vertices = triangulated.points.flatten().tolist()

    # Extract face indices
    # PyVista faces format: [n, i0, i1, i2, n, i0, i1, i2, ...]
    # where n is the number of vertices per face (3 for triangles)
    faces = triangulated.faces
    if len(faces) > 0:
        # Reshape to [n_faces, 4] and take columns 1:4 (skip the count)
        indices = faces.reshape(-1, 4)[:, 1:].flatten().tolist()
    else:
        indices = []

    # Extract normals as flat list
    normals = None
    if include_normals and triangulated.point_normals is not None:
        normals = triangulated.point_normals.flatten().tolist()

    return vertices, indices, normals


def flatten_normals(normals_nested: list[list[float]]) -> list[float]:
    """Flatten nested normals [[nx, ny, nz], ...] to [nx, ny, nz, nx, ny, nz, ...].

    Use this when you have normals from marching_cubes or other sources
    that return per-vertex normal vectors as nested lists.
    """
    result = []
    for n in normals_nested:
        result.extend(n)
    return result
