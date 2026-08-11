import pyvista as pv


def volume_to_mesh(volume, decimation=0.9):
    # Wrap the NumPy array into a PyVista dataset.
    wrapped = pv.wrap(volume)

    # Threshold the volume to isolate the region of interest.
    thresholded = wrapped.threshold(0.5)

    # Extract the outer surface.
    mesh = thresholded.extract_surface()

    # Smooth the mesh.
    smoothed_mesh = mesh.smooth(n_iter=1000)

    # Triangulate the mesh so that all faces are triangles.
    tri_mesh = smoothed_mesh.triangulate()

    # Decimate the mesh to reduce the number of triangles.
    decimated_mesh = tri_mesh.decimate(decimation)

    # Compute normals so that both vertices and facet normals are available.
    final_mesh = decimated_mesh.compute_normals()

    # Flip normals
    final_mesh.flip_normals()

    # extract data
    vertices = final_mesh.points.tolist()  # shape: (n_points, 3)
    faces = final_mesh.faces.reshape((-1, 4))  # first number is always 3 for triangle
    indices = faces[:, 1:].flatten().tolist()

    return vertices, indices
