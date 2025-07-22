import numpy as np
import pyvista as pv

def create_sphere_array(size=50):
    """
    Create a binary 3D NumPy array containing a sphere.
    Voxels inside the sphere are 1; outside are 0.
    """
    binary_stack = np.zeros((size, size, size), dtype=np.uint8)
    center = size // 2
    radius = size // 4
    for x in range(size):
        for y in range(size):
            for z in range(size):
                if (x - center)**2 + (y - center)**2 + (z - center)**2 <= radius**2:
                    binary_stack[x, y, z] = 1
    return binary_stack


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

    # extract data
    vertices = final_mesh.points.tolist()  # shape: (n_points, 3)
    faces = final_mesh.faces.reshape((-1, 4))  # first number is always 3 for triangle
    indices = faces[:, 1:].flatten().tolist()

    return vertices, indices

def sphere_example():
    # create a volumetric sphere
    volume = create_sphere_array()

    # convert to mesh
    mesh = volume_to_mesh(volume)

    return mesh
