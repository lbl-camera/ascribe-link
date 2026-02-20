import numpy as np

from ascribe_link.utils import volume_to_mesh


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


def sphere_example():
    # create a volumetric sphere
    volume = create_sphere_array()

    # convert to mesh
    mesh = volume_to_mesh(volume)

    return mesh
