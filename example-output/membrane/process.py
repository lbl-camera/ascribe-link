import os
import numpy as np
import tifffile
from skimage.filters import threshold_yen
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects

paths = [os.path.expanduser('~/Downloads/5dry.tif'),
         os.path.expanduser('~/Downloads/60dry.tif')]

# Common cube edge = smallest axis over all datasets (anisotropic stacks)
dims = []
for p in paths:
    with tifffile.TiffFile(p) as tif:
        dims.extend(tif.series[0].shape)
E = min(dims)
print("cube edge:", E)

def load_cube(p):
    with tifffile.TiffFile(p) as tif:
        nz, ny, nx = tif.series[0].shape
        z0 = (nz - E) // 2
        y0 = (ny - E) // 2
        x0 = (nx - E) // 2
        arr = tif.asarray(key=range(z0, z0 + E))[:, y0:y0 + E, x0:x0 + E]
    return np.ascontiguousarray(arr)

def process(cube):
    # Yen threshold, dark background -> foreground is bright side
    t = threshold_yen(cube)
    mask = cube > t
    # Filter objects smaller than 500 voxels (regionprops-equivalent)
    lbl = label(mask)
    keep = np.zeros(lbl.shape, dtype=bool)
    for r in regionprops(lbl):
        if r.num_pixels >= 500:
            keep[lbl == r.label] = True
    masked = np.where(keep, cube, 0).astype(cube.dtype)
    print("  yen t=", t, "kept voxels=", int(keep.sum()))
    return masked

rows = []
for p in paths:
    print("processing", os.path.basename(p))
    cube = load_cube(p)
    masked = process(cube)
    # each row: [raw | masked] side by side (columns)
    rows.append(np.concatenate([cube, masked], axis=2))

# 2x2: two dataset rows stacked vertically
out = np.concatenate(rows, axis=1)
out = np.ascontiguousarray(out)
print("output shape:", out.shape, out.dtype)
np.save(os.path.expanduser('~/ascribe_work/volume.npy'), out)
print("saved")
