import sys
sys.path.insert(0, r"C:\Users\rp\PycharmProjects\ascribe-link")
import numpy as np, json, time
from scipy import ndimage as ndi
from skimage.measure import marching_cubes
import pymeshlab as ml
from ascribe_link.mesh_utils import extract_mesh_data

t0=time.time()
masks=np.load("masks_sampled.npy")          # (S,H,W) bool
zs=np.load("masks_zs.npy")
S,H,W=masks.shape
mz=251
print("masks",masks.shape,"zs",zs.min(),zs.max(),flush=True)

# ---- per-slice cleanup: keep largest CC + fill holes ----
clean=np.zeros_like(masks)
for i in range(S):
    m=masks[i]
    if m.sum()==0: continue
    lab,n=ndi.label(m)
    if n>1:
        sizes=ndi.sum(np.ones_like(lab),lab,index=range(1,n+1))
        m=lab==(int(np.argmax(sizes))+1)
    m=ndi.binary_fill_holes(m)
    clean[i]=m

# ---- interpolate along z to full resolution ----
zoom_z=mz/float(S)
occ=ndi.zoom(clean.astype(np.float32),(zoom_z,1,1),order=1)
occ=occ[:mz] if occ.shape[0]>=mz else np.pad(occ,((0,mz-occ.shape[0]),(0,0),(0,0)))
vol=occ>0.5
print("occupancy voxels",int(vol.sum()),flush=True)

# ---- 3D cleanup: largest component, close, fill ----
vol=ndi.binary_closing(vol,iterations=2)
lab,n=ndi.label(vol)
if n>1:
    sizes=ndi.sum(np.ones_like(lab),lab,index=range(1,n+1))
    vol=lab==(int(np.argmax(sizes))+1)
vol=ndi.binary_fill_holes(vol)
print("final voxels",int(vol.sum()),"components",n,flush=True)

# ---- smooth + marching cubes ----
volf=ndi.gaussian_filter(vol.astype(np.float32),sigma=1.0)
verts,faces,norms,_=marching_cubes(volf,level=0.5)
print("raw mesh: V",len(verts),"F",len(faces),flush=True)

# ---- adaptive remeshing (pymeshlab) ----
ms=ml.MeshSet()
ms.add_mesh(ml.Mesh(vertex_matrix=np.ascontiguousarray(verts,dtype=np.float64),
                    face_matrix=np.ascontiguousarray(faces,dtype=np.int32)))
ms.meshing_remove_duplicate_vertices()
ms.meshing_remove_unreferenced_vertices()
try:
    tl=ml.PercentageValue(1.5)
except Exception:
    tl=ml.Percentage(1.5)
ms.meshing_isotropic_explicit_remeshing(iterations=5, adaptive=True, targetlen=tl)
print("after adaptive remesh: V",ms.current_mesh().vertex_number(),
      "F",ms.current_mesh().face_number(),flush=True)

# recompute normals and export
ms.compute_normal_per_vertex()
ms.save_current_mesh("final_mesh.ply")

import pyvista as pv
mesh=pv.read("final_mesh.ply")
vertices,indices,normals=extract_mesh_data(mesh)
with open("mesh.json","w") as f:
    json.dump({"vertices":vertices,"indices":indices,"normals":normals},f)
print("SUBMIT_READY verts",len(vertices)//3,"tris",len(indices)//3,
      "time",round(time.time()-t0,1),"s",flush=True)
