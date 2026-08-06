import time, numpy as np, torch
from scipy import ndimage as ndi
from segment_anything import sam_model_registry, SamPredictor

t0 = time.time()
vol = np.load("vol_sub.npy")
mz, my, mx = vol.shape
H, W = my, mx
sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b.pth"); sam.to("cpu")
pred = SamPredictor(sam)
print("model loaded", round(time.time()-t0, 1), "s", flush=True)

def texture_seed(f):
    # local std as texture; plant interior is textured, medium is smooth
    m = ndi.uniform_filter(f, size=5)
    sq = ndi.uniform_filter(f*f, size=5)
    std = np.sqrt(np.clip(sq - m*m, 0, None))
    thr = np.percentile(std, 92)
    hi = std > thr
    hi = ndi.binary_opening(hi, iterations=1)
    lab, n = ndi.label(hi)
    if n == 0:
        return None, None
    sizes = ndi.sum(np.ones_like(lab), lab, index=range(1, n+1))
    k = int(np.argmax(sizes)) + 1
    cc = lab == k
    ys, xs = np.where(cc)
    cy, cx = ys.mean(), xs.mean()
    # pick a few interior points near centroid
    order = np.argsort((ys-cy)**2 + (xs-cx)**2)
    pts = [[xs[i], ys[i]] for i in order[:5]]
    return np.array(pts, dtype=np.float32), cc.sum()

zs = list(range(0, mz, 2))  # every 2nd slice
masks = np.zeros((len(zs), H, W), dtype=bool)
neg = np.array([[2, 2], [W-3, 2], [2, H-3], [W-3, H-3]], dtype=np.float32)  # corners = medium
neg_lab = np.zeros(len(neg))

for i, z in enumerate(zs):
    s = vol[z].astype(np.float32)
    sn = (s - s.min()) / (s.ptp() + 1e-6) * 255.0
    rgb = np.repeat(sn.astype(np.uint8)[:, :, None], 3, axis=2)
    seed, seed_area = texture_seed(sn)
    if seed is None:
        continue
    pred.set_image(rgb)
    pos_lab = np.ones(len(seed))
    pc = np.concatenate([seed, neg], axis=0)
    pl = np.concatenate([pos_lab, neg_lab], axis=0)
    m, sc, _ = pred.predict(point_coords=pc, point_labels=pl, multimask_output=True)
    areas = np.array([x.sum() for x in m], dtype=float)
    frac = areas / (H*W)
    ok = (frac > 0.02) & (frac < 0.5)
    cy, cx = int(seed[0][1]), int(seed[0][0])
    cand = []
    for j in range(len(m)):
        contains = m[j][cy, cx]
        cand.append((ok[j] and contains, ok[j], sc[j], j))
    cand.sort(reverse=True)
    best = cand[0][3]
    masks[i] = m[best]
    if i % 10 == 0:
        print(f"slice {i}/{len(zs)} z={z} area={int(areas[best])} score={sc[best]:.2f} "
              f"elapsed={round(time.time()-t0,1)}s", flush=True)
        np.save("masks_partial.npy", masks)

np.save("masks_sampled.npy", masks)
np.save("masks_zs.npy", np.array(zs))
print("DONE sam", round(time.time()-t0, 1), "s, sampled", len(zs), "slices", flush=True)
