# Example Output — Archived Agent Sessions

Complete records of three prompt-driven analyses run through ASCRIBE-Link's agent layer.
Each directory is a self-contained audit trail: the prompt that was issued, the full
session transcript including every tool call and its result, the Python the agent wrote,
the console output of each stage, and the array or mesh that was submitted to the VR
client.

These accompany the paper *"Turning Immersive Viewers Into Analytical Workspaces:
ASCRIBE-XR and Agent-Driven Scientific Visualization"* (Journal of Imaging), where they
are described in the "Archived Agent Sessions" section and provided as Supplementary
S1–S3.

## Directories

| Directory | Paper | Use case | Input data |
|---|---|---|---|
| `membrane/` | S1 | Fuel-cell gas diffusion layers | `5dry.tif`, `60dry.tif` |
| `plant/` | S2 | *Panicum hallii* root, zero-shot SAM | `rec20201028_190153_esther-singer_wet2_pipette_z50_YESagar_x00y01_8bitcrop-roi.tif` |
| `concrete/` | S3 | High-resilience concrete | `LOAD5_rec20220824_c3_comp_05_y0002_verticalcrop` (200 slices) |

## Files

Every directory contains:

- `transcript.md` — the verbatim prompt and the complete session, tool call by tool call
- `run.log` (or per-stage logs) — console output of the generated code
- `volume.npy` / `mesh.json` — the result submitted to the client via the MCP tools

Generated scripts, as written by the agent:

- `membrane/process.py` — Yen threshold, `regionprops` size filter, 2×2 raw/masked stack
- `plant/run_sam.py` — SAM inference, slice by slice
- `plant/mesh_and_remesh.py` — mask cleanup, marching cubes, PyMeshLab adaptive remeshing

`plant/` additionally holds the segmentation intermediates (`masks_sampled.npy`,
`masks_zs.npy`, `vol_sub.npy`), diagnostic renderings (`seg_overlay.png`, `montage_z.png`,
`coronal.png`, `mesh_preview.png`), the final surface (`final_mesh.ply`), and the install
and download logs.

## Environment

| Component | Version |
|---|---|
| Model | `claude-opus-4-8` |
| `claude-agent-sdk` | 0.1.50 |
| Claude Code CLI | 2.1.81 |
| Node.js | 24.7.0 |
| Python | 3.12.0 |
| `ascribe_link` | 0.2.0 |
| NumPy / SciPy | 1.26.4 / 1.12.0 |
| scikit-image | 0.22.0 |
| PyVista / VTK | 0.45.3 / 9.4.2 |
| PyMeshLab / Trimesh | 2025.7.post1 / 4.7.1 |

The `plant/` session installed its own segmentation stack at run time, which is not part
of this package's dependencies: `segment-anything` 1.0 with `torch` 2.13.0+cpu and
`torchvision` 0.28.0+cpu.

## SAM checkpoint

The ViT-B checkpoint (375 MB) is **not** committed here. `plant/ckpt_dl.log` records the
download; retrieve it from the same URL the agent used:

```
https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

## A note on reproducibility

The prompts are deliberately under-specified: they name a goal and leave the method open,
which is the condition these examples exist to demonstrate. Re-running them does not
reproduce these sessions step for step. The approach and the character of the result stay
consistent, but choices the prompt leaves open — thresholds, intermediate parameters, the
structure of the generated code — vary between runs.

If you want determinism, lift the generated script out of a session directory and run it
directly; at that point it is ordinary Python. Prompt specificity is a control you hold,
trading adaptability for repeatability.
