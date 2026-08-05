#!/usr/bin/env python3
"""3D surface reconstruction from a label volume — turns a segmentation NIfTI into app-ready 3D meshes.
For each organ/tumor it extracts the binary mask, smooths it, runs marching cubes at the true voxel
spacing (so the mesh is in millimetres), lightly Laplacian-smooths the surface, colors it, and exports:
  - one combined, colored  <tag>_organs.glb   (drop straight into a web/desktop 3D viewer: <model-viewer>,
    three.js, Babylon, Unity, Blender ...)
  - per-organ  <organ>.stl                     (universal: 3D print, Slicer, MeshLab)

Usage:
  python meshify.py LABEL.nii.gz OUT_DIR [--tag gt] [--organs liver,kidney,pancreas,tumor]
"""
import argparse
import os

import numpy as np
import nibabel as nib
import trimesh
from scipy import ndimage
from skimage.measure import marching_cubes

# canonical label -> (organ name, RGBA color). Kidneys kept SEPARATE (KG + metrics.json score them apart).
CANON = {
    1:  ("liver",        (170,  95,  70, 255)),
    2:  ("right_kidney", (125,  95, 180, 255)),
    13: ("left_kidney",  (110,  80, 165, 255)),
    3:  ("spleen",       (180, 125,  95, 255)),
    4:  ("pancreas",     ( 90, 180, 115, 255)),
    14: ("tumor",        (235,  55,  55, 255)),
}
DEFAULT_ORGANS = ["liver", "right_kidney", "left_kidney", "spleen", "pancreas", "tumor"]
MIN_VOXELS = 60          # skip specks
SMOOTH_SIGMA = 0.7       # binary pre-smoothing -> nicer surface
TAUBIN_ITERS = 10        # volume-PRESERVING smoothing (Laplacian shrank small structures up to ~48%)


def _organ_labels(organ):
    return [lab for lab, (name, _) in CANON.items() if name == organ]


def mesh_one(binary, spacing, color, step=2):
    """binary volume -> a colored trimesh (or None if too small)."""
    if int(binary.sum()) < MIN_VOXELS:
        return None
    vol = ndimage.gaussian_filter(binary.astype(np.float32), SMOOTH_SIGMA)
    if vol.max() < 0.5:
        return None
    try:
        verts, faces, _, _ = marching_cubes(vol, level=0.5, spacing=spacing, step_size=step)
    except (ValueError, RuntimeError):
        return None
    m = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    if len(m.faces) == 0:
        return None
    pre = m.vertices.copy()
    try:
        trimesh.smoothing.filter_taubin(m, iterations=TAUBIN_ITERS)   # volume-preserving (unlike Laplacian)
    except Exception:
        m.vertices = pre
    if not np.isfinite(m.vertices).all():        # guard: smoothing can diverge to NaN on odd components
        m.vertices = pre
    m.visual.face_colors = np.array(color, dtype=np.uint8)
    return m


def meshify(label_path, out_dir, tag="seg", organs=None, glb=True):
    organs = organs or DEFAULT_ORGANS
    os.makedirs(out_dir, exist_ok=True)
    nii = nib.load(label_path)
    lab = np.asarray(nii.dataobj)
    spacing = tuple(float(z) for z in nii.header.get_zooms()[:3])
    scene = trimesh.Scene()
    made = {}
    for organ in organs:
        labs = _organ_labels(organ)
        if not labs:
            continue
        binary = np.isin(lab, labs)
        color = CANON[labs[0]][1]
        m = mesh_one(binary, spacing, color)
        if m is None:
            continue
        m.export(os.path.join(out_dir, f"{organ}.stl"))
        scene.add_geometry(m, geom_name=organ)
        try:
            mvol = round(abs(float(m.volume)) / 1000.0, 2)      # mesh-enclosed volume (mm^3 -> cm^3)
        except Exception:
            mvol = None
        made[organ] = {"voxels": int(binary.sum()),
                       "volume_cm3": round(int(binary.sum()) * float(np.prod(spacing)) / 1000.0, 1),
                       "mesh_volume_cm3": mvol, "faces": int(len(m.faces))}
    glb_path = None
    if made and glb:
        glb_path = os.path.join(out_dir, f"{tag}_organs.glb")
        scene.export(glb_path)
    return {"glb": glb_path, "spacing_mm": spacing, "organs": made}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("label"); ap.add_argument("out_dir")
    ap.add_argument("--tag", default="seg")
    ap.add_argument("--organs", default=",".join(DEFAULT_ORGANS))
    a = ap.parse_args()
    r = meshify(a.label, a.out_dir, a.tag, a.organs.split(","))
    print(f"spacing {r['spacing_mm']} mm | organs:")
    for o, d in r["organs"].items():
        print(f"  {o:9s} {d['volume_cm3']:7.1f} cm3  {d['faces']:6d} faces")
    print(f"-> {r['glb']}")
