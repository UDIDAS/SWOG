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

# canonical label -> (organ name, RGBA color). Left+right kidney merge into one "kidney".
CANON = {
    1:  ("liver",    (170,  95,  70, 255)),
    2:  ("kidney",   (125,  95, 180, 255)),
    13: ("kidney",   (125,  95, 180, 255)),
    4:  ("pancreas", ( 90, 180, 115, 255)),
    14: ("tumor",    (235,  55,  55, 255)),
    3:  ("spleen",   (180, 125,  95, 255)),
}
DEFAULT_ORGANS = ["liver", "kidney", "pancreas", "tumor"]
MIN_VOXELS = 60          # skip specks
SMOOTH_SIGMA = 0.7       # binary pre-smoothing -> nicer surface
LAPLACIAN_ITERS = 8      # surface smoothing after marching cubes


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
    trimesh.smoothing.filter_laplacian(m, iterations=LAPLACIAN_ITERS)
    m.visual.face_colors = np.array(color, dtype=np.uint8)
    return m


def meshify(label_path, out_dir, tag="seg", organs=None):
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
        made[organ] = {"voxels": int(binary.sum()),
                       "volume_cm3": round(int(binary.sum()) * float(np.prod(spacing)) / 1000.0, 1),
                       "faces": int(len(m.faces))}
    glb = None
    if made:
        glb = os.path.join(out_dir, f"{tag}_organs.glb")
        scene.export(glb)
    return {"glb": glb, "spacing_mm": spacing, "organs": made}


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
