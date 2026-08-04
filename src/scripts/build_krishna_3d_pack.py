#!/usr/bin/env python3
"""Build the 3D-reconstruction resource pack for the application handoff. CPU-only (safe alongside GPU
training). Produces app-ready 3D assets from data we already hold — no model inference needed:

  cases_mesh/<id>/   : 3D organ reconstructions meshed from the LOCAL flare_labels.bin (2,200 available)
                       -> seg.nii.gz + meshes/{liver,kidney,pancreas,tumor}.stl + gt_organs.glb
  cases_ct/<id>/     : the same but WITH the CT volume as an underlay (from flare_full_cases)
                       -> CT.nii.gz (CT-affine) + seg_gt.nii.gz + meshes/ + preview.png
  cases_pred/<id>/   : GT vs our model's 3D prediction, both meshed  -> meshes/{gt,pred}/ + preview.png
  MANIFEST.json      : every case, its organs, volumes (cm3), and files

out: /scratch/ud3d4/acm_data/krishna_3d_pack
"""
import json
import os
import struct
import subprocess
import sys
import zlib
from collections import Counter

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh

sys.path.insert(0, "/home/ud3d4/Desktop/SWOG/src/scripts")
from meshify import meshify, CANON

LAB = "/scratch/ud3d4/acm_data/flare23_labels"
FULL = "/scratch/ud3d4/acm_data/flare_full_cases"
PRED = "/scratch/ud3d4/acm_data/flare23_pred_masks"
OUT = "/scratch/ud3d4/acm_data/krishna_3d_pack"
N_MESH = 150            # label-only 3D reconstructions from the local store
N_CT = 12              # cases that also ship the CT underlay
ORGAN_LABELS = [1, 2, 4, 13, 14]

li = json.load(open(f"{LAB}/label_index.json")); LFIRST = li["first_offset"]
LIDX = {os.path.basename(n).replace(".nii.gz", ""): (off, cs) for n, off, cs in li["labels"]}
LBIN = f"{LAB}/flare_labels.bin"


def inflate(buf):
    assert buf[:4] == b"PK\x03\x04", repr(buf[:4])
    nl = struct.unpack("<H", buf[26:28])[0]; el = struct.unpack("<H", buf[28:30])[0]
    return zlib.decompress(buf[30 + nl + el:], -15)


def load_label(cid, tmp):
    """full 3D label nii (keeps affine + spacing) from the local byte store."""
    off, cs = LIDX[cid]
    with open(LBIN, "rb") as f:
        f.seek(off - LFIRST); buf = f.read(cs + 300)
    open(tmp, "wb").write(inflate(buf))
    return nib.load(tmp)


def preview(mesh_glbs, titles, out_png):
    """cheap matplotlib 3D render of the meshed organs (subsampled faces) for a quick visual check."""
    n = len(mesh_glbs)
    fig = plt.figure(figsize=(5 * n, 5))
    for k, (glb, title) in enumerate(zip(mesh_glbs, titles)):
        ax = fig.add_subplot(1, n, k + 1, projection="3d")
        if glb and os.path.exists(glb):
            sc = trimesh.load(glb, force="scene")
            allv = []
            for name, g in sc.geometry.items():
                v, f = np.asarray(g.vertices), np.asarray(g.faces)
                if len(f) > 4000:
                    idx = np.linspace(0, len(f) - 1, 4000).astype(int); f = f[idx]
                try:
                    col = np.array(g.visual.face_colors[0][:3]) / 255.0
                except Exception:
                    col = np.array([0.6, 0.6, 0.6])
                pc = Poly3DCollection(v[f], alpha=0.9, linewidths=0)
                pc.set_facecolor(col); ax.add_collection3d(pc); allv.append(v)
            if allv:
                V = np.concatenate(allv)
                for setlim, i in ((ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)):
                    setlim(V[:, i].min(), V[:, i].max())
                ax.set_box_aspect([V[:, i].ptp() for i in range(3)])
        ax.set_title(title, fontsize=11); ax.view_init(elev=15, azim=-70); ax.set_axis_off()
    fig.tight_layout(); fig.savefig(out_png, dpi=85, bbox_inches="tight"); plt.close(fig)


def save_seg(nii_or_data, affine, dst):
    data = nii_or_data if isinstance(nii_or_data, np.ndarray) else np.asarray(nii_or_data.dataobj)
    nib.save(nib.Nifti1Image(data.astype(np.uint8), affine), dst)


manifest = {"cases_mesh": [], "cases_ct": [], "cases_pred": []}
os.makedirs(OUT, exist_ok=True)
tmp = "/dev/shm/_k3d.nii.gz"

# ---- 1. CT-underlay cases (from the 40 full CT+label we already have; tumor cases first) ----
import glob
full_ids = sorted(os.path.basename(f).replace("_label.nii.gz", "")
                  for f in glob.glob(f"{FULL}/*_label.nii.gz"))
def has_tumor(cid):
    return int((np.asarray(nib.load(f"{FULL}/{cid}_label.nii.gz").dataobj) == 14).sum()) > 200
full_ids = sorted(full_ids, key=lambda c: (not has_tumor(c), c))[:N_CT]
for cid in full_ids:
    d = f"{OUT}/cases_ct/{cid}"; os.makedirs(f"{d}/meshes", exist_ok=True)
    ct = nib.load(f"{FULL}/{cid}_ct.nii.gz")
    save_seg(nib.load(f"{FULL}/{cid}_label.nii.gz"), ct.affine, f"{d}/seg_gt.nii.gz")   # align to CT
    nib.save(ct, f"{d}/CT.nii.gz")
    r = meshify(f"{d}/seg_gt.nii.gz", f"{d}/meshes", tag="gt")
    preview([r["glb"]], [f"{cid} — GT organs"], f"{d}/preview.png")
    manifest["cases_ct"].append({"id": cid, "organs": r["organs"], "has_ct": True})
    print(f"[ct] {cid}: {list(r['organs'])}", flush=True)

# ---- 2. GT vs model-prediction pairs (both meshed) ----
pred_ids = sorted(os.path.basename(f).replace(".nii.gz", "")
                  for f in glob.glob(f"{PRED}/ssl_predictions/flare/*.nii.gz"))
for cid in pred_ids:
    d = f"{OUT}/cases_pred/{cid}"; os.makedirs(d, exist_ok=True)
    gt = nib.load(f"{PRED}/ground_truth/flare/{cid}.nii.gz")
    pr = nib.load(f"{PRED}/ssl_predictions/flare/{cid}.nii.gz")
    save_seg(gt, gt.affine, f"{d}/seg_gt.nii.gz"); save_seg(pr, gt.affine, f"{d}/seg_pred.nii.gz")
    rg = meshify(f"{d}/seg_gt.nii.gz", f"{d}/meshes/gt", tag="gt")
    rp = meshify(f"{d}/seg_pred.nii.gz", f"{d}/meshes/pred", tag="pred")
    preview([rg["glb"], rp["glb"]], [f"{cid} — GT", f"{cid} — model prediction"], f"{d}/preview.png")
    manifest["cases_pred"].append({"id": cid, "gt_organs": rg["organs"], "pred_organs": rp["organs"]})
    print(f"[pred] {cid}: gt {list(rg['organs'])} | pred {list(rp['organs'])}", flush=True)

# ---- 3. large mesh library from the LOCAL label store (no download) ----
mesh_ids = [c for c in sorted(LIDX) if c not in set(full_ids)]
made = 0
for cid in mesh_ids:
    if made >= N_MESH:
        break
    try:
        nii = load_label(cid, tmp)
        L = np.asarray(nii.dataobj)
        if not any(int((L == v).sum()) > 60 for v in ORGAN_LABELS):
            continue
        d = f"{OUT}/cases_mesh/{cid}"; os.makedirs(f"{d}/meshes", exist_ok=True)
        save_seg(L, nii.affine, f"{d}/seg.nii.gz")
        r = meshify(f"{d}/seg.nii.gz", f"{d}/meshes", tag="gt")
        if not r["organs"]:
            continue
        manifest["cases_mesh"].append({"id": cid, "organs": r["organs"]})
        made += 1
        if made % 20 == 0:
            print(f"[mesh] {made}/{N_MESH} ...", flush=True)
    except Exception as e:
        print(f"[mesh] {cid} ERR {type(e).__name__}: {str(e)[:60]}", flush=True)

# ---- 4. bundle the scripts + manifest ----
os.makedirs(f"{OUT}/scripts", exist_ok=True)
for s in ("meshify.py", "reconstruct_3d.py"):
    src = f"/home/ud3d4/Desktop/SWOG/src/scripts/{s}"
    if os.path.exists(src):
        import shutil; shutil.copy(src, f"{OUT}/scripts/{s}")
manifest["summary"] = {"cases_ct": len(manifest["cases_ct"]), "cases_pred": len(manifest["cases_pred"]),
                       "cases_mesh": len(manifest["cases_mesh"]),
                       "total_3d_reconstructions": len(manifest["cases_ct"]) + len(manifest["cases_pred"]) + len(manifest["cases_mesh"])}
json.dump(manifest, open(f"{OUT}/MANIFEST.json", "w"), indent=2)
print("\n=== 3D PACK DONE ===", manifest["summary"], flush=True)
print(f"-> {OUT}", flush=True)
