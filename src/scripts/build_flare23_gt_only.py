#!/usr/bin/env python3
"""
Ground-truth-only test batch for the FLARE23 cases Krishna sent — package CT + multi-label GT
in the viewer layout so he can confirm his pipeline renders the FLARE23 format, BEFORE we ask
for anything more. No model inference; whatever labels each case has are delivered as-is.

  ct/flare/<case>.nii.gz            CT, Hounsfield (int16), real affine
  ground_truth/flare/<case>.nii.gz  multi-label GT, all provided FLARE23 IDs preserved
  flare_manifest.json               labels present, volumes, spacing, case id, has_tumor
  previews/<case>_gt_triplanar.png  axial/coronal/sagittal CT + GT overlay
"""
import os, json, glob
import numpy as np
import nibabel as nib
from nibabel.orientations import aff2axcodes
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

DATA = "/home/ud3d4/Desktop/SWOG/data"
OUT = "/scratch/ud3d4/acm_data/FLARE_Task2/flare23_gt_test"
FL = {1: "liver", 2: "right kidney", 3: "spleen", 4: "pancreas", 5: "aorta", 6: "IVC",
      7: "right adrenal", 8: "left adrenal", 9: "gallbladder", 10: "esophagus",
      11: "stomach", 12: "duodenum", 13: "left kidney", 14: "tumor"}
CMAP = plt.get_cmap("tab20")
COLORS = {l: CMAP((i * 2 % 20) / 20)[:3] for i, l in enumerate(FL)}
COLORS[1] = (0.90, 0.28, 0.28); COLORS[14] = (1.0, 0.10, 0.10)  # liver red-ish, tumor bright red
COLORS[4] = (0.96, 0.80, 0.20)  # pancreas yellow
WIN = (-125, 275)


def win(a):
    lo, hi = WIN
    return np.clip((a - lo) / (hi - lo), 0, 1)


def show(ax, ct2d, m2d, title, aspect):
    ax.imshow(win(ct2d).T, cmap="gray", origin="lower", aspect=aspect)
    rgba = np.zeros(m2d.shape + (4,))
    for lid in np.unique(m2d):
        if lid == 0:
            continue
        col = COLORS.get(int(lid), (1, 1, 1))
        mm = m2d == lid
        for c in range(3):
            rgba[..., c][mm] = col[c]
        rgba[..., 3][mm] = 0.5
    ax.imshow(np.transpose(rgba, (1, 0, 2)), origin="lower", aspect=aspect)
    ax.set_title(title, fontsize=9); ax.axis("off")


def preview(cid, ct_img, gt):
    ci = nib.as_closest_canonical(ct_img); ct = ci.get_fdata()
    sx, sy, sz = [float(z) for z in ci.header.get_zooms()[:3]]
    g = nib.as_closest_canonical(nib.Nifti1Image(gt, ct_img.affine)).get_fdata().astype(int)
    idx = np.argwhere(g > 0); cx, cy, cz = [int(round(v)) for v in idx.mean(0)]
    fig, ax = plt.subplots(1, 3, figsize=(11, 4.2))
    show(ax[0], ct[:, :, cz], g[:, :, cz], f"axial z={cz}", sy / sx)
    show(ax[1], ct[:, cy, :], g[:, cy, :], f"coronal y={cy}", sz / sx)
    show(ax[2], ct[cx, :, :], g[cx, :, :], f"sagittal x={cx}", sz / sy)
    present = [int(l) for l in np.unique(g) if l > 0]
    handles = [Patch(color=COLORS.get(l, (1, 1, 1)), label=FL.get(l, l)) for l in present]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 7), fontsize=8, frameon=False)
    fig.suptitle(f"FLARE23 {cid} — CT + ground-truth overlay", fontsize=11)
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    fig.savefig(f"{OUT}/previews/{cid}_gt_triplanar.png", dpi=110); plt.close(fig)


def main():
    for d in ("ct/flare", "ground_truth/flare", "previews"):
        os.makedirs(f"{OUT}/{d}", exist_ok=True)
    cts = sorted(glob.glob(f"{DATA}/FLARE23_*_0000.nii.gz"))
    manifest = {"dataset": "FLARE23 cases (collaborator-provided) — GROUND-TRUTH-ONLY test batch",
                "label_ids": {str(k): v for k, v in FL.items()},
                "notes": ["GROUND TRUTH ONLY — no model prediction (viewer-format test).",
                          "Each case delivered with whatever labels it currently has (some are tumour-only).",
                          "CT in Hounsfield units (int16); GT shares CT shape + affine.",
                          "Predictions (organ GT-vs-pred) follow once full organ labels are available."],
                "cases": {}}
    for cf in cts:
        cid = os.path.basename(cf).replace("_0000.nii.gz", "")
        lf = f"{DATA}/{cid}.nii.gz"
        if not os.path.exists(lf):
            print(f"  SKIP {cid}: no label"); continue
        ni = nib.load(cf); aff, hdr = ni.affine, ni.header
        ct = np.rint(ni.get_fdata()).astype(np.int16)
        gt = nib.load(lf).get_fdata().astype(np.uint8)
        h = hdr.copy(); h.set_data_dtype(np.int16)
        nib.save(nib.Nifti1Image(ct, aff, h), f"{OUT}/ct/flare/{cid}.nii.gz")
        hu = hdr.copy(); hu.set_data_dtype(np.uint8)
        nib.save(nib.Nifti1Image(gt, aff, hu), f"{OUT}/ground_truth/flare/{cid}.nii.gz")
        preview(cid, ni, gt)
        vox = float(np.prod(hdr.get_zooms()[:3])) / 1000.0
        present = sorted(int(x) for x in np.unique(gt) if x > 0)
        manifest["cases"][cid] = {
            "public_flare23_case_id": cid,
            "shape_xyz": [int(s) for s in ct.shape],
            "voxel_spacing_mm": [round(float(z), 3) for z in hdr.get_zooms()[:3]],
            "orientation": "".join(aff2axcodes(aff)),
            "gt_labels_present": [FL.get(l, l) for l in present],
            "gt_label_ids": present,
            "has_tumor": 14 in present,
            "gt_volume_cm3": {FL.get(l, l): round(int((gt == l).sum()) * vox, 2) for l in present}}
        print(f"  {cid}: {ct.shape}  labels {[FL.get(l, l) for l in present]}", flush=True)
    json.dump(manifest, open(f"{OUT}/flare_manifest.json", "w"), indent=2)
    print(f"\nGT-only test batch: {len(manifest['cases'])} cases -> {OUT}")


if __name__ == "__main__":
    main()
