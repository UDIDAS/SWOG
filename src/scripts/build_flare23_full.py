#!/usr/bin/env python3
"""
Full FLARE23 deliverable for the collaborator's cases: GT + model PREDICTION (organs + tumour),
per-patient viewer-ready volumes, semi-oracle (GT-box), same protocol as Pancreas/LiTS/FLARE.

Organs (FLARE22-trained, GT-box): liver 1, right kidney 2, spleen 3, pancreas 4, left kidney 13.
Tumour (FLARE23 slice-level model, GT-box, val Dice ~0.90): label 14.
Window (-125,225) HU for both (calibrated: ~0.90 tumour Dice, robust to window).

Per case, predict whichever of these labels the GT contains. Output mirrors the prior FLARE set:
  ct/flare/<case>.nii.gz  ground_truth/flare/<case>.nii.gz  ssl_predictions/flare/<case>.nii.gz
  flare_manifest.json  previews/<case>_triplanar.png
"""
import sys, os, json, glob
sys.path.insert(0, "/home/ud3d4/Desktop/SWOG/src/scripts")
import numpy as np, nibabel as nib, torch
from nibabel.orientations import aff2axcodes
from transformers import Sam3Processor
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from run_pancreas_sam3 import _sam3_infer_slice, _load_sam3_ckpt, SAM3_MODEL_ID, HF_TOKEN
from run_flare_task2_sam3 import _resize, _resize_to, MIN_ORGAN_PX

DATA = "/home/ud3d4/Desktop/SWOG/data"
OUT = "/scratch/ud3d4/acm_data/FLARE_Task2/flare23_full"
ORGAN_CKPT = "/scratch/ud3d4/acm_data/FLARE_Task2/sam3/flare_t2_sam3_v3_{}.pth"
TUMOR_CKPT = "/scratch/ud3d4/acm_data/FLARE/sam3/flare_sam3_v3_tumor.pth"
WIN = (-125, 225); DEV = torch.device("cuda:0")
# (label id, name, checkpoint) — organs first, tumour LAST so tumour wins overlaps
TARGETS = [(1, "liver", ORGAN_CKPT.format("liver")), (2, "right_kidney", ORGAN_CKPT.format("right_kidney")),
           (3, "spleen", ORGAN_CKPT.format("spleen")), (4, "pancreas", ORGAN_CKPT.format("pancreas")),
           (13, "left_kidney", ORGAN_CKPT.format("left_kidney")), (14, "tumor", TUMOR_CKPT)]
FL = {1: "liver", 2: "right kidney", 3: "spleen", 4: "pancreas", 5: "aorta", 6: "IVC", 7: "right adrenal",
      8: "left adrenal", 9: "gallbladder", 10: "esophagus", 11: "stomach", 12: "duodenum",
      13: "left kidney", 14: "tumor"}
CMAP = plt.get_cmap("tab20"); COLORS = {l: CMAP((i * 2 % 20) / 20)[:3] for i, l in enumerate(FL)}
COLORS[1] = (0.90, 0.28, 0.28); COLORS[4] = (0.96, 0.80, 0.20); COLORS[14] = (1.0, 0.10, 0.10)


def win_rgb(sl):
    lo, hi = WIN; x = np.clip(sl, lo, hi); x = ((x - lo) / (hi - lo) * 255).astype(np.uint8)
    return np.repeat(x[..., None], 3, axis=2)


def dice(p, g):
    p, g = p.astype(bool), g.astype(bool); s = p.sum() + g.sum()
    return round(float(2 * (p & g).sum() / s), 4) if s else None


def show(ax, ct2d, m2d, title, aspect):
    lo, hi = WIN
    ax.imshow(np.clip((ct2d - lo) / (hi - lo), 0, 1).T, cmap="gray", origin="lower", aspect=aspect)
    rgba = np.zeros(m2d.shape + (4,))
    for lid in np.unique(m2d):
        if lid == 0:
            continue
        col = COLORS.get(int(lid), (1, 1, 1)); mm = m2d == lid
        for c in range(3):
            rgba[..., c][mm] = col[c]
        rgba[..., 3][mm] = 0.5
    ax.imshow(np.transpose(rgba, (1, 0, 2)), origin="lower", aspect=aspect)
    ax.set_title(title, fontsize=9); ax.axis("off")


def preview(cid, ni, gt, pred):
    ci = nib.as_closest_canonical(ni); ct = ci.get_fdata()
    sx, sy, sz = [float(z) for z in ci.header.get_zooms()[:3]]
    g = nib.as_closest_canonical(nib.Nifti1Image(gt, ni.affine)).get_fdata().astype(int)
    p = nib.as_closest_canonical(nib.Nifti1Image(pred, ni.affine)).get_fdata().astype(int)
    idx = np.argwhere(g > 0); cx, cy, cz = [int(round(v)) for v in idx.mean(0)]
    fig, ax = plt.subplots(2, 3, figsize=(11, 7.6))
    for r, (m, tag) in enumerate([(g, "GT"), (p, "Pred")]):
        show(ax[r][0], ct[:, :, cz], m[:, :, cz], f"{tag} axial z={cz}", sy / sx)
        show(ax[r][1], ct[:, cy, :], m[:, cy, :], f"{tag} coronal y={cy}", sz / sx)
        show(ax[r][2], ct[cx, :, :], m[cx, :, :], f"{tag} sagittal x={cx}", sz / sy)
    present = [int(l) for l in np.unique(g) if l > 0]
    handles = [Patch(color=COLORS.get(l, (1, 1, 1)), label=FL.get(l, l)) for l in present]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 7), fontsize=8, frameon=False)
    fig.suptitle(f"FLARE23 {cid} — CT + overlay (GT top, prediction bottom)", fontsize=11)
    fig.tight_layout(rect=[0, 0.06, 1, 0.96]); fig.savefig(f"{OUT}/previews/{cid}_triplanar.png", dpi=110); plt.close(fig)


def main():
    for d in ("ct/flare", "ground_truth/flare", "ssl_predictions/flare", "previews"):
        os.makedirs(f"{OUT}/{d}", exist_ok=True)
    proc = Sam3Processor.from_pretrained(SAM3_MODEL_ID, token=HF_TOKEN)
    cases = {}
    for cf in sorted(glob.glob(f"{DATA}/FLARE23_*_0000.nii.gz")):
        cid = os.path.basename(cf).replace("_0000.nii.gz", "")
        lf = f"{DATA}/{cid}.nii.gz"
        if not os.path.exists(lf):
            continue
        ni = nib.load(cf)
        cases[cid] = {"ni": ni, "ct": ni.get_fdata(), "lab": nib.load(lf).get_fdata().astype(np.uint8),
                      "pred": np.zeros(ni.shape, np.uint8), "dice": {}}
    print("cases:", list(cases))

    for oid, name, ckpt in TARGETS:
        need = [cid for cid, c in cases.items() if (c["lab"] == oid).any()]
        if not need:
            continue
        model = _load_sam3_ckpt(ckpt, DEV)
        for cid in need:
            c = cases[cid]; vol = c["ct"]; go = (c["lab"] == oid)
            po = np.zeros(go.shape, np.uint8)
            for z in range(go.shape[2]):
                gz = go[:, :, z]
                if gz.sum() < MIN_ORGAN_PX:
                    continue
                rgb = win_rgb(_resize(vol[:, :, z], 1)).astype(np.uint8)
                m256 = (_resize(gz.astype(np.uint8), 0) > 0.5).astype(np.uint8)
                prob = _sam3_infer_slice(model, proc, rgb, m256, DEV)
                if prob is not None:
                    H, W = gz.shape; po[:, :, z] = (_resize_to(prob, (H, W)) > 0.5).astype(np.uint8)
            c["pred"][po.astype(bool)] = oid                 # organs then tumour (tumour wins)
            c["dice"][FL[oid]] = dice(po, go)
            print(f"  [{name}] {cid}: Dice {c['dice'][FL[oid]]}", flush=True)
        del model; torch.cuda.empty_cache()

    manifest = {"dataset": "FLARE23 cases (collaborator-provided) — GT + prediction",
                "label_ids": {str(k): v for k, v in FL.items()},
                "predicted_labels": {"organs": [1, 2, 3, 4, 13], "tumor": 14},
                "notes": ["Predictions GT-box prompted (semi-oracle), window (-125,225) HU.",
                          "Organs: FLARE22-trained SAM3 models. Tumour: FLARE23 slice-level SAM3 model.",
                          "GT keeps all provided FLARE23 IDs; prediction covers the labels each case's GT contains.",
                          "CT in Hounsfield units (int16); GT/pred share CT shape + affine."],
                "cases": {}}
    for cid, c in cases.items():
        aff, hdr = c["ni"].affine, c["ni"].header
        h = hdr.copy(); h.set_data_dtype(np.int16)
        nib.save(nib.Nifti1Image(np.rint(c["ct"]).astype(np.int16), aff, h), f"{OUT}/ct/flare/{cid}.nii.gz")
        hu = hdr.copy(); hu.set_data_dtype(np.uint8)
        nib.save(nib.Nifti1Image(c["lab"], aff, hu), f"{OUT}/ground_truth/flare/{cid}.nii.gz")
        nib.save(nib.Nifti1Image(c["pred"], aff, hu), f"{OUT}/ssl_predictions/flare/{cid}.nii.gz")
        preview(cid, c["ni"], c["lab"], c["pred"])
        vox = float(np.prod(hdr.get_zooms()[:3])) / 1000.0
        present = sorted(int(x) for x in np.unique(c["lab"]) if x > 0)
        manifest["cases"][cid] = {"public_flare23_case_id": cid, "shape_xyz": [int(s) for s in c["ct"].shape],
            "voxel_spacing_mm": [round(float(z), 3) for z in hdr.get_zooms()[:3]],
            "orientation": "".join(aff2axcodes(aff)),
            "gt_labels_present": [FL.get(l, l) for l in present], "has_tumor": 14 in present,
            "dice_per_label_recomputed": c["dice"],
            "predicted_labels": [k for k, v in c["dice"].items() if v is not None]}
        print(f"  saved {cid}: pred labels {manifest['cases'][cid]['predicted_labels']}", flush=True)
    json.dump(manifest, open(f"{OUT}/flare_manifest.json", "w"), indent=2)
    print(f"\nFull FLARE23 deliverable: {len(cases)} cases -> {OUT}")


if __name__ == "__main__":
    main()
