#!/usr/bin/env python3
"""Patient-level (3-D whole-volume) AUSAM eval — the honest deployment number + the phenotype bridge.

For each held-out TEST patient (same seed-42 per-dataset split as training), run the dataset's AUSAM model
slice-by-slice over the ORIGINAL NIfTI volume — semi-oracle (GT box + organ text on organ-present slices),
exactly the training preprocessing (HU window (-125,225), 256x256, label per pool) — assemble the per-slice
masks into a 3-D volume at native resolution, and score **3-D Dice** vs the GT volume. Also emit per-patient
**phenotypes** (volume cm³ using real spacing, predicted vs GT) — the input to the KG stage.

Precedent: the original AUSAM (run_pancreas_nifti.py) did exactly this patient-level assembly, but with SAM1 +
points. This reproduces it for the SAM3 + box AUSAM models. -> results/ausam_3d_<dataset>.json

  python eval_ausam_3d.py --dataset msd [--limit N]
"""
import argparse
import json
import sys
from collections import defaultdict

import numpy as np
import nibabel as nib
import torch
from torch.amp import autocast
from skimage.transform import resize

sys.path.insert(0, "/home/ud3d4/Desktop/SWOG/src/scripts")
from train_organ_generic import patient_split, POOL
from run_pancreas_sam3 import bbox_from_mask, extract_best_mask_soft, _load_sam3_ckpt, SAM3_MODEL_ID, HF_TOKEN

WIN = (-125, 225)                       # HU window used to build the organ pool (must match training)
MINPX = 50                              # min organ px (on the 256 mask) to count a slice as organ-present
DEV = "cuda:0"
PAN = "/scratch/ud3d4/acm_data/Pancreas"
FT2 = "/scratch/ud3d4/acm_data/FLARE_Task2"
import glob
import os


def load_msd(c):
    img = nib.load(f"{PAN}/imagesTr/{c}.nii.gz")
    seg = np.asarray(nib.load(f"{PAN}/labelsTr/{c}.nii.gz").dataobj).astype(int)
    return img.get_fdata(), seg, [float(z) for z in img.header.get_zooms()[:3]]


def load_flare_task2(c):
    lp = None
    for p in (f"{FT2}/train_gt_label/labelsTr/{c}.nii.gz", f"{FT2}/validation/Validation-Public-Labels/{c}.nii.gz"):
        if os.path.exists(p):
            lp = p; break
    ip = lp.replace("labelsTr", "imagesTr").replace("Validation-Public-Labels", "Validation-Public-Images") \
           .replace(".nii.gz", "_0000.nii.gz")
    img = nib.load(ip)
    seg = np.asarray(nib.load(lp).dataobj).astype(int)
    return img.get_fdata(), seg, [float(z) for z in img.header.get_zooms()[:3]]


def load_lits(c):                        # c = "volume-N"; .npy volumes carry NO spacing
    vid = c.replace("volume-", "")
    ct = np.load(f"/scratch/ud3d4/acm_data/Data/ct/volume-{vid}.npy")
    seg = np.load(f"/scratch/ud3d4/acm_data/Data/seg/segmentation-{vid}.npy").astype(int)
    return ct, seg, None


FFC = "/scratch/ud3d4/acm_data/flare_full_cases"   # FLARE23 full volumes (local subset)


def load_flare23(c):
    img = nib.load(f"{FFC}/{c}_ct.nii.gz")
    seg = np.asarray(nib.load(f"{FFC}/{c}_label.nii.gz").dataobj).astype(int)
    return img.get_fdata(), seg, [float(z) for z in img.header.get_zooms()[:3]]


# per-dataset config: organ -> GT label(s); volume loader; slice axis
CFG = {
    "msd":         {"ckpt": f"{POOL}/sam3_organ_generic_ausam_msd.pth", "axial": 2,
                    "organs": {"pancreas": [1]}, "load": load_msd},
    "flare_task2": {"ckpt": f"{POOL}/sam3_organ_generic_ausam_flare_task2.pth", "axial": 2,
                    "organs": {"liver": [1], "kidney": [2, 13], "pancreas": [4]}, "load": load_flare_task2},
    "lits":        {"ckpt": f"{POOL}/sam3_organ_generic_ausam_lits.pth", "axial": 0,
                    "organs": {"liver": [1]}, "load": load_lits},
    "flare23":     {"ckpt": f"{POOL}/sam3_organ_generic_ausam_flare23.pth", "axial": 2,
                    "organs": {"liver": [1], "kidney": [2, 13], "pancreas": [4]}, "load": load_flare23,
                    "avail": lambda: set(os.path.basename(f).replace("_ct.nii.gz", "")
                                         for f in glob.glob(f"{FFC}/*_ct.nii.gz"))},
}


def hu_rgb(sl):
    lo, hi = WIN
    x = np.clip(sl, lo, hi); x = ((x - lo) / (hi - lo) * 255).astype(np.uint8)
    return np.stack([x] * 3, -1)


def _slice(vol, ax, z):
    return vol[z] if ax == 0 else vol[:, :, z]


from scipy.ndimage import distance_transform_edt, binary_erosion
NSD_TAUS = (1.0, 2.0)                    # surface-distance tolerances (mm)


def _surf_dt(mask, spacing):
    surf = mask & ~binary_erosion(mask)                      # 1-voxel-thick boundary
    return surf, distance_transform_edt(~surf, sampling=spacing)  # mm distance to nearest surface voxel


def nsd(pred, gt, spacing, taus=NSD_TAUS):
    """Normalized Surface Distance (surface Dice): fraction of both surfaces within tolerance tau (mm)."""
    ps, pdt = _surf_dt(pred, spacing); gs, gdt = _surf_dt(gt, spacing)
    tot = int(ps.sum()) + int(gs.sum())
    if tot == 0:
        return {f"{t}mm": 1.0 for t in taus}
    if ps.sum() == 0 or gs.sum() == 0:
        return {f"{t}mm": 0.0 for t in taus}
    out = {}
    for t in taus:
        within = int((pdt[gs] <= t).sum()) + int((gdt[ps] <= t).sum())
        out[f"{t}mm"] = round(within / tot, 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="msd"); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    cfg = CFG[a.dataset]
    from transformers import Sam3Processor
    proc = Sam3Processor.from_pretrained(SAM3_MODEL_ID, token=HF_TOKEN)
    model = _load_sam3_ckpt(cfg["ckpt"], DEV)

    M = json.load(open(f"{POOL}/meta.json"))
    Mf = [m for m in M if m["dataset"] == a.dataset]
    _, _, te = patient_split(Mf)
    test_cases = sorted({Mf[i]["case"] for i in te})
    if "avail" in cfg:                       # partial local volumes (e.g. FLARE23): keep those we have
        av = cfg["avail"](); n0 = len(test_cases)
        test_cases = [c for c in test_cases if c in av]
        print(f"  (local volumes cover {len(test_cases)}/{n0} test patients)", flush=True)
    if a.limit:
        test_cases = test_cases[:a.limit]
    print(f"3-D AUSAM [{a.dataset}] {len(test_cases)} held-out test patients; organs {list(cfg['organs'])}", flush=True)

    def predict_slice(rgb, organ, box):
        inp = proc(images=[rgb], text=[organ], input_boxes=[[box]], input_boxes_labels=[[1]], return_tensors="pt")
        kw = {"pixel_values": inp["pixel_values"].to(DEV)}
        for k in ("input_ids", "attention_mask", "input_boxes", "input_boxes_labels"):
            if inp.get(k) is not None:
                kw[k] = inp[k].to(DEV)
        with torch.no_grad(), autocast("cuda"):
            out = model(**kw)
            pm = extract_best_mask_soft(out.pred_masks, out.pred_logits)
            pm = torch.nn.functional.interpolate(pm.float(), size=(256, 256), mode="bilinear", align_corners=False)
        return pm.sigmoid().squeeze().cpu().numpy() > 0.5

    per_organ = defaultdict(list); per_organ_nsd = defaultdict(list); cases_out = []
    for ci, case in enumerate(test_cases):
        ct, lbl, sp = cfg["load"](case)
        vox_cm3 = (sp[0] * sp[1] * sp[2] / 1000.0) if sp else None
        ax = cfg["axial"]; Z = ct.shape[ax]
        rec = {"case": case, "spacing": sp, "organs": {}}
        for organ, labs in cfg["organs"].items():
            pred3d = np.zeros(ct.shape, np.uint8)
            gt3d = np.isin(lbl, labs).astype(np.uint8)
            for z in range(Z):
                ct_s = _slice(ct, ax, z); seg_s = _slice(lbl, ax, z)
                gm256 = resize(np.isin(seg_s, labs).astype(float), (256, 256), order=0, preserve_range=True) > 0.5
                if gm256.sum() < MINPX:
                    continue
                rgb = hu_rgb(resize(ct_s, (256, 256), preserve_range=True, anti_aliasing=True))
                box = bbox_from_mask(gm256.astype(np.uint8), pad=3)
                pr256 = predict_slice(rgb, organ, box if box is not None else [0, 0, 255, 255])
                pr = resize(pr256.astype(float), ct_s.shape, order=0, preserve_range=True) > 0.5
                if ax == 0:
                    pred3d[z] = pr
                else:
                    pred3d[:, :, z] = pr
            inter = int((pred3d & gt3d).sum()); s = int(pred3d.sum()) + int(gt3d.sum())
            d3 = 2 * inter / s if s else 1.0
            per_organ[organ].append(d3)
            rec["organs"][organ] = {"dice3d": round(d3, 4),
                                    "vol_pred_cm3": round(int(pred3d.sum()) * vox_cm3, 2) if vox_cm3 else None,
                                    "vol_gt_cm3": round(int(gt3d.sum()) * vox_cm3, 2) if vox_cm3 else None,
                                    "vox_pred": int(pred3d.sum()), "vox_gt": int(gt3d.sum())}
            if sp is not None:                       # NSD needs mm spacing (skip .npy/LiTS)
                nv = nsd(pred3d.astype(bool), gt3d.astype(bool), sp)
                rec["organs"][organ]["nsd3d"] = nv
                per_organ_nsd[organ].append(nv["2.0mm"])
        cases_out.append(rec)
        if (ci + 1) % 5 == 0 or ci == 0:
            print(f"  [{ci+1}/{len(test_cases)}] {case}: " +
                  ", ".join(f"{o} 3Ddice={v['dice3d']} volP/G={v['vol_pred_cm3']}/{v['vol_gt_cm3']}cc"
                            for o, v in rec["organs"].items()), flush=True)

    summary = {o: round(float(np.mean(v)), 4) for o, v in per_organ.items()}
    nsd_summary = {o: round(float(np.mean(v)), 4) for o, v in per_organ_nsd.items() if v}
    out = {"dataset": a.dataset, "n_patients": len(test_cases), "hu_window": WIN, "nsd_taus_mm": list(NSD_TAUS),
           "mean_3d_dice": summary, "mean_nsd_2mm": nsd_summary, "cases": cases_out}
    fp = f"/home/ud3d4/Desktop/SWOG/results/ausam_3d_{a.dataset}.json"
    json.dump(out, open(fp, "w"), indent=2)
    print(f"\n=== {a.dataset} patient-level 3-D  DSC {summary}  |  NSD@2mm {nsd_summary}  (n={len(test_cases)}) ===", flush=True)
    print(f"-> {fp}", flush=True)


if __name__ == "__main__":
    main()
