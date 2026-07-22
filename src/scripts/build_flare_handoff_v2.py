#!/usr/bin/env python3
"""
Repackage the FLARE-Task2 (2024) PER-PATIENT segmentation into the viewer-ready handoff
layout used for Pancreas/LiTS — one full volume per patient case, multi-label mask, real
HU CT, correct affine. Replaces the old pre-sliced per-class 2D-stack delivery.

Per test case:
  ct/flare/<case>.nii.gz              full CT volume, Hounsfield units (int16), real affine
  ground_truth/flare/<case>.nii.gz    single multi-label GT (FLARE class IDs, all organs)
  ssl_predictions/flare/<case>.nii.gz single multi-label prediction (same FLARE IDs), same
                                      shape+affine as CT/GT (voxel-aligned side-by-side)
  flare_manifest.json                 case list, labels present, per-case Dice/label (recomputed
                                      + reported), spacing, public FLARE case id, split

FLARE22 label IDs preserved: 1 liver, 2 right kidney, 3 spleen, 4 pancreas, 13 left kidney.
NOTE: FLARE-Task2 (FLARE22) has NO tumour label and duodenum was not trained — the 3D FLARE
view shows these 5 organs. (Tumour existed only in the old pre-sliced FLARE23 set.)
"""
import sys, os, json
sys.path.insert(0, "/home/ud3d4/Desktop/SWOG/src/scripts")
import numpy as np
import nibabel as nib
from run_flare_task2_sam3 import split_pairs

D = "/scratch/ud3d4/acm_data/FLARE_Task2"
DELIV = f"{D}/sam3_delivery"
OUT = f"{D}/flare_handoff_v2"
ID2NAME = {1: "liver", 2: "right_kidney", 3: "spleen", 4: "pancreas", 13: "left_kidney"}
# reported per-patient 3D Dice from training (test n=20)
REPORTED = {"liver": 0.9852, "right_kidney": 0.9695, "spleen": 0.9795,
            "pancreas": 0.9189, "left_kidney": 0.9703}
# assign larger organs first so smaller ones win any rare overlap
MERGE_ORDER = [(1, "liver"), (3, "spleen"), (2, "right_kidney"), (13, "left_kidney"), (4, "pancreas")]


def dice(p, g):
    p, g = p.astype(bool), g.astype(bool)
    i = float((p & g).sum()); s = float(p.sum() + g.sum())
    return round(2 * i / s, 4) if s else None


def cid_of(ip):
    return os.path.basename(ip).replace("_0000.nii.gz", "")


def main():
    tr, va, te = split_pairs()
    for d in ("ct", "ground_truth", "ssl_predictions"):
        os.makedirs(f"{OUT}/{d}/flare", exist_ok=True)

    manifest = {
        "dataset": "FLARE-Task2 2024 (FLARE22), per-patient volumes",
        "label_ids": {str(k): v for k, v in ID2NAME.items()},
        "notes": ["Multi-label mask per case; FLARE22 class IDs preserved.",
                  "FLARE22 has NO tumour label; duodenum not trained -> 5 organs shown.",
                  "CT in Hounsfield units (int16); GT/pred share CT shape + affine.",
                  "Predictions are GT-box prompted (semi-oracle), matching the Pancreas/LiTS gtbox delivery.",
                  "case id = public FLARE-Task2 id (FLARE22_Tr_* = train_gt_label; FLARETs_* = validation public)."],
        "reported_dice_3d_test": REPORTED,
        "splits": {"train": len(tr), "val": len(va), "test": len(te)},
        "cases": {},
    }

    for ip, lp in te:                                  # deliver the 20 test patients
        cid = cid_of(ip)
        ni = nib.load(ip); aff = ni.affine; hdr = ni.header
        ct = np.rint(ni.get_fdata()).astype(np.int16)
        lab = nib.load(lp).get_fdata().astype(np.uint8)

        # GT multi-label (5 organs, FLARE IDs)
        gt = np.zeros(lab.shape, np.uint8)
        for oid in ID2NAME:
            gt[lab == oid] = oid

        # prediction multi-label: merge per-organ delivery preds
        pred = np.zeros(lab.shape, np.uint8)
        per_dice = {}
        for oid, name in MERGE_ORDER:
            pf = f"{DELIV}/{name}/{cid}_pred.nii.gz"
            if not os.path.exists(pf):
                per_dice[name] = None; continue
            pm = nib.load(pf).get_fdata().astype(bool)
            pred[pm] = oid
            per_dice[name] = dice(pm, lab == oid)

        h = hdr.copy(); h.set_data_dtype(np.int16)
        nib.save(nib.Nifti1Image(ct, aff, h), f"{OUT}/ct/flare/{cid}.nii.gz")
        hu = hdr.copy(); hu.set_data_dtype(np.uint8)
        nib.save(nib.Nifti1Image(gt, aff, hu), f"{OUT}/ground_truth/flare/{cid}.nii.gz")
        nib.save(nib.Nifti1Image(pred, aff, hu), f"{OUT}/ssl_predictions/flare/{cid}.nii.gz")

        vox_cm3 = float(np.prod(hdr.get_zooms()[:3])) / 1000.0
        manifest["cases"][cid] = {
            "split": "test",
            "public_flare_case_id": cid,
            "shape_xyz": [int(s) for s in ct.shape],
            "voxel_spacing_mm": [round(float(z), 3) for z in hdr.get_zooms()[:3]],
            "labels_present": [ID2NAME[o] for o in ID2NAME if (gt == o).any()],
            "dice_per_label_recomputed": {ID2NAME[o]: per_dice.get(ID2NAME[o]) for o in ID2NAME},
            "gt_volume_cm3": {ID2NAME[o]: round(int((gt == o).sum()) * vox_cm3, 2) for o in ID2NAME if (gt == o).any()},
            "pred_volume_cm3": {ID2NAME[o]: round(int((pred == o).sum()) * vox_cm3, 2) for o in ID2NAME if (pred == o).any()},
        }
        print(f"  {cid}: shape {ct.shape} organs {manifest['cases'][cid]['labels_present']}", flush=True)

    json.dump(manifest, open(f"{OUT}/flare_manifest.json", "w"), indent=2)
    # cohort mean recomputed dice per organ
    import statistics as st
    means = {}
    for name in ID2NAME.values():
        vals = [c["dice_per_label_recomputed"][name] for c in manifest["cases"].values()
                if c["dice_per_label_recomputed"].get(name) is not None]
        means[name] = round(st.mean(vals), 4) if vals else None
    print(f"\nDelivered {len(manifest['cases'])} test cases -> {OUT}")
    print("Cohort mean recomputed Dice:", means)
    print("Reported (training) Dice:   ", REPORTED)


if __name__ == "__main__":
    main()
