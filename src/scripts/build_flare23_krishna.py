#!/usr/bin/env python3
"""
Deliverables for the FLARE23 cases Krishna sent (in SWOG/data/): run our 5 FLARE organ
SAM3 models (GT-box prompted, semi-oracle — same protocol as the Pancreas/LiTS/FLARE
deliveries) and package as per-patient viewer-ready volumes.

Only cases that have CT + our organ GT are processed (the box prompt needs organ GT).
Output mirrors flare_perpatient_v2:
  ct/flare/<case>.nii.gz   ground_truth/flare/<case>.nii.gz   ssl_predictions/flare/<case>.nii.gz
  flare_manifest.json
GT keeps ALL provided FLARE IDs (incl. tumour where present, for display); prediction covers
the 5 trained organs (liver 1, right kidney 2, spleen 3, pancreas 4, left kidney 13).
"""
import sys, os, json
sys.path.insert(0, "/home/ud3d4/Desktop/SWOG/src/scripts")
import numpy as np
import nibabel as nib
import torch
from transformers import Sam3Processor
from run_pancreas_sam3 import _sam3_infer_slice, _load_sam3_ckpt, SAM3_MODEL_ID, HF_TOKEN
from run_flare_task2_sam3 import window_rgb, _resize, _resize_to, CKPT_DIR, MIN_ORGAN_PX

DATA = "/home/ud3d4/Desktop/SWOG/data"
OUT = "/scratch/ud3d4/acm_data/FLARE_Task2/flare23_krishna"
ORGANS = [(1, "liver"), (2, "right_kidney"), (3, "spleen"), (4, "pancreas"), (13, "left_kidney")]
TUMOR = 14
# cases with CT + our organ GT (the box prompt needs organ GT)
CASES = ["FLARE23_0217", "FLARE23_0491", "FLARE23_0747", "FLARE23_1283", "FLARE23_1823"]
DEV = torch.device("cuda:0")


def dice(p, g):
    p, g = p.astype(bool), g.astype(bool); s = p.sum() + g.sum()
    return round(float(2 * (p & g).sum() / s), 4) if s else None


def main():
    for d in ("ct", "ground_truth", "ssl_predictions"):
        os.makedirs(f"{OUT}/{d}/flare", exist_ok=True)
    proc = Sam3Processor.from_pretrained(SAM3_MODEL_ID, token=HF_TOKEN)

    # load CT + GT once per case
    cases = {}
    for cid in CASES:
        ni = nib.load(f"{DATA}/{cid}_0000.nii.gz")
        lb = nib.load(f"{DATA}/{cid}.nii.gz").get_fdata().astype(np.uint8)
        cases[cid] = {"ni": ni, "ct": ni.get_fdata(), "lab": lb,
                      "pred": np.zeros(lb.shape, np.uint8), "dice": {}}
        print(f"{cid}: CT {ni.shape}  GT labels {sorted(int(x) for x in np.unique(lb) if x>0)}", flush=True)

    # one organ model at a time (memory); gtbox inference on every case
    for oid, name in ORGANS:
        ckpt = f"{CKPT_DIR}/flare_t2_sam3_v3_{name}.pth"
        model = _load_sam3_ckpt(ckpt, DEV)
        for cid, c in cases.items():
            vol, lab = c["ct"], c["lab"]
            gt_o = (lab == oid)
            if gt_o.sum() == 0:
                c["dice"][name] = None; continue
            po = np.zeros(lab.shape, np.uint8)
            for z in range(lab.shape[2]):
                gz = gt_o[:, :, z]
                if gz.sum() < MIN_ORGAN_PX:
                    continue
                rgb = window_rgb(_resize(vol[:, :, z], 1)).astype(np.uint8)
                m256 = (_resize(gz.astype(np.uint8), 0) > 0.5).astype(np.uint8)
                prob = _sam3_infer_slice(model, proc, rgb, m256, DEV)
                if prob is None:
                    continue
                H, W = gz.shape
                po[:, :, z] = (_resize_to(prob, (H, W)) > 0.5).astype(np.uint8)
            c["pred"][po.astype(bool)] = oid           # merge into multi-label pred
            c["dice"][name] = dice(po, gt_o)
            print(f"  [{name}] {cid}: Dice {c['dice'][name]}", flush=True)
        del model; torch.cuda.empty_cache()

    # save + manifest
    manifest = {"dataset": "FLARE23 cases (provided by collaborator)",
                "label_ids": {str(o): n for o, n in ORGANS} | {"14": "tumor (GT only, not predicted)"},
                "notes": ["Predictions cover 5 organs (FLARE22-trained models); NO tumour prediction.",
                          "GT keeps all provided FLARE23 label IDs (incl. tumour where present) for display.",
                          "Predictions GT-box prompted (semi-oracle), same protocol as Pancreas/LiTS/FLARE.",
                          "CT in Hounsfield units; GT/pred share CT shape + affine."],
                "cases": {}}
    for cid, c in cases.items():
        aff, hdr = c["ni"].affine, c["ni"].header
        ct16 = np.rint(c["ct"]).astype(np.int16)
        h = hdr.copy(); h.set_data_dtype(np.int16)
        nib.save(nib.Nifti1Image(ct16, aff, h), f"{OUT}/ct/flare/{cid}.nii.gz")
        hu = hdr.copy(); hu.set_data_dtype(np.uint8)
        nib.save(nib.Nifti1Image(c["lab"], aff, hu), f"{OUT}/ground_truth/flare/{cid}.nii.gz")
        nib.save(nib.Nifti1Image(c["pred"], aff, hu), f"{OUT}/ssl_predictions/flare/{cid}.nii.gz")
        vox = float(np.prod(hdr.get_zooms()[:3])) / 1000.0
        present = sorted(int(x) for x in np.unique(c["lab"]) if x > 0)
        manifest["cases"][cid] = {
            "public_flare23_case_id": cid,
            "shape_xyz": [int(s) for s in c["ct"].shape],
            "voxel_spacing_mm": [round(float(z), 3) for z in hdr.get_zooms()[:3]],
            "gt_labels_present": present,
            "gt_has_tumor": TUMOR in present,
            "dice_per_label_recomputed": c["dice"],
            "predicted_organs": [n for _, n in ORGANS if c["dice"].get(n) is not None]}
    json.dump(manifest, open(f"{OUT}/flare_manifest.json", "w"), indent=2)
    import statistics as st
    for _, name in ORGANS:
        vals = [c["dice"][name] for c in cases.values() if c["dice"].get(name) is not None]
        print(f"mean Dice {name:<13s}: {round(st.mean(vals),4) if vals else '—'}")
    print(f"\nDelivered {len(cases)} cases -> {OUT}")


if __name__ == "__main__":
    main()
