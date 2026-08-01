#!/usr/bin/env python3
"""Produce PREDICTED FLARE23 phenotypes for the gt+pred handoff KG.
  organs : gtbox (semi-oracle) SAM3 per-organ models flare_t2_sam3_v3_{organ}.pth  (same protocol
           as the delivered Pancreas/LiTS predicted KG) -> pred_volume_cm3 + Dice vs GT
  tumor  : the generic text-'tumor' model (cross-dataset!) -> pred tumor, kept inside predicted organs
Outputs a pred corpus keyed by case_id, merged later by build_flare23_enriched_kg with pred_.

Usage: python flare23_predict.py CID1 CID2 ...        (images must be at data/<cid>_0000.nii.gz)
       python flare23_predict.py --all-local          (all local FLARE23 image+GT pairs)
"""
import glob
import json
import os
import sys

import nibabel as nib
import numpy as np
import torch
from torch.amp import autocast
from transformers import Sam3Processor

sys.path.insert(0, "/home/ud3d4/Desktop/SWOG/src/scripts")
from run_flare_task2_sam3 import window_rgb, _resize, _resize_to, CKPT_DIR, MIN_ORGAN_PX
from run_pancreas_sam3 import _sam3_infer_slice, _load_sam3_ckpt, SAM3_MODEL_ID, HF_TOKEN, extract_best_mask_soft

DATA = "/home/ud3d4/Desktop/SWOG/data"
DEV = "cuda:0"
ORGANS = [("liver", 1), ("right_kidney", 2), ("spleen", 3), ("pancreas", 4), ("left_kidney", 13)]
TUMOR_LABEL = 14
TUMOR_CKPT = os.environ.get("TUMOR_CKPT", "/scratch/ud3d4/acm_data/tumor_pool/sam3_tumor_generic_v3.pth")
OUT = "/scratch/ud3d4/acm_data/flare23_labels/flare23_pred.json"
# viewer-layout predicted masks for Krishna (ssl_predictions/flare/<cid>.nii.gz, matching the delivery)
VIEW = "/scratch/ud3d4/acm_data/flare23_pred_masks"


def dice(p, g):
    p, g = np.asarray(p).astype(bool), np.asarray(g).astype(bool)
    s = int(p.sum()) + int(g.sum())
    return round(2 * int((p & g).sum()) / s, 4) if s else None


def tumor_slice(model, proc, rgb):
    inputs = proc(images=[rgb], text=["tumor"], return_tensors="pt")
    kw = {"pixel_values": inputs["pixel_values"].to(DEV)}
    for k in ("input_ids", "attention_mask"):
        if inputs.get(k) is not None:
            kw[k] = inputs[k].to(DEV)
    with torch.no_grad(), autocast("cuda"):
        out = model(**kw)
        pm = extract_best_mask_soft(out.pred_masks, out.pred_logits)
        pm = torch.nn.functional.interpolate(pm.float(), size=(256, 256), mode="bilinear", align_corners=False)
    return pm.sigmoid().squeeze().cpu().numpy()


def main(cids):
    proc = Sam3Processor.from_pretrained(SAM3_MODEL_ID, token=HF_TOKEN)
    cases = {}
    for cid in cids:
        nii = nib.load(f"{DATA}/{cid}_0000.nii.gz")
        cases[cid] = {"nii": nii, "ct": nii.get_fdata(),
                      "gt": nib.load(f"{DATA}/{cid}.nii.gz").get_fdata().astype(np.uint8),
                      "sp": float(np.prod(nii.header.get_zooms()[:3])) / 1000.0,
                      "porg": {}, "organs": {}}
        print(f"loaded {cid} {cases[cid]['ct'].shape}", flush=True)

    # ---- organs: gtbox per organ model ----
    for oname, oid in ORGANS:
        model = _load_sam3_ckpt(f"{CKPT_DIR}/flare_t2_sam3_v3_{oname}.pth", DEV)
        for cid, c in cases.items():
            gt_o = (c["gt"] == oid)
            if gt_o.sum() == 0:
                continue
            po = np.zeros(c["gt"].shape, np.uint8)
            for z in range(c["gt"].shape[2]):
                gz = gt_o[:, :, z]
                if gz.sum() < MIN_ORGAN_PX:
                    continue
                rgb = window_rgb(_resize(c["ct"][:, :, z], 1)).astype(np.uint8)
                m256 = (_resize(gz.astype(np.uint8), 0) > 0.5).astype(np.uint8)
                prob = _sam3_infer_slice(model, proc, rgb, m256, DEV)
                if prob is None:
                    continue
                H, W = gz.shape
                po[:, :, z] = (_resize_to(prob, (H, W)) > 0.5).astype(np.uint8)
            c["porg"][oname] = po
            c["organs"][oname] = {"gt_volume_cm3": round(int(gt_o.sum()) * c["sp"], 2),
                                  "pred_volume_cm3": round(int(po.sum()) * c["sp"], 2),
                                  "dice": dice(po, gt_o)}
            print(f"  organ {oname:13s} {cid}: Dice {c['organs'][oname]['dice']}", flush=True)
        del model
        torch.cuda.empty_cache()

    # ---- tumor: generic text-'tumor' model, keep inside predicted organs, attribute by overlap ----
    tmodel = _load_sam3_ckpt(TUMOR_CKPT, DEV) if os.path.exists(TUMOR_CKPT) else None
    if tmodel is not None:
        for cid, c in cases.items():
            union = np.zeros(c["gt"].shape, bool)
            for po in c["porg"].values():
                union |= (po > 0)
            gt_t = (c["gt"] == 14)
            tm = np.zeros(c["gt"].shape, np.uint8)
            for z in range(c["gt"].shape[2]):
                rgb = window_rgb(_resize(c["ct"][:, :, z], 1)).astype(np.uint8)
                H, W = c["gt"].shape[:2]
                tm[:, :, z] = (_resize_to(tumor_slice(tmodel, proc, rgb), (H, W)) > 0.5).astype(np.uint8)
            kept = (tm > 0) & union
            c["ptumor"] = kept
            best, torg = 0, None
            for oname, po in c["porg"].items():
                ov = int((kept & (po > 0)).sum())
                if ov > best:
                    best, torg = ov, oname
            c["tumor"] = {"gt_tumor_volume_cm3": round(int(gt_t.sum()) * c["sp"], 2),
                          "pred_tumor_volume_cm3": round(int(kept.sum()) * c["sp"], 2),
                          "tumor_dice": dice(kept, gt_t), "attributed_organ": torg}
            print(f"  tumor {cid}: gt={c['tumor']['gt_tumor_volume_cm3']} "
                  f"pred={c['tumor']['pred_tumor_volume_cm3']} Dice={c['tumor']['tumor_dice']} -> {torg}", flush=True)

    # ---- save viewer-layout predicted + GT multi-label mask NIfTIs (what Krishna's app renders) ----
    os.makedirs(f"{VIEW}/ssl_predictions/flare", exist_ok=True)
    os.makedirs(f"{VIEW}/ground_truth/flare", exist_ok=True)
    for cid, c in cases.items():
        ml = np.zeros(c["gt"].shape, np.uint8)
        for oname, oid in ORGANS:                     # organs first, tumor overwrites inside
            po = c["porg"].get(oname)
            if po is not None:
                ml[po > 0] = oid
        if c.get("ptumor") is not None:
            ml[c["ptumor"]] = TUMOR_LABEL
        aff, hdr = c["nii"].affine, c["nii"].header
        h = hdr.copy(); h.set_data_dtype(np.uint8)
        nib.save(nib.Nifti1Image(ml, aff, h), f"{VIEW}/ssl_predictions/flare/{cid}.nii.gz")
        nib.save(nib.Nifti1Image(c["gt"], aff, h), f"{VIEW}/ground_truth/flare/{cid}.nii.gz")
    print(f"saved predicted+GT masks -> {VIEW}", flush=True)

    out = {cid: {"organs": c["organs"], "tumor": c.get("tumor")} for cid, c in cases.items()}
    prev = json.load(open(OUT)) if os.path.exists(OUT) else {}
    prev.update(out)
    json.dump(prev, open(OUT, "w"))
    # summary
    for oname, _ in ORGANS:
        ds = [c["organs"][oname]["dice"] for c in cases.values() if oname in c["organs"] and c["organs"][oname]["dice"] is not None]
        if ds:
            print(f"mean organ Dice {oname:13s}: {np.mean(ds):.4f} (n={len(ds)})", flush=True)
    tds = [c["tumor"]["tumor_dice"] for c in cases.values() if c.get("tumor") and c["tumor"]["tumor_dice"] is not None]
    if tds:
        print(f"mean tumor Dice (generic model, cross-dataset): {np.mean(tds):.4f} (n={len(tds)})", flush=True)
    print(f"-> {OUT} ({len(prev)} cases total)", flush=True)


if __name__ == "__main__":
    if "--all-local" in sys.argv:
        cids = sorted({os.path.basename(f).replace("_0000.nii.gz", "")
                       for f in glob.glob(f"{DATA}/FLARE23_*_0000.nii.gz")
                       if os.path.exists(f.replace("_0000.nii.gz", ".nii.gz"))})
    else:
        cids = [a for a in sys.argv[1:] if a.startswith("FLARE23_")]
    print(f"predicting {len(cids)} cases", flush=True)
    main(cids)
