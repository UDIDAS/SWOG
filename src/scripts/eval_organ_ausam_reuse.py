#!/usr/bin/env python3
"""AUSAM ceiling via REUSED weights — the existing FLARE-Task2 box-fine-tuned per-organ models
(semi-oracle: they gave §4a 0.985/0.970/0.919). Score them with a GT box on OUR pooled organ test set,
per (organ, dataset), so we can see how a box model trained on ONE dataset holds up on the new pooled data
(esp. KiTS/LiTS/MSD, which these models never saw). Kidney = union of the split L/R models vs the merged mask.
Same pool + patient split as train_organ_generic. Needs one GPU. -> results/organ_ausam_reuse.json
"""
import json
import sys
from collections import defaultdict

import numpy as np
import torch
from torch.amp import autocast

sys.path.insert(0, "/home/ud3d4/Desktop/SWOG/src/scripts")
from train_organ_generic import patient_split, POOL
from run_pancreas_sam3 import bbox_from_mask, extract_best_mask_soft, _load_sam3_ckpt, SAM3_MODEL_ID, HF_TOKEN

FT2 = "/scratch/ud3d4/acm_data/FLARE_Task2/sam3"
CKPTS = {                                   # organ -> (checkpoint, text) list (kidney = 2 models, unioned)
    "liver":    [(f"{FT2}/flare_t2_sam3_v3_liver.pth", "liver")],
    "pancreas": [(f"{FT2}/flare_t2_sam3_v3_pancreas.pth", "pancreas")],
    "kidney":   [(f"{FT2}/flare_t2_sam3_v3_right_kidney.pth", "right kidney"),
                 (f"{FT2}/flare_t2_sam3_v3_left_kidney.pth", "left kidney")],
}


def main():
    X = np.load(f"{POOL}/images.npy"); Y = np.load(f"{POOL}/masks.npy"); M = json.load(open(f"{POOL}/meta.json"))
    tr, va, te = patient_split(M)
    from transformers import Sam3Processor
    proc = Sam3Processor.from_pretrained(SAM3_MODEL_ID, token=HF_TOKEN)
    dev = "cuda:0"
    cache = {}                              # load each unique checkpoint once
    for specs in CKPTS.values():
        for ckpt, _ in specs:
            if ckpt not in cache:
                cache[ckpt] = _load_sam3_ckpt(ckpt, dev)
    print(f"reuse-AUSAM over {len(te)} test slices; {len(cache)} box models loaded", flush=True)

    def predict(model, img, box, text):
        inp = proc(images=[img], text=[text], input_boxes=[[box]], input_boxes_labels=[[1]], return_tensors="pt")
        kw = {"pixel_values": inp["pixel_values"].to(dev)}
        for k in ("input_ids", "attention_mask", "input_boxes", "input_boxes_labels"):
            if inp.get(k) is not None:
                kw[k] = inp[k].to(dev)
        with torch.no_grad(), autocast("cuda"):
            out = model(**kw)
            pm = extract_best_mask_soft(out.pred_masks, out.pred_logits)
            pm = torch.nn.functional.interpolate(pm.float(), size=(256, 256), mode="bilinear", align_corners=False)
        return pm.sigmoid().squeeze().cpu().numpy() > 0.5

    dd = defaultdict(list); byo = defaultdict(list)
    for n, i in enumerate(te):
        organ = M[i]["organ"]; gm = Y[i] > 0
        if int(gm.sum()) < 1:
            continue
        box = bbox_from_mask(gm.astype(np.uint8), pad=3)
        H, W = gm.shape
        if box is None:
            box = [0, 0, W - 1, H - 1]
        pred = np.zeros_like(gm)
        for ckpt, text in CKPTS[organ]:      # union over the organ's model(s)
            pred |= predict(cache[ckpt], X[i], box, text)
        s = int(pred.sum()) + int(gm.sum())
        d = 2 * int((pred & gm).sum()) / s if s else 1.0
        dd[(organ, M[i]["dataset"])].append(d); byo[organ].append(d)
        if (n + 1) % 1500 == 0:
            print(f"  {n+1}/{len(te)}", flush=True)
    per_ds = {f"{o}/{ds}": round(float(np.mean(v)), 4) for (o, ds), v in sorted(dd.items())}
    per_o = {o: round(float(np.mean(v)), 4) for o, v in byo.items()}
    json.dump({"note": "reused FLARE-Task2 box-fine-tuned per-organ models + GT box on the pooled test",
               "per_organ": per_o, "per_organ_dataset": per_ds},
              open("/home/ud3d4/Desktop/SWOG/results/organ_ausam_reuse.json", "w"), indent=2)
    print("REUSE-AUSAM per-organ:", per_o, flush=True)
    print("per organ/dataset:", per_ds, flush=True)
    print("-> results/organ_ausam_reuse.json", flush=True)


if __name__ == "__main__":
    main()
