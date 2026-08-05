#!/usr/bin/env python3
"""AUSAM-based (GT-box, semi-oracle) ORGAN eval — the AUSAM ceiling arm for the organ 3-way
(baseline vs KG vs AUSAM). For each held-out organ test slice, feed a **GT-derived box** (+ the organ text)
and score per organ / dataset — the same pool + patient split as train_organ_generic, so numbers line up
directly with organ_generic.json (baseline) and organ_generic_kg.json (KG).

--ckpt base  -> BASE SAM3 (the literal AUSAM-on-SAM3 reference; needs no training).
--ckpt PATH  -> score a box-fine-tuned organ model instead.
Needs one GPU. Writes results/organ_ausam_<tag>.json.
"""
import argparse
import json
import sys
from collections import defaultdict

import numpy as np
import torch
from torch.amp import autocast

sys.path.insert(0, "/home/ud3d4/Desktop/SWOG/src/scripts")
from train_organ_generic import patient_split, POOL
from run_pancreas_sam3 import bbox_from_mask, extract_best_mask_soft, _load_sam3_ckpt, SAM3_MODEL_ID, HF_TOKEN
import infer_ensemble as IE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="base"); ap.add_argument("--tag", default="quick")
    a = ap.parse_args()
    X = np.load(f"{POOL}/images.npy"); Y = np.load(f"{POOL}/masks.npy"); M = json.load(open(f"{POOL}/meta.json"))
    tr, va, te = patient_split(M)
    print(f"AUSAM organ eval [{a.tag}, ckpt={a.ckpt}] over {len(te)} held-out test slices", flush=True)
    from transformers import Sam3Processor
    proc = Sam3Processor.from_pretrained(SAM3_MODEL_ID, token=HF_TOKEN)
    dev = "cuda:0"
    model = IE.load_base()[0] if a.ckpt == "base" else _load_sam3_ckpt(a.ckpt, dev)

    dd = defaultdict(list); byo = defaultdict(list)
    for n, i in enumerate(te):
        organ = M[i]["organ"]; gm = Y[i] > 0
        if int(gm.sum()) < 1:
            continue
        box = bbox_from_mask(gm.astype(np.uint8), pad=3)
        H, W = gm.shape
        if box is None:
            box = [0, 0, W - 1, H - 1]
        inp = proc(images=[X[i]], text=[organ], input_boxes=[[box]], input_boxes_labels=[[1]], return_tensors="pt")
        kw = {"pixel_values": inp["pixel_values"].to(dev)}
        for k in ("input_ids", "attention_mask", "input_boxes", "input_boxes_labels"):
            if inp.get(k) is not None:
                kw[k] = inp[k].to(dev)
        with torch.no_grad(), autocast("cuda"):
            out = model(**kw)
            pm = extract_best_mask_soft(out.pred_masks, out.pred_logits)
            pm = torch.nn.functional.interpolate(pm.float(), size=(256, 256), mode="bilinear", align_corners=False)
        pred = pm.sigmoid().squeeze().cpu().numpy() > 0.5
        s = int(pred.sum()) + int(gm.sum())
        d = 2 * int((pred & gm).sum()) / s if s else 1.0
        dd[(organ, M[i]["dataset"])].append(d); byo[organ].append(d)
        if (n + 1) % 1500 == 0:
            print(f"  {n+1}/{len(te)}", flush=True)
    per_ds = {f"{o}/{ds}": round(float(np.mean(v)), 4) for (o, ds), v in sorted(dd.items())}
    per_o = {o: round(float(np.mean(v)), 4) for o, v in byo.items()}
    json.dump({"tag": a.tag, "ckpt": a.ckpt, "per_organ": per_o, "per_organ_dataset": per_ds},
              open(f"/home/ud3d4/Desktop/SWOG/results/organ_ausam_{a.tag}.json", "w"), indent=2)
    print("AUSAM (GT-box) organ per-organ Dice:", per_o, flush=True)
    print(f"-> results/organ_ausam_{a.tag}.json", flush=True)


if __name__ == "__main__":
    main()
