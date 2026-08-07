#!/usr/bin/env python3
"""Cross-dataset organ generalization matrix for the per-dataset AUSAM models.

Each per-dataset AUSAM model is applied to EVERY dataset's held-out test slices (same seed-42 split, GT box +
organ text, slice-level Dice). The diagonal = in-distribution (matches results/organ_generic_ausam_*.json);
the off-diagonal = cross-dataset (train on A, test on B) — the leave-one-dataset-out generalization the
assessment highlights. -> results/crossdataset_organ.json

  python eval_crossdataset_organ.py
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

DATASETS = ["lits", "kits", "msd", "flare_task2", "flare23"]
MODEL_ORGANS = {"lits": ["liver"], "kits": ["kidney"], "msd": ["pancreas"],
                "flare_task2": ["liver", "kidney", "pancreas"], "flare23": ["liver", "kidney", "pancreas"]}
DEV = "cuda:0"


def main():
    from transformers import Sam3Processor
    proc = Sam3Processor.from_pretrained(SAM3_MODEL_ID, token=HF_TOKEN)
    X = np.load(f"{POOL}/images.npy", mmap_mode="r"); Y = np.load(f"{POOL}/masks.npy", mmap_mode="r")
    M = json.load(open(f"{POOL}/meta.json"))
    _, _, te = patient_split(M)
    # test indices grouped by (dataset, organ)
    idx_by = defaultdict(list)
    for i in te:
        idx_by[(M[i]["dataset"], M[i]["organ"])].append(i)

    def predict(model, img, box, organ):
        inp = proc(images=[img], text=[organ], input_boxes=[[box]], input_boxes_labels=[[1]], return_tensors="pt")
        kw = {"pixel_values": inp["pixel_values"].to(DEV)}
        for k in ("input_ids", "attention_mask", "input_boxes", "input_boxes_labels"):
            if inp.get(k) is not None:
                kw[k] = inp[k].to(DEV)
        with torch.no_grad(), autocast("cuda"):
            out = model(**kw)
            pm = extract_best_mask_soft(out.pred_masks, out.pred_logits)
            pm = torch.nn.functional.interpolate(pm.float(), size=(256, 256), mode="bilinear", align_corners=False)
        return pm.sigmoid().squeeze().cpu().numpy() > 0.5

    matrix = {}                      # organ -> {model_ds -> {test_ds -> dice}}
    for organ in ["liver", "kidney", "pancreas"]:
        matrix[organ] = {}
    for model_ds in DATASETS:
        model = _load_sam3_ckpt(f"{POOL}/sam3_organ_generic_ausam_{model_ds}.pth", DEV)
        for organ in MODEL_ORGANS[model_ds]:
            matrix[organ][model_ds] = {}
            for test_ds in DATASETS:
                idx = idx_by.get((test_ds, organ), [])
                if not idx:
                    continue
                ds = []
                for i in idx:
                    gm = np.asarray(Y[i]) > 0
                    if int(gm.sum()) < 1:
                        continue
                    box = bbox_from_mask(gm.astype(np.uint8), pad=3)
                    H, W = gm.shape
                    pred = predict(model, np.asarray(X[i]), box if box is not None else [0, 0, W - 1, H - 1], organ)
                    s = int(pred.sum()) + int(gm.sum())
                    ds.append(2 * int((pred & gm).sum()) / s if s else 1.0)
                d = round(float(np.mean(ds)), 4)
                matrix[organ][model_ds][test_ds] = d
                tag = "in-dist" if model_ds == test_ds else "CROSS"
                print(f"  {organ:9s} model={model_ds:12s} -> test={test_ds:12s}: {d}  [{tag}] (n={len(ds)})", flush=True)
        del model; torch.cuda.empty_cache()

    json.dump({"note": "per-dataset AUSAM model x test-dataset, slice-level Dice, GT box; diagonal=in-dist",
               "matrix": matrix}, open("/home/ud3d4/Desktop/SWOG/results/crossdataset_organ.json", "w"), indent=2)
    print("-> results/crossdataset_organ.json", flush=True)


if __name__ == "__main__":
    main()
