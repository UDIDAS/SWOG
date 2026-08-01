#!/usr/bin/env python3
"""Train ONE generic text-'tumor' SAM3 model on pooled LiTS+Pancreas tumor slices, on BOTH GPUs.
Prompt = text 'tumor', NO box (use_boxes=False) -> autonomous at inference. Then evaluate
autonomously (text 'tumor', no box) per dataset. Split: pancreas held out patient-level, LiTS
slice-level (no patient ids in the pre-sliced npy)."""
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.amp import autocast

sys.path.insert(0, "/home/ud3d4/Desktop/SWOG/src/scripts")
from run_pancreas_sam3 import (train_worker_v3, WORLD_SIZE, find_free_port,   # noqa: E402
                               _load_sam3_ckpt, extract_best_mask_soft, SAM3_MODEL_ID, HF_TOKEN)

POOL = "/scratch/ud3d4/acm_data/tumor_pool"
CKPT = f"{POOL}/sam3_tumor_generic.pth"


def split():
    X = np.load(f"{POOL}/images.npy")
    Y = np.load(f"{POOL}/masks.npy")
    meta = json.load(open(f"{POOL}/meta.json"))
    rng = np.random.RandomState(42)
    panc = sorted({m["case"] for m in meta if m["dataset"] == "pancreas"})
    rng.shuffle(panc)
    te_panc = set(panc[:int(0.2 * len(panc))])
    lits = [i for i, m in enumerate(meta) if m["dataset"] == "lits"]
    rng.shuffle(lits)
    te_lits = set(lits[:int(0.2 * len(lits))])
    tr, va, te = [], [], []
    for i, m in enumerate(meta):
        if (m["dataset"] == "pancreas" and m["case"] in te_panc) or \
           (m["dataset"] == "lits" and i in te_lits):
            te.append(i)
        elif rng.rand() < 0.1:
            va.append(i)
        else:
            tr.append(i)
    print(f"split  train={len(tr)} val={len(va)} test={len(te)}", flush=True)
    return X, Y, meta, tr, va, te


def autonomous_eval(X, Y, meta, te):
    from transformers import Sam3Processor
    dev = "cuda:0"
    proc = Sam3Processor.from_pretrained(SAM3_MODEL_ID, token=HF_TOKEN)
    model = _load_sam3_ckpt(CKPT, dev)
    dd = defaultdict(list)
    for i in te:
        inputs = proc(images=[X[i]], text=["tumor"], return_tensors="pt")
        kw = {"pixel_values": inputs["pixel_values"].to(dev)}
        for k in ("input_ids", "attention_mask"):
            if inputs.get(k) is not None:
                kw[k] = inputs[k].to(dev)
        with torch.no_grad(), autocast("cuda"):
            out = model(**kw)
            pm = extract_best_mask_soft(out.pred_masks, out.pred_logits)
            pm = torch.nn.functional.interpolate(pm.float(), size=(256, 256), mode="bilinear",
                                                 align_corners=False)
        pred = pm.sigmoid().squeeze().cpu().numpy() > 0.5
        gm = Y[i] > 0
        s = int(pred.sum()) + int(gm.sum())
        dd[meta[i]["dataset"]].append(2 * int((pred & gm).sum()) / s if s else 1.0)
    print("\n===== autonomous text-'tumor' Dice (no box) =====", flush=True)
    for ds, v in dd.items():
        print(f"  {ds:10s} Dice={np.mean(v):.4f} +/- {np.std(v):.4f}  (n={len(v)})", flush=True)


def main():
    X, Y, meta, tr, va, te = split()
    cfg = {"model_save_path": CKPT, "epochs": 40, "batch_size": 4, "patience": 8,
           "strong_augment": True, "text_prompt": "tumor", "use_boxes": False,
           "freeze_blocks": 20, "encoder_lr": 1e-5, "decoder_lr": 1e-4,
           "warmup_epochs": 3, "cosine_T0": 30, "dice_weight": 0.7, "focal_weight": 0.3}
    print(f"training generic tumor model on {WORLD_SIZE} GPUs (text='tumor', use_boxes=False)", flush=True)
    port = find_free_port()
    mp.spawn(train_worker_v3, args=(WORLD_SIZE, port, cfg, X[tr], Y[tr], X[va], Y[va]),
             nprocs=WORLD_SIZE, join=True)
    if os.path.exists(CKPT):
        autonomous_eval(X, Y, meta, te)
    else:
        print("[warn] no checkpoint saved — training may not have improved", flush=True)


if __name__ == "__main__":
    main()
