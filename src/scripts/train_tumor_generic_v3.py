#!/usr/bin/env python3
"""Generic tumor model v3 — abdominal CT only, FLARE-organ scope: LiTS + Pancreas + FLARE + KiTS.
Warm-start from v2 (0.9145), text='tumor', use_boxes=False, both GPUs. Per-dataset autonomous eval."""
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.amp import autocast

sys.path.insert(0, "/home/ud3d4/Desktop/SWOG/src/scripts")
from run_pancreas_sam3 import (train_worker_v3, WORLD_SIZE, find_free_port,
                               _load_sam3_ckpt, extract_best_mask_soft, SAM3_MODEL_ID, HF_TOKEN)

POOLS = ["/scratch/ud3d4/acm_data/tumor_pool",          # LiTS + Pancreas
         "/scratch/ud3d4/acm_data/flare_tumor_pool",     # FLARE
         "/scratch/ud3d4/acm_data/kits_tumor_pool"]      # KiTS (kidney)
WARM = "/scratch/ud3d4/acm_data/tumor_pool/sam3_tumor_generic_v2.pth"
CKPT = "/scratch/ud3d4/acm_data/tumor_pool/sam3_tumor_generic_v3.pth"
CASE_LEVEL = {"pancreas", "flare", "kits"}               # patient-level holdout; lits is slice-pooled


def load_all():
    Xs, Ys, M = [], [], []
    for p in POOLS:
        Xs.append(np.load(f"{p}/images.npy"))
        Ys.append(np.load(f"{p}/masks.npy"))
        M += json.load(open(f"{p}/meta.json"))
    return np.concatenate(Xs), np.concatenate(Ys), M


def split(meta):
    rng = np.random.RandomState(42)
    cases = defaultdict(list)
    for i, m in enumerate(meta):
        cases[(m["dataset"], m["case"])].append(i)
    te_cases = set()
    for ds in CASE_LEVEL:
        cs = sorted({c for (d, c) in cases if d == ds})
        rng.shuffle(cs)
        te_cases |= {(ds, c) for c in cs[:int(0.2 * len(cs))]}
    lits = [i for i, m in enumerate(meta) if m["dataset"] == "lits"]
    rng.shuffle(lits)
    te_lits = set(lits[:int(0.2 * len(lits))])
    tr, va, te = [], [], []
    for i, m in enumerate(meta):
        if (m["dataset"], m["case"]) in te_cases or (m["dataset"] == "lits" and i in te_lits):
            te.append(i)
        elif rng.rand() < 0.1:
            va.append(i)
        else:
            tr.append(i)
    return tr, va, te


def autonomous_eval(X, Y, meta, te):
    from transformers import Sam3Processor
    dev = "cuda:0"
    proc = Sam3Processor.from_pretrained(SAM3_MODEL_ID, token=HF_TOKEN)
    model = _load_sam3_ckpt(CKPT, dev)
    dd = defaultdict(list)
    for i in te:
        inp = proc(images=[X[i]], text=["tumor"], return_tensors="pt")
        kw = {"pixel_values": inp["pixel_values"].to(dev)}
        for k in ("input_ids", "attention_mask"):
            if inp.get(k) is not None:
                kw[k] = inp[k].to(dev)
        with torch.no_grad(), autocast("cuda"):
            out = model(**kw)
            pm = extract_best_mask_soft(out.pred_masks, out.pred_logits)
            pm = torch.nn.functional.interpolate(pm.float(), size=(256, 256), mode="bilinear",
                                                 align_corners=False)
        pred = pm.sigmoid().squeeze().cpu().numpy() > 0.5
        gm = Y[i] > 0
        s = int(pred.sum()) + int(gm.sum())
        dd[meta[i]["dataset"]].append(2 * int((pred & gm).sum()) / s if s else 1.0)
    print("\n===== v3 autonomous text-'tumor' Dice per dataset =====", flush=True)
    for ds, v in sorted(dd.items()):
        print(f"  {ds:10s} Dice={np.mean(v):.4f} +/- {np.std(v):.4f}  (n={len(v)})", flush=True)


def main():
    X, Y, M = load_all()
    tr, va, te = split(M)
    print("pool:", dict(Counter(m["dataset"] for m in M)),
          "| split tr/va/te", len(tr), len(va), len(te), flush=True)
    warm = WARM if os.path.exists(WARM) else "/scratch/ud3d4/acm_data/tumor_pool/sam3_tumor_generic.pth"
    cfg = {"model_save_path": CKPT, "pretrained_path": warm, "epochs": 25, "batch_size": 4,
           "patience": 6, "strong_augment": True, "text_prompt": "tumor", "use_boxes": False,
           "freeze_blocks": 20, "encoder_lr": 1e-5, "decoder_lr": 1e-4, "warmup_epochs": 2,
           "cosine_T0": 40, "dice_weight": 0.7, "focal_weight": 0.3}
    port = find_free_port()
    mp.spawn(train_worker_v3, args=(WORLD_SIZE, port, cfg, X[tr], Y[tr], X[va], Y[va]),
             nprocs=WORLD_SIZE, join=True)
    if os.path.exists(CKPT):
        autonomous_eval(X, Y, M, te)


if __name__ == "__main__":
    main()
