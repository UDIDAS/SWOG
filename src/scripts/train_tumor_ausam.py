#!/usr/bin/env python3
"""Per-dataset AUSAM TUMOR model (GT-box, semi-oracle) — the tumor counterpart of train_organ_generic --ausam.
Faithful to how the original AUSAM prompted tumors (a GT-mask-derived prompt), on the SAM3 backbone with a
single GT box + text='tumor'. One model per dataset (lits liver-tumor / pancreas MSD07 / kits KiTS23 /
flare FLARE23), so it's the supervised AUSAM control the KG-unsupervised approach will later be compared to.

  python train_tumor_ausam.py --dataset lits|pancreas|kits|flare
Strictly patient-level split (seed 42). SAM3 partial-freeze, Dice+Focal, disc-LR, cosine, DDP 2 GPUs.
-> tumor_pool/sam3_tumor_ausam_<ds>.pth ; results/tumor_ausam_<ds>.json
"""
import json
import os
import sys
from collections import defaultdict, Counter

import numpy as np
import torch
from torch.amp import autocast
import torch.multiprocessing as mp

sys.path.insert(0, "/home/ud3d4/Desktop/SWOG/src/scripts")
from train_tumor_incremental import load_all, patient_split, CKPT_DIR
from run_pancreas_sam3 import (train_worker_v3, WORLD_SIZE, find_free_port, bbox_from_mask,
                               _load_sam3_ckpt, SAM3_MODEL_ID, HF_TOKEN, extract_best_mask_soft)

DATASET = sys.argv[sys.argv.index("--dataset") + 1] if "--dataset" in sys.argv else None
assert DATASET in {"lits", "pancreas", "kits", "flare"}, "need --dataset lits|pancreas|kits|flare"
EVAL_ONLY = "--eval-only" in sys.argv
CKPT = f"{CKPT_DIR}/sam3_tumor_ausam_{DATASET}.pth"
OUT = f"/home/ud3d4/Desktop/SWOG/results/tumor_ausam_{DATASET}.json"
EPOCHS, PATIENCE = 12, 4


def eval_ausam(X, Y, meta, test_idx):
    from transformers import Sam3Processor
    dev = "cuda:0"
    proc = Sam3Processor.from_pretrained(SAM3_MODEL_ID, token=HF_TOKEN)
    model = _load_sam3_ckpt(CKPT, dev)
    ds = []
    for i in test_idx:
        gm = Y[i] > 0
        if int(gm.sum()) < 1:
            continue
        box = bbox_from_mask(gm.astype(np.uint8), pad=3)
        H, W = gm.shape
        if box is None:
            box = [0, 0, W - 1, H - 1]
        inp = proc(images=[X[i]], text=["tumor"], input_boxes=[[box]], input_boxes_labels=[[1]],
                   return_tensors="pt")
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
        ds.append(2 * int((pred & gm).sum()) / s if s else 1.0)
    return round(float(np.mean(ds)), 4), len(ds)


def main():
    X, Y, M = load_all()
    keep = [i for i, m in enumerate(M) if m["dataset"] == DATASET]
    assert keep, f"no {DATASET} tumor slices"
    X = np.ascontiguousarray(X[keep]); Y = np.ascontiguousarray(Y[keep]); M = [M[i] for i in keep]
    tr, va, te = patient_split(M)
    print(f"tumor AUSAM [{DATASET}] {len(M)} slices, "
          f"{len({m['case'] for m in M})} patients | split tr/va/te = {len(tr)}/{len(va)}/{len(te)}", flush=True)

    cfg = {"model_save_path": CKPT,
           "pretrained_path": CKPT if os.path.exists(CKPT) else None,   # resume best-so-far (preempt-safe)
           "epochs": EPOCHS, "batch_size": 4, "patience": PATIENCE, "strong_augment": True,
           "text_prompt": "tumor", "use_boxes": True,                   # AUSAM: GT box (semi-oracle)
           "freeze_blocks": 20, "encoder_lr": 1e-5, "decoder_lr": 1e-4, "warmup_epochs": 2,
           "cosine_T0": 40, "dice_weight": 0.7, "focal_weight": 0.3}
    if not EVAL_ONLY:
        print(f"tumor AUSAM (GT-box) training — use_boxes=True, text='tumor', epochs={EPOCHS}"
              f"{', resuming' if cfg['pretrained_path'] else ''}", flush=True)
        port = find_free_port()
        mp.spawn(train_worker_v3, args=(WORLD_SIZE, port, cfg, X[tr], Y[tr], X[va], Y[va]),
                 nprocs=WORLD_SIZE, join=True)

    dice, n = eval_ausam(X, Y, M, te)
    print(f"\n=== tumor AUSAM [{DATASET}] held-out patient-level test Dice = {dice} (n={n}) ===", flush=True)
    json.dump({"dataset": DATASET, "tumor_ausam_dice": dice, "n_test_slices": n}, open(OUT, "w"), indent=2)
    print("-> " + OUT, flush=True)


if __name__ == "__main__":
    main()
