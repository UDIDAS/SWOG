#!/usr/bin/env python3
"""Incremental, STRICTLY patient-level tumor training — gives three views at once:
  (1) coverage-growth progression  (Dice rises as datasets are added),
  (2) per-dataset held-out test Dice (final model), and
  (3) cross-dataset generalization (a dataset NOT yet trained is scored each stage).

Stages, each warm-started from the previous:
  v1 = LiTS + Pancreas   →   v2 = +FLARE   →   v3 = +KiTS
Every dataset's val AND test are held out by WHOLE PATIENT (seed 42). No slice-level data anywhere.
Each dataset's test set is FIXED across stages, so the progression is a fair comparison.
Output: results/tumor_incremental.json  (+ per-stage checkpoints).
"""
import json
import os
import sys
from collections import defaultdict, Counter

import numpy as np
import torch
from torch.amp import autocast

sys.path.insert(0, "/home/ud3d4/Desktop/SWOG/src/scripts")
from train_tumor_generic_v3 import extract_best_mask_soft
from run_pancreas_sam3 import (train_worker_v3, WORLD_SIZE, find_free_port,
                               _load_sam3_ckpt, SAM3_MODEL_ID, HF_TOKEN)
import torch.multiprocessing as mp

POOLS = {"tumor_pool": "/scratch/ud3d4/acm_data/tumor_pool",           # LiTS + Pancreas
         "flare_tumor_pool": "/scratch/ud3d4/acm_data/flare_tumor_pool",
         "kits_tumor_pool": "/scratch/ud3d4/acm_data/kits_tumor_pool"}
CKPT_DIR = "/scratch/ud3d4/acm_data/tumor_pool"
OUT = "/home/ud3d4/Desktop/SWOG/results/tumor_incremental.json"
# 4 stages: FLARE is HELD OUT (never trained) for s1-s3 -> its test set is the cross-dataset generalization
# probe; added at s4 for the full deployment model. FLARE's TEST patients stay held out throughout.
STAGES = [("s1_lits", {"lits"}),
          ("s2_pancreas", {"lits", "pancreas"}),
          ("s3_kits", {"lits", "pancreas", "kits"}),          # <- headline cross-dataset number vs FLARE
          ("s4_flare", {"lits", "pancreas", "kits", "flare"})]  # <- full deployment model
EPOCHS, PATIENCE = 14, 5


def load_all():
    Xs, Ys, M = [], [], []
    for p in POOLS.values():
        Xs.append(np.load(f"{p}/images.npy")); Ys.append(np.load(f"{p}/masks.npy"))
        M += json.load(open(f"{p}/meta.json"))
    return np.concatenate(Xs), np.concatenate(Ys), M


def patient_split(meta):
    """val + test held out by WHOLE PATIENT, per dataset (seed 42)."""
    rng = np.random.RandomState(42)
    by = defaultdict(set)
    for m in meta:
        by[m["dataset"]].add(m["case"])
    te_cases, va_cases = set(), set()
    for ds, cs in by.items():
        cs = sorted(cs); rng.shuffle(cs)
        n = len(cs); nte = max(1, int(0.2 * n)); nva = max(1, int(0.1 * n))
        te_cases |= {(ds, c) for c in cs[:nte]}
        va_cases |= {(ds, c) for c in cs[nte:nte + nva]}
    tr, va, te = [], [], []
    for i, m in enumerate(meta):
        k = (m["dataset"], m["case"])
        (te if k in te_cases else va if k in va_cases else tr).append(i)
    return tr, va, te


def eval_per_dataset(ckpt, X, Y, meta, test_idx):
    from transformers import Sam3Processor
    dev = "cuda:0"
    proc = Sam3Processor.from_pretrained(SAM3_MODEL_ID, token=HF_TOKEN)
    model = _load_sam3_ckpt(ckpt, dev)
    dd = defaultdict(list)
    for i in test_idx:
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
    del model; torch.cuda.empty_cache()
    return {ds: round(float(np.mean(v)), 4) for ds, v in dd.items()}


def main():
    X, Y, M = load_all()
    tr, va, te = patient_split(M)
    print("pool:", dict(Counter(m["dataset"] for m in M)), flush=True)
    print("patients:", {ds: len({m['case'] for m in M if m['dataset'] == ds}) for ds in
                        {m['dataset'] for m in M}}, flush=True)
    print(f"global split (patient-level) tr/va/te = {len(tr)}/{len(va)}/{len(te)}", flush=True)

    results = {"stages": [], "epochs": EPOCHS}
    warm = None
    for name, dss in STAGES:
        tr_s = [i for i in tr if M[i]["dataset"] in dss]
        va_s = [i for i in va if M[i]["dataset"] in dss]
        ckpt = f"{CKPT_DIR}/sam3_tumor_{name}_patientlevel.pth"
        print(f"\n===== STAGE {name}: train on {sorted(dss)} | tr/va = {len(tr_s)}/{len(va_s)} =====", flush=True)
        if not os.path.exists(ckpt):
            cfg = {"model_save_path": ckpt, "pretrained_path": warm, "epochs": EPOCHS, "batch_size": 4,
                   "patience": PATIENCE, "strong_augment": True, "text_prompt": "tumor", "use_boxes": False,
                   "freeze_blocks": 20, "encoder_lr": 1e-5, "decoder_lr": 1e-4, "warmup_epochs": 2,
                   "cosine_T0": 40, "dice_weight": 0.7, "focal_weight": 0.3}
            port = find_free_port()
            mp.spawn(train_worker_v3, args=(WORLD_SIZE, port, cfg, X[tr_s], Y[tr_s], X[va_s], Y[va_s]),
                     nprocs=WORLD_SIZE, join=True)
        # eval on EVERY dataset's fixed patient-level test set (trained = in-dist, untrained = cross-dataset)
        per = eval_per_dataset(ckpt, X, Y, M, te)
        indist = {d: v for d, v in per.items() if d in dss}
        cross = {d: v for d, v in per.items() if d not in dss}     # held-out datasets = cross-dataset probe
        overall = round(float(np.mean(list(indist.values()))), 4)
        print(f"  {name} IN-DIST: {indist}  overall={overall}", flush=True)
        print(f"  {name} CROSS-DATASET (held-out): {cross}", flush=True)
        results["stages"].append({"name": name, "trained_on": sorted(dss), "per_dataset_test": per,
                                  "in_distribution": indist, "cross_dataset_heldout": cross,
                                  "overall_indist": overall})
        json.dump(results, open(OUT, "w"), indent=2)
        warm = ckpt
    # canonicalize the final (all-4-datasets) stage as the deployment model
    import shutil
    final_ckpt = f"{CKPT_DIR}/sam3_tumor_{STAGES[-1][0]}_patientlevel.pth"
    shutil.copy(final_ckpt, f"{CKPT_DIR}/sam3_tumor_generic.pth")
    print(f"\ncanonicalized {STAGES[-1][0]} (all 4 datasets, patient-level) -> sam3_tumor_generic.pth", flush=True)
    print("-> " + OUT, flush=True)


if __name__ == "__main__":
    main()
