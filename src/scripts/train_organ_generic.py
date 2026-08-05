#!/usr/bin/env python3
"""Train ONE generic organ model — liver / kidney / pancreas — via SAM3 concept prompts (per-slice
organ name, no box). Same recipe as the tumor model (partial-freeze, Dice+Focal, disc-LR, cosine, DDP
2 GPUs), strictly patient-level split (val AND test held out by whole patient, per dataset). Evaluates
each organ on its held-out patients, per source dataset. Data: organ_pool_lkp (l/k/p from LiTS + KiTS +
MSD + FLARE-Task2 + FLARE23).
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
from run_pancreas_sam3 import (train_worker_v3, WORLD_SIZE, find_free_port,
                               _load_sam3_ckpt, SAM3_MODEL_ID, HF_TOKEN, extract_best_mask_soft)

POOL = "/scratch/ud3d4/acm_data/organ_pool_lkp"
KG = "--kg" in sys.argv                                   # KG-in-training plausibility loss on/off (ablation)
EVAL_ONLY = "--eval-only" in sys.argv                     # skip training, just eval an existing checkpoint
AUSAM = "--ausam" in sys.argv                             # GT-box (semi-oracle): the supervised segmenter
DATASET = sys.argv[sys.argv.index("--dataset") + 1] if "--dataset" in sys.argv else None  # per-dataset AUSAM
TAG = ("_ausam" if AUSAM else ("_kg" if KG else "")) + (f"_{DATASET}" if DATASET else "")
CKPT = f"{POOL}/sam3_organ_generic{TAG}.pth"
OUT = f"/home/ud3d4/Desktop/SWOG/results/organ_generic{TAG}.json"
EPOCHS, PATIENCE = (12, 4) if AUSAM else (6, 2)            # AUSAM: train to convergence for the best (semi-oracle) result


def patient_split(meta):
    """val + test held out by whole patient (dataset, case), per dataset (seed 42)."""
    rng = np.random.RandomState(42)
    by = defaultdict(set)
    for m in meta:
        by[m["dataset"]].add(m["case"])
    te, va = set(), set()
    for d, cs in by.items():
        cs = sorted(cs); rng.shuffle(cs)
        n = len(cs); nte = max(1, int(0.2 * n)); nva = max(1, int(0.1 * n))
        te |= {(d, c) for c in cs[:nte]}; va |= {(d, c) for c in cs[nte:nte + nva]}
    tr_i, va_i, te_i = [], [], []
    for i, m in enumerate(meta):
        k = (m["dataset"], m["case"])
        (te_i if k in te else va_i if k in va else tr_i).append(i)
    return tr_i, va_i, te_i


def eval_per_organ(X, Y, meta, test_idx):
    from transformers import Sam3Processor
    from run_pancreas_sam3 import bbox_from_mask
    dev = "cuda:0"
    proc = Sam3Processor.from_pretrained(SAM3_MODEL_ID, token=HF_TOKEN)
    model = _load_sam3_ckpt(CKPT, dev)
    dd = defaultdict(list)
    for i in test_idx:
        organ = meta[i]["organ"]; gm = Y[i] > 0
        pk = {"images": [X[i]], "text": [organ], "return_tensors": "pt"}
        if AUSAM:                                          # semi-oracle: feed the GT-derived box
            box = bbox_from_mask(gm.astype(np.uint8), pad=3)
            H, W = gm.shape
            pk["input_boxes"] = [[box if box is not None else [0, 0, W - 1, H - 1]]]
            pk["input_boxes_labels"] = [[1]]
        inp = proc(**pk)
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
        dd[(organ, meta[i]["dataset"])].append(2 * int((pred & gm).sum()) / s if s else 1.0)
    res = {f"{o}/{d}": round(float(np.mean(v)), 4) for (o, d), v in sorted(dd.items())}
    by_organ = defaultdict(list)
    for i in test_idx:
        by_organ[meta[i]["organ"]]  # ensure keys
    for (o, d), v in dd.items():
        by_organ[o] += v
    res_organ = {o: round(float(np.mean(v)), 4) for o, v in by_organ.items() if v}
    return res, res_organ


def main():
    X = np.load(f"{POOL}/images.npy", mmap_mode="r"); Y = np.load(f"{POOL}/masks.npy", mmap_mode="r")
    M = json.load(open(f"{POOL}/meta.json"))
    if DATASET:                                            # per-dataset AUSAM: keep only this source's slices
        keep = [i for i, m in enumerate(M) if m["dataset"] == DATASET]
        assert keep, f"no slices for --dataset {DATASET}; have {sorted(set(m['dataset'] for m in M))}"
        X = np.ascontiguousarray(X[keep]); Y = np.ascontiguousarray(Y[keep]); M = [M[i] for i in keep]
        print(f"--dataset {DATASET}: {len(M)} slices, organs {dict(Counter(m['organ'] for m in M))}", flush=True)
    print("pool:", dict(Counter(m["organ"] for m in M)),
          "| datasets:", dict(Counter(m["dataset"] for m in M)), flush=True)
    tr, va, te = patient_split(M)
    print(f"patient-level split tr/va/te = {len(tr)}/{len(va)}/{len(te)}", flush=True)
    print("TEST per organ:", dict(Counter(M[i]["organ"] for i in te)), flush=True)

    texts_tr = [M[i]["organ"] for i in tr]
    texts_va = [M[i]["organ"] for i in va]
    cfg = {"model_save_path": CKPT,
           "pretrained_path": CKPT if (AUSAM and os.path.exists(CKPT)) else None,  # AUSAM: resume best-so-far (preempt-safe)
           "epochs": EPOCHS, "batch_size": 4,
           "patience": PATIENCE, "strong_augment": True,
           "use_boxes": AUSAM,                                              # AUSAM -> GT box (semi-oracle); else concept only
           "freeze_blocks": 20, "encoder_lr": 1e-5, "decoder_lr": 1e-4, "warmup_epochs": 2,
           "cosine_T0": 40, "dice_weight": 0.7, "focal_weight": 0.3,
           "texts_tr": texts_tr, "texts_va": texts_va}                          # per-slice organ prompts
    if AUSAM:
        print(f"AUSAM (GT-box, supervised) organ training — use_boxes=True, epochs={EPOCHS}"
              f"{', resuming from ' + CKPT if cfg['pretrained_path'] else ''}", flush=True)
    if KG and not AUSAM:
        priors = json.load(open(f"{POOL}/organ_train_priors.json"))
        cfg["kg_priors"] = priors; cfg["kg_weight"] = 0.1; cfg["kg_centroid_w"] = 0.5
        print("KG-in-training ON — priors:", {o: (p["cy"], p["cx"], p["area_lo"], p["area_hi"])
                                              for o, p in priors.items()}, flush=True)
    if not EVAL_ONLY:
        port = find_free_port()
        mp.spawn(train_worker_v3, args=(WORLD_SIZE, port, cfg, X[tr], Y[tr], X[va], Y[va]),
                 nprocs=WORLD_SIZE, join=True)
    else:
        print(f"--eval-only: scoring existing {CKPT}", flush=True)

    per_dataset, per_organ = eval_per_organ(X, Y, M, te)
    print("\n=== organ model — held-out patient-level test Dice ===", flush=True)
    print("per organ:", per_organ, flush=True)
    print("per organ/dataset:", per_dataset, flush=True)
    json.dump({"per_organ": per_organ, "per_organ_dataset": per_dataset}, open(OUT, "w"), indent=2)
    print("-> " + OUT, flush=True)


if __name__ == "__main__":
    main()
