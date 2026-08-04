#!/usr/bin/env python3
"""Train ONE generic organ model — liver / kidney / pancreas — via SAM3 concept prompts (per-slice
organ name, no box). Same recipe as the tumor model (partial-freeze, Dice+Focal, disc-LR, cosine, DDP
2 GPUs), strictly patient-level split (val AND test held out by whole patient, per dataset). Evaluates
each organ on its held-out patients, per source dataset. Data: organ_pool_lkp (l/k/p from LiTS + KiTS +
MSD + FLARE-Task2 + FLARE23).
"""
import json
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
CKPT = f"{POOL}/sam3_organ_generic.pth"
OUT = "/home/ud3d4/Desktop/SWOG/results/organ_generic.json"
EPOCHS, PATIENCE = 16, 5


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
    dev = "cuda:0"
    proc = Sam3Processor.from_pretrained(SAM3_MODEL_ID, token=HF_TOKEN)
    model = _load_sam3_ckpt(CKPT, dev)
    dd = defaultdict(list)
    for i in test_idx:
        organ = meta[i]["organ"]
        inp = proc(images=[X[i]], text=[organ], return_tensors="pt")
        kw = {"pixel_values": inp["pixel_values"].to(dev)}
        for k in ("input_ids", "attention_mask"):
            if inp.get(k) is not None:
                kw[k] = inp[k].to(dev)
        with torch.no_grad(), autocast("cuda"):
            out = model(**kw)
            pm = extract_best_mask_soft(out.pred_masks, out.pred_logits)
            pm = torch.nn.functional.interpolate(pm.float(), size=(256, 256), mode="bilinear", align_corners=False)
        pred = pm.sigmoid().squeeze().cpu().numpy() > 0.5
        gm = Y[i] > 0
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
    X = np.load(f"{POOL}/images.npy"); Y = np.load(f"{POOL}/masks.npy"); M = json.load(open(f"{POOL}/meta.json"))
    print("pool:", dict(Counter(m["organ"] for m in M)),
          "| datasets:", dict(Counter(m["dataset"] for m in M)), flush=True)
    tr, va, te = patient_split(M)
    print(f"patient-level split tr/va/te = {len(tr)}/{len(va)}/{len(te)}", flush=True)
    print("TEST per organ:", dict(Counter(M[i]["organ"] for i in te)), flush=True)

    texts_tr = [M[i]["organ"] for i in tr]
    texts_va = [M[i]["organ"] for i in va]
    cfg = {"model_save_path": CKPT, "pretrained_path": None, "epochs": EPOCHS, "batch_size": 4,
           "patience": PATIENCE, "strong_augment": True, "use_boxes": False,   # concept prompt, no box
           "freeze_blocks": 20, "encoder_lr": 1e-5, "decoder_lr": 1e-4, "warmup_epochs": 2,
           "cosine_T0": 40, "dice_weight": 0.7, "focal_weight": 0.3,
           "texts_tr": texts_tr, "texts_va": texts_va}                          # per-slice organ prompts
    port = find_free_port()
    mp.spawn(train_worker_v3, args=(WORLD_SIZE, port, cfg, X[tr], Y[tr], X[va], Y[va]),
             nprocs=WORLD_SIZE, join=True)

    per_dataset, per_organ = eval_per_organ(X, Y, M, te)
    print("\n=== organ model — held-out patient-level test Dice ===", flush=True)
    print("per organ:", per_organ, flush=True)
    print("per organ/dataset:", per_dataset, flush=True)
    json.dump({"per_organ": per_organ, "per_organ_dataset": per_dataset}, open(OUT, "w"), indent=2)
    print("-> " + OUT, flush=True)


if __name__ == "__main__":
    main()
