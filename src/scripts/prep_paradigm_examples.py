#!/usr/bin/env python3
"""Prepare REAL test-case examples for the two-model-paradigm notebook.

On the SAME held-out patient-level test cases (seed 42, box-free text='tumor'), run:
  A) ORIGINAL  = LiTS-only tumor model  (single-dataset paradigm)
  B) OURS      = pooled tumor model     (all 4 datasets — the two-model paradigm)
Pick a few clear, real cases per cancer type (plus one liver control the original DID train on), and save the
actual images + ground truth + both models' predicted masks so the notebook can show them with no GPU.

Output: results/paradigm_examples.npz  (imgs, gts, pred_orig, pred_ours)
        results/paradigm_examples.json (per-dataset mean Dice + per-example metadata, in array order)
Needs one GPU. ~10 min.
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from torch.amp import autocast

sys.path.insert(0, "/home/ud3d4/Desktop/SWOG/src/scripts")
from train_tumor_incremental import load_all, patient_split, CKPT_DIR
from run_pancreas_sam3 import _load_sam3_ckpt, SAM3_MODEL_ID, HF_TOKEN, extract_best_mask_soft

DEV = "cuda:0"
OURS = f"{CKPT_DIR}/sam3_tumor_generic.pth"                                # pooled, 4 datasets
ORIG = f"{CKPT_DIR}/sam3_tumor_s1_lits_patientlevel.pth"                   # clean LiTS-only
if not os.path.exists(ORIG):
    ORIG = f"{CKPT_DIR}/sam3_tumor_s1_lits_kg_patientlevel.pth"
MINPX = 40
# per-dataset selection rule: (original-fails-below, ours-succeeds-above, how-many)
SEL = {"pancreas": (0.10, 0.50, 2), "kits": (0.40, 0.55, 2), "flare": (0.40, 0.55, 2)}
DSNAME = {"pancreas": "MSD07 pancreatic tumor", "kits": "KiTS23 kidney tumor",
          "flare": "FLARE23 pan-cancer tumor", "lits": "LiTS liver tumor"}


def dice(p, g):
    s = int(p.sum()) + int(g.sum())
    return 2 * int((p & g).sum()) / s if s else 1.0


def predict(model, proc, img):
    inp = proc(images=[img], text=["tumor"], return_tensors="pt")
    kw = {"pixel_values": inp["pixel_values"].to(DEV)}
    for k in ("input_ids", "attention_mask"):
        if inp.get(k) is not None:
            kw[k] = inp[k].to(DEV)
    with torch.no_grad(), autocast("cuda"):
        out = model(**kw)
        pm = extract_best_mask_soft(out.pred_masks, out.pred_logits)
        pm = torch.nn.functional.interpolate(pm.float(), size=(256, 256), mode="bilinear", align_corners=False)
    return pm.sigmoid().squeeze().cpu().numpy() > 0.5


def main():
    from transformers import Sam3Processor
    proc = Sam3Processor.from_pretrained(SAM3_MODEL_ID, token=HF_TOKEN)
    X, Y, M = load_all()
    _, _, te = patient_split(M)
    present = [i for i in te if int((Y[i] > 0).sum()) >= MINPX]
    print(f"ORIG={os.path.basename(ORIG)}  OURS={os.path.basename(OURS)}  present test slices={len(present)}", flush=True)

    per = {"orig": defaultdict(list), "ours": defaultdict(list)}
    rec = {}                                       # idx -> [dataset, case, d_orig, d_ours]
    for tag, ckpt in [("orig", ORIG), ("ours", OURS)]:
        model = _load_sam3_ckpt(ckpt, DEV)
        for n, i in enumerate(present):
            d = dice(predict(model, proc, X[i]), Y[i] > 0)
            per[tag][M[i]["dataset"]].append(d)
            r = rec.setdefault(i, [M[i]["dataset"], M[i]["case"], None, None])
            r[2 if tag == "orig" else 3] = d
            if (n + 1) % 800 == 0:
                print(f"  [{tag}] {n+1}/{len(present)}", flush=True)
        del model; torch.cuda.empty_cache()

    agg = {ds: {"orig": round(float(np.mean(per["orig"][ds])), 4),
                "ours": round(float(np.mean(per["ours"][ds])), 4),
                "n_slices": len(per["orig"][ds]),
                "n_patients": len({M[i]["case"] for i in present if M[i]["dataset"] == ds})}
           for ds in sorted(per["orig"])}
    print("per-dataset mean Dice:", json.dumps(agg, indent=1), flush=True)

    rows = [[i] + rec[i] for i in present]         # [idx, ds, case, d_orig, d_ours]
    picks = []
    # cross-dataset cancers the original never trained on: the miss->hit cases
    for ds, (lo, hi, k) in SEL.items():
        cand = sorted([r for r in rows if r[1] == ds and r[3] < lo and r[4] > hi],
                      key=lambda r: r[4] - r[3], reverse=True)
        seen = set()
        for r in cand:
            if r[2] in seen:
                continue
            seen.add(r[2]); picks.append(r)
            if sum(p[1] == ds for p in picks) >= k:
                break
    # one liver CONTROL the original DID train on (both should do OK) — sharpens the point
    ctrl = sorted([r for r in rows if r[1] == "lits" and r[3] > 0.5 and r[4] > 0.5],
                  key=lambda r: min(r[3], r[4]), reverse=True)
    if ctrl:
        picks = [ctrl[0]] + picks

    # re-predict masks for the picked cases and stack
    imgs, gts, po, pu, meta = [], [], [], [], []
    mo = _load_sam3_ckpt(ORIG, DEV)
    preds_o = {r[0]: predict(mo, proc, X[r[0]]) for r in picks}
    del mo; torch.cuda.empty_cache()
    mu = _load_sam3_ckpt(OURS, DEV)
    preds_u = {r[0]: predict(mu, proc, X[r[0]]) for r in picks}
    del mu; torch.cuda.empty_cache()
    for i, ds, case, do, du in picks:
        imgs.append(X[i]); gts.append((Y[i] > 0)); po.append(preds_o[i]); pu.append(preds_u[i])
        meta.append({"dataset": ds, "dataset_name": DSNAME[ds], "case": case,
                     "seen_by_original": ds == "lits",
                     "dice_orig": round(float(dice(preds_o[i], Y[i] > 0)), 3),
                     "dice_ours": round(float(dice(preds_u[i], Y[i] > 0)), 3)})

    np.savez_compressed("/home/ud3d4/Desktop/SWOG/results/paradigm_examples.npz",
                        imgs=np.stack(imgs), gts=np.stack(gts),
                        pred_orig=np.stack(po), pred_ours=np.stack(pu))
    json.dump({"orig_ckpt": os.path.basename(ORIG), "ours_ckpt": os.path.basename(OURS),
               "per_dataset_mean_dice": agg, "examples": meta},
              open("/home/ud3d4/Desktop/SWOG/results/paradigm_examples.json", "w"), indent=2)
    print(f"saved {len(picks)} example cases -> results/paradigm_examples.npz + .json", flush=True)
    for m in meta:
        print(f"  {m['dataset_name']:26s} {m['case']:14s} original={m['dice_orig']:.2f}  ours={m['dice_ours']:.2f}", flush=True)


if __name__ == "__main__":
    main()
