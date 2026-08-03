#!/usr/bin/env python3
"""Per-dataset autonomous test Dice for the final generic tumor model (sam3_tumor_generic, 0.938 pooled).
Reconstructs the exact seed-42 held-out test split used in training and evaluates the canonical model
per source dataset (LiTS / Pancreas / FLARE / KiTS) with text='tumor', no box. This is the honest
per-dataset breakdown of where the tumor model was trained and tested."""
import sys
from collections import Counter

sys.path.insert(0, "/home/ud3d4/Desktop/SWOG/src/scripts")
from train_tumor_generic_v3 import load_all, split, autonomous_eval

X, Y, M = load_all()
tr, va, te = split(M)
print("pool per dataset      :", dict(Counter(m["dataset"] for m in M)), flush=True)
print("TEST slices per dataset:", dict(Counter(M[i]["dataset"] for i in te)), flush=True)
print(f"train/val/test slices : {len(tr)}/{len(va)}/{len(te)}", flush=True)
autonomous_eval(X, Y, M, te)   # loads CKPT (= canonical v3), prints per-dataset Dice
