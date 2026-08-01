#!/usr/bin/env python3
"""Train + test-eval the Pancreas AUSAM models (organ label=1, tumor label=2) on the MSD Task07
volumes. Organ: H2E curriculum; Tumor: Traditional Increments w/ transfer from organ. Held-out 20%
of CASES for test (seed 42) -> real train+test Dice. Skips training if the .pth already exists.
Run pinned to one GPU: CUDA_VISIBLE_DEVICES=1 python train_pancreas_ausam.py"""
import os
import sys

sys.path.insert(0, "/home/ud3d4/Desktop/SWOG/src/scripts")
from sklearn.model_selection import train_test_split          # noqa: E402
import run_flare                                               # noqa: E402
from run_pancreas_nifti import (                               # noqa: E402
    train_pancreas_model, get_case_ids, extract_slices_from_volumes,
    OUTPUT_DIR, ORGAN_WEIGHTS, TUMOR_WEIGHTS)
from run_flare import evaluate_test, generate_coordinates      # noqa: E402
run_flare.OUTPUT_DIR = OUTPUT_DIR

# ── train (each skips if its .pth exists) ──
train_pancreas_model(os.path.join(OUTPUT_DIR, ORGAN_WEIGHTS), label_value=1, label_name="organ")
train_pancreas_model(os.path.join(OUTPUT_DIR, TUMOR_WEIGHTS), label_value=2, label_name="tumor")

# ── test-eval on the held-out cases (same split as training) ──
cids = get_case_ids()
_tr, test_cases = train_test_split(cids, test_size=0.2, random_state=42)
print(f"\ntest cases held out: {len(test_cases)}", flush=True)
for wname, lab, tag in [(ORGAN_WEIGHTS, 1, "organ"), (TUMOR_WEIGHTS, 2, "tumor")]:
    p = os.path.join(OUTPUT_DIR, wname)
    if not os.path.exists(p):
        print(f"[skip eval] {wname} missing (training may have failed)", flush=True)
        continue
    xt, yt, _ = extract_slices_from_volumes(test_cases, lab)
    tec = generate_coordinates(yt)
    print(f"\n===== EVAL pancreas {tag} ({len(xt)} test slices) =====", flush=True)
    try:
        res = evaluate_test(p, xt, yt, tec, viz_tag=f"pancreas_{tag}")
        print(f"RESULT pancreas_{tag}: test Dice={res['dice'][0]:.4f} +/- {res['dice'][1]:.4f}", flush=True)
    except Exception as e:
        print(f"[error] {tag}: {type(e).__name__}: {e}", flush=True)
