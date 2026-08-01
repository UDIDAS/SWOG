#!/usr/bin/env python3
"""Test-evaluate the EXISTING LiTS AUSAM models (no retraining) -> real test Dice.
Uses the notebook-exact split (seed 42) + cached test coords, so it matches how the
models were trained. LiTS AUSAM target here is the tumor class (SAM-DBSCAN_LiTS_Tumor)."""
import os
import sys

sys.path.insert(0, "/home/ud3d4/Desktop/SWOG/src/scripts")
from run_flare import evaluate_test               # noqa: E402
from run_lits import load_lits_notebook_exact, get_lits_coords, LITS_OUT, go  # noqa: E402
import run_lits                                    # noqa: E402
run_lits.OUTPUT_DIR = LITS_OUT
import run_flare                                   # noqa: E402
run_flare.OUTPUT_DIR = LITS_OUT

x_tr, y_tr, x_va, y_va, x_te, y_te = load_lits_notebook_exact(label_threshold=50)
te_c = get_lits_coords(y_te, "test")
print(f"test slices={len(x_te)}  coords={len(te_c)}", flush=True)

for name in ["lits_v2_single_stage.pth", "lits_v2_entropy.pth"]:
    p = os.path.join(LITS_OUT, name)
    if not os.path.exists(p):
        print(f"[skip] {name} missing", flush=True)
        continue
    print(f"\n===== EVAL {name} =====", flush=True)
    try:
        res = evaluate_test(p, x_te, y_te, te_c, viz_tag=name.replace(".pth", ""))
        print(f"RESULT {name}: test Dice={res['dice'][0]:.4f} +/- {res['dice'][1]:.4f}", flush=True)
    except Exception as e:
        print(f"[error] {name}: {type(e).__name__}: {e}", flush=True)
