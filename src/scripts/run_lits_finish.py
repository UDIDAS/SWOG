#!/usr/bin/env python3
"""Evaluate LiTS Exp 2 and run Exp 3."""
import sys, os
sys.path.insert(0, "/home/ud3d4/Desktop/SWOG/src/scripts")
from run_flare import *

def go():
    LITS_DIR = "/scratch/ud3d4/acm_data/LiTS"
    LITS_OUT = "/scratch/ud3d4/acm_data/LiTS/runs"

    global OUTPUT_DIR
    OUTPUT_DIR = LITS_OUT

    # Load data
    x_train = np.load(os.path.join(LITS_DIR, "splits/x_train.npy"))
    x_val = np.load(os.path.join(LITS_DIR, "splits/x_val.npy"))
    x_test = np.load(os.path.join(LITS_DIR, "splits/x_test.npy"))
    y_train = np.load(os.path.join(LITS_DIR, "splits/y_train.npy"))
    y_val = np.load(os.path.join(LITS_DIR, "splits/y_val.npy"))
    y_test = np.load(os.path.join(LITS_DIR, "splits/y_test.npy"))

    y_train = (y_train > 0).astype(np.uint8)
    y_val = (y_val > 0).astype(np.uint8)
    y_test = (y_test > 0).astype(np.uint8)

    if x_train.max() <= 1.0:
        x_train = (x_train * 255).astype(np.uint8)
        x_val = (x_val * 255).astype(np.uint8)
        x_test = (x_test * 255).astype(np.uint8)

    print(f"LiTS: train={x_train.shape[0]}, val={x_val.shape[0]}, test={x_test.shape[0]}")

    tr_c = np.load(os.path.join(LITS_OUT, "train_coords_lits.npy"), allow_pickle=True)
    va_c = np.load(os.path.join(LITS_OUT, "val_coords_lits.npy"), allow_pickle=True)
    te_c = np.load(os.path.join(LITS_OUT, "test_coords_lits.npy"), allow_pickle=True)

    # Evaluate Exp 2
    print("\n" + "="*70)
    print("LiTS Exp 2: Single-Stage — Test Evaluation")
    print("="*70)
    evaluate_test(os.path.join(LITS_OUT, "lits_single_stage.pth"),
                  x_test, y_test, te_c, viz_tag="lits_single_stage")

    # Exp 3: Traditional Increments
    print("\n" + "="*70)
    print("LiTS Exp 3: Traditional Increments")
    print("="*70)
    e3 = os.path.join(LITS_OUT, "lits_traditional.pth")
    if not os.path.exists(e3):
        run_traditional_increments({"model_save_path": e3, "epochs": 1000, "batch_size": 5, "lr": 1e-5},
                                   x_train, y_train, x_val, y_val, tr_c, va_c)
    evaluate_test(e3, x_test, y_test, te_c, viz_tag="lits_traditional")

    print("\n" + "="*70)
    print("LiTS COMPLETE")
    print("="*70)

if __name__ == '__main__':
    go()
