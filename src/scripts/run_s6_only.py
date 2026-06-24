#!/usr/bin/env python3
"""Evaluate S5 Transfer and run S6 Tumor."""
import sys, os
sys.path.insert(0, "/home/ud3d4/Desktop/SWOG/src/scripts")
from run_flare import *

def go():
    OUT = "/scratch/ud3d4/acm_data/FLARE/runs"

    # Evaluate S5 Transfer
    print("\n" + "="*70)
    print("S5 Transfer — Test Evaluation")
    print("="*70)
    x_tr12, y_tr12, x_va12, y_va12, x_te12, y_te12 = load_and_split(12)
    te_c12 = get_or_generate_coords(y_te12, "test", 12)
    s5_xfer = os.path.join(OUT, "s5_transfer_class12.pth")
    evaluate_test(s5_xfer, x_te12, y_te12, te_c12, viz_tag="s5_transfer_class12")

    # S6: Tumor
    print("\n" + "="*70)
    print("SECTION 6: Tumor — Class 14")
    print("="*70)
    x_tr14, y_tr14, x_va14, y_va14, x_te14, y_te14 = load_and_split(14)
    tr_c14 = get_or_generate_coords(y_tr14, "train", 14)
    va_c14 = get_or_generate_coords(y_va14, "val", 14)
    te_c14 = get_or_generate_coords(y_te14, "test", 14)

    s6 = os.path.join(OUT, "s6_h2e_class14.pth")
    if not os.path.exists(s6):
        run_curriculum({"model_save_path": s6, "epochs": 1000, "batch_size": 5, "lr": 1e-5},
                       x_tr14, y_tr14, x_va14, y_va14, tr_c14, va_c14, order="h2e")
    print("\n=== S6 Tumor Test ===")
    evaluate_test(s6, x_te14, y_te14, te_c14, viz_tag="s6_h2e_class14")

    print("\n" + "="*70)
    print("DONE")
    print("="*70)

if __name__ == '__main__':
    go()
