#!/usr/bin/env python3
"""Re-run Tumor Traditional with global patience reset on step switch."""
import sys, os
sys.path.insert(0, "/home/ud3d4/Desktop/SWOG/src/scripts")
from run_flare import *

def go():
    OUT = "/scratch/ud3d4/acm_data/FLARE/runs"
    global OUTPUT_DIR
    OUTPUT_DIR = OUT

    x_tr14, y_tr14, x_va14, y_va14, x_te14, y_te14 = load_and_split(14)
    tr_c14 = get_or_generate_coords(y_tr14, "train", 14)
    va_c14 = get_or_generate_coords(y_va14, "val", 14)
    te_c14 = get_or_generate_coords(y_te14, "test", 14)

    path = os.path.join(OUT, "r2_traditional_class14.pth")
    if not os.path.exists(path):
        print("=== Tumor Traditional (patience reset fix) ===")
        run_traditional_increments(
            {"model_save_path": path, "epochs": 1000, "batch_size": 5, "lr": 1e-5},
            x_tr14, y_tr14, x_va14, y_va14, tr_c14, va_c14)
    evaluate_test(path, x_te14, y_te14, te_c14, viz_tag="r2_traditional_class14")

if __name__ == '__main__':
    go()
