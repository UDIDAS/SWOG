#!/usr/bin/env python3
"""
Round 2: Match exact original notebook strategies to close the gap.

Pancreas (paper1 notebook):
  - Original used 4 GPUs, 1000 epochs, patience=10
  - We increase patience to 20 to compensate for 2 GPUs
  - Re-run H2E with more room for data expansion

Tumor (Tumor notebook):
  - Original best result came from "Traditional With Increments" (NOT H2E)
  - Fixed percentage steps: 14.8% -> 30.3% -> ... -> 100%
  - 4 GPUs, 1000 epochs, patience=10, percentage_switch_patience=5
"""
import sys, os
sys.path.insert(0, "/home/ud3d4/Desktop/SWOG/src/scripts")
from run_flare import *

def go():
    OUT = "/scratch/ud3d4/acm_data/FLARE/runs"

    # ══════════════════════════════════════════════════════════════════════
    # PANCREAS Round 2: H2E with increased patience (20 vs 10)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("ROUND 2 — Pancreas H2E (patience=20 to allow more data expansion)")
    print("="*70)

    x_tr4, y_tr4, x_va4, y_va4, x_te4, y_te4 = load_and_split(4)
    tr_c4 = get_or_generate_coords(y_tr4, "train", 4)
    va_c4 = get_or_generate_coords(y_va4, "val", 4)
    te_c4 = get_or_generate_coords(y_te4, "test", 4)

    r2_pancreas = os.path.join(OUT, "r2_h2e_patience20_class4.pth")
    if not os.path.exists(r2_pancreas):
        run_curriculum({"model_save_path": r2_pancreas, "epochs": 1000, "batch_size": 5, "lr": 1e-5,
                        "patience": 20, "data_increment_patience": 3},
                       x_tr4, y_tr4, x_va4, y_va4, tr_c4, va_c4, order="h2e")
    print("\n=== Round 2 Pancreas H2E Test ===")
    evaluate_test(r2_pancreas, x_te4, y_te4, te_c4, viz_tag="r2_h2e_patience20_class4")

    # ══════════════════════════════════════════════════════════════════════
    # TUMOR Round 2: Traditional With Increments (exact notebook strategy)
    # This is what produced the original's best Dice 0.839
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("ROUND 2 — Tumor: Traditional With Increments (original best method)")
    print("="*70)

    x_tr14, y_tr14, x_va14, y_va14, x_te14, y_te14 = load_and_split(14)
    tr_c14 = get_or_generate_coords(y_tr14, "train", 14)
    va_c14 = get_or_generate_coords(y_va14, "val", 14)
    te_c14 = get_or_generate_coords(y_te14, "test", 14)

    r2_tumor = os.path.join(OUT, "r2_traditional_class14.pth")
    if not os.path.exists(r2_tumor):
        # Exact percentages from the original Tumor notebook (Cell 30)
        run_traditional_increments(
            {"model_save_path": r2_tumor, "epochs": 1000, "batch_size": 5, "lr": 1e-5},
            x_tr14, y_tr14, x_va14, y_va14, tr_c14, va_c14,
        )
    print("\n=== Round 2 Tumor Traditional Test ===")
    evaluate_test(r2_tumor, x_te14, y_te14, te_c14, viz_tag="r2_traditional_class14")

    # ══════════════════════════════════════════════════════════════════════
    # TUMOR Round 2b: E2H (the other strategy from the notebook)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("ROUND 2b — Tumor: E2H Curriculum")
    print("="*70)

    r2_tumor_e2h = os.path.join(OUT, "r2_e2h_class14.pth")
    if not os.path.exists(r2_tumor_e2h):
        run_curriculum({"model_save_path": r2_tumor_e2h, "epochs": 1000, "batch_size": 5, "lr": 1e-5},
                       x_tr14, y_tr14, x_va14, y_va14, tr_c14, va_c14, order="e2h")
    print("\n=== Round 2b Tumor E2H Test ===")
    evaluate_test(r2_tumor_e2h, x_te14, y_te14, te_c14, viz_tag="r2_e2h_class14")

    # ══════════════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("ROUND 2 COMPLETE")
    print("="*70)

if __name__ == '__main__':
    go()
