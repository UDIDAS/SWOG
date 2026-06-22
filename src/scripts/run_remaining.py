#!/usr/bin/env python3
"""Run remaining FLARE sections (S3 fixed curriculum + S5 + S6)."""
import sys, os
sys.path.insert(0, "/home/ud3d4/Desktop/SWOG")
from run_flare import *

def go():
    OUT = "/scratch/ud3d4/acm_data/FLARE/runs"

    # ── S3: Curriculum on Class 4 (Pancreas) ──
    print("\n" + "="*70)
    print("SECTION 3: Curriculum Learning — Class 4 (Pancreas)")
    print("="*70)

    x_tr4, y_tr4, x_va4, y_va4, x_te4, y_te4 = load_and_split(4)
    tr_c4 = get_or_generate_coords(y_tr4, "train", 4)
    va_c4 = get_or_generate_coords(y_va4, "val", 4)
    te_c4 = get_or_generate_coords(y_te4, "test", 4)

    # E2H
    s3e = os.path.join(OUT, "s3_e2h_class4.pth")
    if not os.path.exists(s3e):
        print("\n--- E2H ---")
        run_curriculum({"model_save_path": s3e, "epochs": 1000, "batch_size": 5, "lr": 1e-5},
                       x_tr4, y_tr4, x_va4, y_va4, tr_c4, va_c4, order="e2h")
    print("\n=== S3 E2H Test ===")
    evaluate_test(s3e, x_te4, y_te4, te_c4)

    # H2E
    s3h = os.path.join(OUT, "s3_h2e_class4.pth")
    if not os.path.exists(s3h):
        print("\n--- H2E ---")
        run_curriculum({"model_save_path": s3h, "epochs": 1000, "batch_size": 5, "lr": 1e-5},
                       x_tr4, y_tr4, x_va4, y_va4, tr_c4, va_c4, order="h2e")
    print("\n=== S3 H2E Test ===")
    evaluate_test(s3h, x_te4, y_te4, te_c4)

    # ── S5: Transfer Learning ──
    print("\n" + "="*70)
    print("SECTION 5: Transfer Learning — Liver -> Duodenum")
    print("="*70)

    x_tr1, y_tr1, x_va1, y_va1, x_te1, y_te1 = load_and_split(1)
    tr_c1 = get_or_generate_coords(y_tr1, "train", 1)
    va_c1 = get_or_generate_coords(y_va1, "val", 1)

    s5_liver = os.path.join(OUT, "s5_liver_class1.pth")
    if not os.path.exists(s5_liver):
        print("--- Training Liver ---")
        run_training({"model_save_path": s5_liver, "epochs": 250, "batch_size": 5, "lr": 1e-5, "patience": 10},
                     x_tr1, y_tr1, x_va1, y_va1, tr_c1, va_c1)

    x_tr12, y_tr12, x_va12, y_va12, x_te12, y_te12 = load_and_split(12)
    tr_c12 = get_or_generate_coords(y_tr12, "train", 12)
    va_c12 = get_or_generate_coords(y_va12, "val", 12)
    te_c12 = get_or_generate_coords(y_te12, "test", 12)

    s5_xfer = os.path.join(OUT, "s5_transfer_class12.pth")
    if not os.path.exists(s5_xfer):
        print("--- Transfer to Duodenum ---")
        run_training({"model_save_path": s5_xfer, "pretrained_path": s5_liver,
                      "epochs": 250, "batch_size": 5, "lr": 1e-5, "patience": 10},
                     x_tr12, y_tr12, x_va12, y_va12, tr_c12, va_c12)
    print("\n=== S5 Transfer Test ===")
    evaluate_test(s5_xfer, x_te12, y_te12, te_c12)

    # ── S6: Tumor ──
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
    evaluate_test(s6, x_te14, y_te14, te_c14)

    print("\n" + "="*70)
    print("ALL SECTIONS COMPLETE")
    print("="*70)

if __name__ == '__main__':
    go()
