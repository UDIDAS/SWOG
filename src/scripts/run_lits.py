#!/usr/bin/env python3
"""
LiTS AUSAM Reproduction
-----------------------
Reproduces the LiTS tumor segmentation experiments from:
  - SAM-DBSCAN_LiTS_Tumor (copy).ipynb
  - SAM-DBSCAN_LiTS_Tumor_TF_125pixels.ipynb

LiTS (Liver Tumor Segmentation) dataset: 131 abdominal CT scans.
Each 3D scan sliced into 2D axial slices (256x256), tumor-bearing slices kept.
Pre-split into train (3,394) / val (485) / test (970).

Original notebook approach (single GPU, entropy curriculum):
  - Cell 40: Entropy curriculum, patience=10, increment=5, epochs=1000
  - Cell 42: Continue training to 100% data
  - Test result: Dice 0.857

Original best results (from experiments_results_summary.csv):
  - AUSAM-HPS (entropy 48.6%, single): Dice 0.875
  - AUSAM-HPM (entropy 48.6%, multi):  Dice 0.902
  - AUSAM-RPS (random, single):        Dice 0.903
  - AUSAM-RPSF (100% full):            Dice 0.927
  - Liver:                              Dice 0.963

Our experiments:
  Exp 1: Entropy curriculum (matches notebook Cell 40 exactly)
  Exp 2: Single-stage 100% (matches AUSAM-RPSF — best original result)
  Exp 3: Traditional Increments (best method from FLARE findings)
"""
import sys, os
sys.path.insert(0, "/home/ud3d4/Desktop/SWOG/src/scripts")
from run_flare import *

LITS_DIR = "/scratch/ud3d4/acm_data/LiTS"
LITS_OUT = "/scratch/ud3d4/acm_data/LiTS/runs"


def load_lits_splits():
    """Load pre-split LiTS data. Images are 256x256x3 float32 [0,1]."""
    x_train = np.load(os.path.join(LITS_DIR, "splits/x_train.npy"))
    x_val = np.load(os.path.join(LITS_DIR, "splits/x_val.npy"))
    x_test = np.load(os.path.join(LITS_DIR, "splits/x_test.npy"))
    y_train = np.load(os.path.join(LITS_DIR, "splits/y_train.npy"))
    y_val = np.load(os.path.join(LITS_DIR, "splits/y_val.npy"))
    y_test = np.load(os.path.join(LITS_DIR, "splits/y_test.npy"))

    y_train = (y_train > 0).astype(np.uint8)
    y_val = (y_val > 0).astype(np.uint8)
    y_test = (y_test > 0).astype(np.uint8)

    # Convert float [0,1] to uint8 [0,255] for SAM processor consistency
    if x_train.max() <= 1.0:
        x_train = (x_train * 255).astype(np.uint8)
        x_val = (x_val * 255).astype(np.uint8)
        x_test = (x_test * 255).astype(np.uint8)

    print(f"LiTS: train={x_train.shape[0]}, val={x_val.shape[0]}, test={x_test.shape[0]}")
    print(f"  Tumor fraction: {y_train.mean()*100:.2f}%")
    return x_train, y_train, x_val, y_val, x_test, y_test


def get_lits_coords(labels, split_name):
    cache = os.path.join(LITS_OUT, f"{split_name}_coords_lits.npy")
    if os.path.exists(cache):
        return np.load(cache, allow_pickle=True)
    print(f"  Generating {split_name} coordinates...")
    coords = generate_coordinates(labels)
    np.save(cache, coords)
    return coords


def go():
    os.makedirs(LITS_OUT, exist_ok=True)
    os.makedirs(os.path.join(LITS_OUT, "viz"), exist_ok=True)

    # Override OUTPUT_DIR for LiTS visualizations
    global OUTPUT_DIR
    OUTPUT_DIR = LITS_OUT

    x_train, y_train, x_val, y_val, x_test, y_test = load_lits_splits()
    tr_c = get_lits_coords(y_train, "train")
    va_c = get_lits_coords(y_val, "val")
    te_c = get_lits_coords(y_test, "test")
    print(f"  Coords: train={len(tr_c)}, val={len(va_c)}, test={len(te_c)}")

    # ══════════════════════════════════════════════════════════════════════
    # Exp 1: Entropy Curriculum (matches notebook Cell 40 exactly)
    # Original notebook: patience=10, increment=5, epochs=1000, single GPU
    # We use DDP (2 GPU) with same patience/increment as the notebook
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("LiTS Exp 1: Entropy Curriculum (notebook Cell 40)")
    print("  Original: patience=10, increment=5, epochs=1000")
    print("  Target: Dice 0.857 (notebook result)")
    print("="*70)
    e1 = os.path.join(LITS_OUT, "lits_entropy_curriculum.pth")
    if not os.path.exists(e1):
        run_curriculum({"model_save_path": e1, "epochs": 1000, "batch_size": 5, "lr": 1e-5,
                        "patience": 10, "data_increment_patience": 5},
                       x_train, y_train, x_val, y_val, tr_c, va_c, order="e2h")
    evaluate_test(e1, x_test, y_test, te_c, viz_tag="lits_entropy_curriculum")

    # ══════════════════════════════════════════════════════════════════════
    # Exp 2: Single-stage 100% data (AUSAM-RPSF — best original result)
    # Original: 100% data from epoch 1, no curriculum
    # Target: Dice 0.927
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("LiTS Exp 2: Single-Stage 100% Data (AUSAM-RPSF)")
    print("  Target: Dice 0.927 (best original result)")
    print("="*70)
    e2 = os.path.join(LITS_OUT, "lits_single_stage.pth")
    if not os.path.exists(e2):
        run_training({"model_save_path": e2, "epochs": 250, "batch_size": 5, "lr": 1e-5, "patience": 15},
                     x_train, y_train, x_val, y_val, tr_c, va_c)
    evaluate_test(e2, x_test, y_test, te_c, viz_tag="lits_single_stage")

    # ══════════════════════════════════════════════════════════════════════
    # Exp 3: Traditional Increments (best method from FLARE Tumor)
    # Fixed percentage steps with random sampling
    # ══════════════════════════════════════════════════════════════════════
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
