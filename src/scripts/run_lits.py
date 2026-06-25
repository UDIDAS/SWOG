#!/usr/bin/env python3
"""
LiTS AUSAM Reproduction — matching original notebook exactly
-------------------------------------------------------------
Original notebook: SAM-DBSCAN_LiTS_Tumor (copy).ipynb

Key original settings:
  - Data: processed_images_tumor.npy (7153 grayscale slices)
  - Filter: label_threshold=50 (keep slices with >=50 tumor pixels)
  - Split: 70/10/20 (train/val/test), random_state=42
  - Training: single GPU, batch_size=1, lr=1e-5
  - Epochs: 1000, patience=10, data_increment=5
  - Images: grayscale float32 [0,1], converted to 3-channel at train time
  - Processor: do_rescale=False (already [0,1])

We only change:
  - point_annotations -> input_points (API compatibility)
  - DDP with 2 GPUs (hardware, batch_size=1 per GPU to match)
  - broadcast improved flag (DDP stability)
"""
import sys, os
sys.path.insert(0, "/home/ud3d4/Desktop/SWOG/src/scripts")
from run_flare import *

LITS_DIR = "/scratch/ud3d4/acm_data/LiTS"
LITS_OUT = "/scratch/ud3d4/acm_data/LiTS/runs"


def load_lits_notebook_exact(label_threshold=50):
    """Load LiTS data exactly as the original notebook does.
    - Load processed_images_tumor.npy (7153 grayscale slices)
    - Filter slices with < label_threshold tumor pixels
    - Split 70/10/20
    - Keep as float32 [0,1] grayscale, convert to 3-channel
    """
    imgs = np.load(os.path.join(LITS_DIR, "processed_images_tumor.npy"))   # (7153, 256, 256) float32
    segs = np.load(os.path.join(LITS_DIR, "processed_segmentations_tumor.npy"))  # (7153, 256, 256) int64

    # Filter: keep slices with >= label_threshold tumor pixels (notebook Cell 22)
    keep = []
    for i in range(len(segs)):
        if np.sum(segs[i] > 0) >= label_threshold:
            keep.append(i)
    imgs = imgs[keep]
    segs = (segs[keep] > 0).astype(np.uint8)
    print(f"After filtering (>={label_threshold}px): {len(imgs)} slices (from 7153)")

    # Convert grayscale to 3-channel (matching notebook: np.repeat(image[None,:,:], 3, axis=0))
    # But in HWC format for our pipeline
    imgs_rgb = np.stack([imgs, imgs, imgs], axis=-1)  # (N, 256, 256, 3) float32 [0,1]

    # Split exactly as notebook: first 80/20, recombine, then 70/10/20
    x_tr_init, x_te_init, y_tr_init, y_te_init = train_test_split(
        imgs_rgb, segs, test_size=0.2, random_state=42)
    x_combined = np.concatenate([x_tr_init, x_te_init], axis=0)
    y_combined = np.concatenate([y_tr_init, y_te_init], axis=0)
    x_train, x_temp, y_train, y_temp = train_test_split(
        x_combined, y_combined, test_size=0.3, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=2/3, random_state=42)

    print(f"LiTS (notebook-exact): train={len(x_train)}, val={len(x_val)}, test={len(x_test)}")
    print(f"  Tumor fraction: {y_train.mean()*100:.2f}%")
    print(f"  Image format: {x_train.dtype} [{x_train.min():.2f}, {x_train.max():.2f}]")

    # Convert to uint8 for SAM processor consistency (do_rescale=False expects [0,255])
    # Actually the notebook uses do_rescale=False with float [0,1] — but our prepare_batch
    # expects uint8. Let's keep float and not convert — the SAM processor handles float [0,1].
    # We need to pass float images to prepare_batch, so let's convert to uint8 [0,255]
    x_train = (x_train * 255).astype(np.uint8)
    x_val = (x_val * 255).astype(np.uint8)
    x_test = (x_test * 255).astype(np.uint8)

    return x_train, y_train, x_val, y_val, x_test, y_test


def get_lits_coords(labels, split_name):
    cache = os.path.join(LITS_OUT, f"{split_name}_coords_lits_v2.npy")
    if os.path.exists(cache):
        return np.load(cache, allow_pickle=True)
    print(f"  Generating {split_name} coordinates...")
    coords = generate_coordinates(labels)
    np.save(cache, coords)
    return coords


def go():
    os.makedirs(LITS_OUT, exist_ok=True)
    os.makedirs(os.path.join(LITS_OUT, "viz"), exist_ok=True)

    global OUTPUT_DIR
    OUTPUT_DIR = LITS_OUT

    x_train, y_train, x_val, y_val, x_test, y_test = load_lits_notebook_exact(label_threshold=50)
    tr_c = get_lits_coords(y_train, "train")
    va_c = get_lits_coords(y_val, "val")
    te_c = get_lits_coords(y_test, "test")
    print(f"  Coords: train={len(tr_c)}, val={len(va_c)}, test={len(te_c)}")

    # ══════════════════════════════════════════════════════════════════════
    # Exp 1: Entropy Curriculum (notebook Cell 40 exact)
    # Original: single GPU, batch=1, patience=10, increment=5, epochs=1000
    # We use: DDP 2 GPU, batch=1 (same per-GPU), same patience/increment
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("LiTS Exp 1: Entropy Curriculum (notebook exact)")
    print("  patience=10, increment=5, batch=1, epochs=1000")
    print("="*70)
    e1 = os.path.join(LITS_OUT, "lits_v2_entropy.pth")
    if not os.path.exists(e1):
        run_curriculum({"model_save_path": e1, "epochs": 1000, "batch_size": 1, "lr": 1e-5,
                        "patience": 10, "data_increment_patience": 5},
                       x_train, y_train, x_val, y_val, tr_c, va_c, order="e2h")
    evaluate_test(e1, x_test, y_test, te_c, viz_tag="lits_v2_entropy")

    # ══════════════════════════════════════════════════════════════════════
    # Exp 2: Single-stage 100% data (AUSAM-RPSF — best original)
    # Original: 100% data, 78 epochs, Dice 0.927
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("LiTS Exp 2: Single-Stage 100% (target: 0.927)")
    print("  batch=1, patience=15, epochs=250")
    print("="*70)
    e2 = os.path.join(LITS_OUT, "lits_v2_single_stage.pth")
    if not os.path.exists(e2):
        run_training({"model_save_path": e2, "epochs": 250, "batch_size": 1, "lr": 1e-5, "patience": 15},
                     x_train, y_train, x_val, y_val, tr_c, va_c)
    evaluate_test(e2, x_test, y_test, te_c, viz_tag="lits_v2_single_stage")

    # ══════════════════════════════════════════════════════════════════════
    # Exp 3: Traditional Increments
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("LiTS Exp 3: Traditional Increments")
    print("  batch=1, epochs=1000")
    print("="*70)
    e3 = os.path.join(LITS_OUT, "lits_v2_traditional.pth")
    if not os.path.exists(e3):
        run_traditional_increments({"model_save_path": e3, "epochs": 1000, "batch_size": 1, "lr": 1e-5},
                                   x_train, y_train, x_val, y_val, tr_c, va_c)
    evaluate_test(e3, x_test, y_test, te_c, viz_tag="lits_v2_traditional")

    print("\n" + "="*70)
    print("LiTS V2 COMPLETE")
    print("="*70)


if __name__ == '__main__':
    go()
