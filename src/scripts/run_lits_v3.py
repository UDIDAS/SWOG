#!/usr/bin/env python3
"""
LiTS v3 — Aug + Box prompts experiment.
Extracts tumor slices from raw 3D volumes (preprocessed data is gone),
trains single-stage with augmentation + box prompts, evaluates on test set.

Compared to original run_lits.py (v2):
  - v2: batch=1, no aug, no box → Test Dice 0.901
  - v3: batch=5, aug + box → target: beat 0.901
"""
import sys, os
sys.path.insert(0, "/home/ud3d4/Desktop/SWOG/src/scripts")
from run_flare import *

LITS_DATA = "/scratch/ud3d4/acm_data/Data"
LITS_OUT = "/scratch/ud3d4/acm_data/LiTS_v3/runs"
os.makedirs(LITS_OUT, exist_ok=True)

HU_MIN, HU_MAX = -100, 400


def hu_to_rgb(slice_hu):
    clipped = np.clip(slice_hu, HU_MIN, HU_MAX)
    normalized = ((clipped - HU_MIN) / (HU_MAX - HU_MIN) * 255).astype(np.uint8)
    return np.stack([normalized] * 3, axis=-1)


def get_volume_ids():
    return sorted([int(f.replace("volume-", "").replace(".npy", ""))
                   for f in os.listdir(f"{LITS_DATA}/ct") if f.startswith("volume-")])


def extract_tumor_slices(vol_ids, label_threshold=50):
    all_images, all_labels = [], []
    for vid in vol_ids:
        ct = np.load(f"{LITS_DATA}/ct/volume-{vid}.npy")
        seg = np.load(f"{LITS_DATA}/seg/segmentation-{vid}.npy")
        for z in range(ct.shape[0]):
            mask = (seg[z] == 2).astype(np.uint8)
            if mask.sum() >= label_threshold:
                all_images.append(hu_to_rgb(ct[z]))
                all_labels.append(mask)
    return np.array(all_images), np.array(all_labels)


def get_coords(labels, split_name):
    cache = os.path.join(LITS_OUT, f"{split_name}_coords.npy")
    if os.path.exists(cache):
        return np.load(cache, allow_pickle=True)
    print(f"  Generating {split_name} coordinates...")
    coords = generate_coordinates(labels)
    np.save(cache, coords)
    return coords


def go():
    vol_ids = get_volume_ids()
    # Case-level split matching run_lits_nifti.py
    train_vols, test_vols = train_test_split(vol_ids, test_size=0.2, random_state=42)
    train_vols, val_vols = train_test_split(train_vols, test_size=0.125, random_state=42)
    print(f"Volumes: train={len(train_vols)}, val={len(val_vols)}, test={len(test_vols)}")

    print("Extracting tumor slices...")
    x_train, y_train = extract_tumor_slices(train_vols)
    x_val, y_val = extract_tumor_slices(val_vols)
    x_test, y_test = extract_tumor_slices(test_vols)
    print(f"Slices: train={len(x_train)}, val={len(x_val)}, test={len(x_test)}")

    tr_c = get_coords(y_train, "train")
    va_c = get_coords(y_val, "val")
    te_c = get_coords(y_test, "test")

    # ══════════════════════════════════════════════════════════════════════
    # Single-stage with augmentation + box prompts
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("LiTS v3: Single-Stage + Aug + Box (target: beat 0.901)")
    print("=" * 70)

    model_path = os.path.join(LITS_OUT, "lits_v3_augbox.pth")
    if not os.path.exists(model_path):
        run_training(
            {"model_save_path": model_path, "epochs": 250, "batch_size": 5,
             "lr": 1e-5, "patience": 15, "augment": True, "use_boxes": True},
            x_train, y_train, x_val, y_val, tr_c, va_c)

    evaluate_test(model_path, x_test, y_test, te_c, viz_tag="lits_v3_augbox")

    print("\n" + "=" * 70)
    print("LiTS v3 COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    go()
