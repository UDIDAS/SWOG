#!/usr/bin/env python3
"""
LiTS SAM3 experiment — apply the best Pancreas SAM3 recipe (v3 partial-freeze)
to LiTS liver-tumor segmentation.

Reuses the SAM3 v3 training machinery from run_pancreas_sam3 (partial encoder
freeze + discriminative LR + cosine annealing + Dice+Focal loss). LiTS data is
tumor-only (binary masks); liver is taken from GT in the historical pipeline,
matching prior LiTS work. Uses the surviving pre-split slice arrays since the
raw volumes were cleared from scratch.

Run: HF_TOKEN=... python run_lits_sam3.py --v3
"""
import sys, os
sys.path.insert(0, "/home/ud3d4/Desktop/SWOG/src/scripts")

import numpy as np
import torch
import torch.multiprocessing as mp
from sklearn.model_selection import train_test_split

# Reuse all SAM3 infrastructure from the Pancreas experiment
from run_pancreas_sam3 import (
    train_worker_v3, evaluate_finetuned,
    SAM3_MODEL_ID, HF_TOKEN,
)
from run_flare import find_free_port, WORLD_SIZE

LITS_SPLITS = "/scratch/ud3d4/acm_data/LiTS/splits"
LITS_RAW = "/scratch/ud3d4/acm_data/Data"   # ct/volume-N.npy, seg/segmentation-N.npy
LITS_SAM3_DIR = "/scratch/ud3d4/acm_data/LiTS/sam3"
os.makedirs(LITS_SAM3_DIR, exist_ok=True)

HU_MIN, HU_MAX = -100, 400  # LiTS liver window


def lits_hu_to_rgb(slice_hu):
    clipped = np.clip(slice_hu, HU_MIN, HU_MAX)
    norm = ((clipped - HU_MIN) / (HU_MAX - HU_MIN) * 255).astype(np.uint8)
    return np.stack([norm] * 3, axis=-1)


def get_volume_ids():
    return sorted(int(f.replace("volume-", "").replace(".npy", ""))
                  for f in os.listdir(f"{LITS_RAW}/ct")
                  if f.startswith("volume-") and not f.startswith("._"))


def extract_slices_caselevel(vol_ids, label_value=2, threshold=50):
    """Extract uint8 RGB slices + binary masks for a label from raw volumes."""
    imgs, lbls = [], []
    for vid in vol_ids:
        ct = np.load(f"{LITS_RAW}/ct/volume-{vid}.npy")
        seg = np.load(f"{LITS_RAW}/seg/segmentation-{vid}.npy")
        for z in range(ct.shape[0]):
            mask = (seg[z] == label_value).astype(np.uint8)
            if mask.sum() >= threshold:
                imgs.append(lits_hu_to_rgb(ct[z]))
                lbls.append(mask)
    return np.array(imgs, dtype=np.uint8), np.array(lbls, dtype=np.uint8)


def load_lits_split(name):
    """Load a LiTS split; convert float[0,1] RGB -> uint8[0,255] to match the
    SAM3 pipeline (hu_to_rgb produced uint8 for Pancreas), and masks -> uint8."""
    x = np.load(f"{LITS_SPLITS}/x_{name}.npy")
    y = np.load(f"{LITS_SPLITS}/y_{name}.npy")
    if x.dtype != np.uint8:
        x = np.clip(x * 255.0, 0, 255).astype(np.uint8) if x.max() <= 1.0 + 1e-3 else x.astype(np.uint8)
    y = (y > 0.5).astype(np.uint8)
    return x, y


def run_v3():
    """SAM3 v3 (partial freeze + disc LR + cosine + focal) on LiTS tumor."""
    print("Loading LiTS splits (slice-level; raw volumes cleared from scratch)...", flush=True)
    x_train, y_train = load_lits_split("train")
    x_val, y_val = load_lits_split("val")
    x_test, y_test = load_lits_split("test")
    print(f"Slices: train={len(x_train)}, val={len(x_val)}, test={len(x_test)}", flush=True)

    label_name = "tumor"
    model_path = os.path.join(LITS_SAM3_DIR, f"lits_sam3_v3_{label_name}.pth")
    if not os.path.exists(model_path):
        print(f"\n--- V3 Fine-tuning SAM3 for LiTS {label_name} ---", flush=True)
        cfg = {
            "model_save_path": model_path,
            "epochs": 250,
            "batch_size": 2,
            "patience": 25,
            "strong_augment": True,
            "text_prompt": "visual",
            "freeze_blocks": 20,
            "encoder_lr": 1e-5,
            "decoder_lr": 1e-4,
            "warmup_epochs": 5,
            "cosine_T0": 30,
            "dice_weight": 0.7,
            "focal_weight": 0.3,
        }
        port = find_free_port()
        mp.spawn(
            train_worker_v3,
            args=(WORLD_SIZE, port, cfg, x_train, y_train, x_val, y_val),
            nprocs=WORLD_SIZE, join=True,
        )

    ft = evaluate_finetuned(model_path, x_test, y_test, tag=f"lits_{label_name}_v3_finetuned")

    print("\n" + "=" * 70)
    print("LiTS SAM3 V3 (PARTIAL FREEZE + DISC LR + COSINE + DICE+FOCAL) vs SAM1")
    print("=" * 70)
    print(f"{'Config':<45s} {'Tumor Dice':>12s}")
    print("-" * 58)
    print(f"{'SAM1 slice-level (reference)':<45s} {'0.901':>12s}")
    print(f"{'SAM1 case-level (honest ref, not reproduced)':<45s} {'0.845':>12s}")
    print(f"{'SAM3 v3 partial_freeze (slice-level)':<45s} {ft['dice'][0]:>12.4f}")
    print("=" * 70)
    print("NOTE: slice-level split (raw volumes cleared). Case-level + NIfTI need raw data restored.")
    print("DONE")


def run_v3_caselevel():
    """SAM3 v3 on LiTS tumor with a case-level (patient-held-out) split from raw volumes."""
    vol_ids = get_volume_ids()
    train_vols, test_vols = train_test_split(vol_ids, test_size=0.2, random_state=42)
    train_vols, val_vols = train_test_split(train_vols, test_size=0.125, random_state=42)
    print(f"Volumes: train={len(train_vols)}, val={len(val_vols)}, test={len(test_vols)}", flush=True)

    print("Extracting tumor slices from raw volumes (case-level)...", flush=True)
    x_train, y_train = extract_slices_caselevel(train_vols, label_value=2)
    x_val, y_val = extract_slices_caselevel(val_vols, label_value=2)
    x_test, y_test = extract_slices_caselevel(test_vols, label_value=2)
    print(f"Slices: train={len(x_train)}, val={len(x_val)}, test={len(x_test)}", flush=True)

    model_path = os.path.join(LITS_SAM3_DIR, "lits_sam3_v3_caselevel_tumor.pth")
    if not os.path.exists(model_path):
        print("\n--- V3 case-level fine-tuning SAM3 for LiTS tumor ---", flush=True)
        cfg = {
            "model_save_path": model_path, "epochs": 250, "batch_size": 2, "patience": 25,
            "strong_augment": True, "text_prompt": "visual",
            "freeze_blocks": 20, "encoder_lr": 1e-5, "decoder_lr": 1e-4,
            "warmup_epochs": 5, "cosine_T0": 30, "dice_weight": 0.7, "focal_weight": 0.3,
        }
        port = find_free_port()
        mp.spawn(train_worker_v3, args=(WORLD_SIZE, port, cfg, x_train, y_train, x_val, y_val),
                 nprocs=WORLD_SIZE, join=True)

    ft = evaluate_finetuned(model_path, x_test, y_test, tag="lits_tumor_v3_caselevel")

    print("\n" + "=" * 70)
    print("LiTS SAM3 V3 — CASE-LEVEL (patient-held-out) vs SAM1")
    print("=" * 70)
    print(f"{'Config':<45s} {'Tumor Dice':>12s}")
    print("-" * 58)
    print(f"{'SAM1 case-level (reference)':<45s} {'0.845':>12s}")
    print(f"{'SAM3 v3 partial_freeze (slice-level)':<45s} {'0.910':>12s}")
    print(f"{'SAM3 v3 partial_freeze (case-level)':<45s} {ft['dice'][0]:>12.4f}")
    print("=" * 70)
    print("DONE")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--v3", action="store_true", help="Run v3 on the surviving slice-level split")
    parser.add_argument("--caselevel", action="store_true", help="Run v3 case-level from restored raw volumes")
    args = parser.parse_args()
    if args.caselevel:
        run_v3_caselevel()
    else:
        run_v3()
