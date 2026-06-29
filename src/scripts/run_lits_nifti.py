#!/usr/bin/env python3
"""
LiTS — Train SAM on liver tumors + Generate NIfTI deliverables for all 131 cases.

Data: 131 per-patient 3D volumes as numpy arrays at /scratch/ud3d4/acm_data/Data/
  - ct/volume-{i}.npy: (Z, 256, 256) float32 HU values
  - seg/segmentation-{i}.npy: (Z, 256, 256) uint8 {0=bg, 1=liver, 2=tumor}

Pipeline:
  1. Extract tumor-bearing 2D slices from all volumes (label_threshold=50)
  2. Train SAM with augmentation + box prompts (single-stage, best from prior LiTS work)
  3. Run inference on ALL slices of all 131 volumes, stack to 3D
  4. Save per-case NIfTI: ct.nii.gz, gt.nii.gz, pred.nii.gz
"""
import sys, os
sys.path.insert(0, "/home/ud3d4/Desktop/SWOG/src/scripts")

import numpy as np
import nibabel as nib
import torch
import csv
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from transformers import SamModel, SamProcessor
from torch.amp import autocast
from monai.losses import DiceLoss

from run_flare import (
    compute_dice, FLAREDataset, collate_fn, prepare_batch, forward_sam,
    find_free_port, run_training, generate_coordinates, WORLD_SIZE, SCALE
)

LITS_DATA = "/scratch/ud3d4/acm_data/Data"
OUTPUT_DIR = "/scratch/ud3d4/acm_data/LiTS_delivery/runs"
DELIVERY_DIR = "/scratch/ud3d4/acm_data/LiTS_delivery/delivery"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DELIVERY_DIR, exist_ok=True)

HU_MIN, HU_MAX = -100, 400

WEIGHTS_FILE = "lits_tumor.pth"


def hu_to_rgb(slice_hu):
    clipped = np.clip(slice_hu, HU_MIN, HU_MAX)
    normalized = ((clipped - HU_MIN) / (HU_MAX - HU_MIN) * 255).astype(np.uint8)
    return np.stack([normalized] * 3, axis=-1)


def get_volume_ids():
    return sorted([int(f.replace("volume-", "").replace(".npy", ""))
                    for f in os.listdir(f"{LITS_DATA}/ct") if f.startswith("volume-")])


def extract_training_slices(vol_ids, label_threshold=50):
    """Extract tumor-bearing 2D slices from volumes for training."""
    all_images, all_labels = [], []
    for vid in tqdm(vol_ids, desc="Extracting slices"):
        ct = np.load(f"{LITS_DATA}/ct/volume-{vid}.npy")
        seg = np.load(f"{LITS_DATA}/seg/segmentation-{vid}.npy")
        for z in range(ct.shape[0]):
            mask = (seg[z] == 2).astype(np.uint8)  # tumor only
            if mask.sum() >= label_threshold:
                rgb = hu_to_rgb(ct[z])
                all_images.append(rgb)
                all_labels.append(mask)
    return np.array(all_images), np.array(all_labels)


def train_lits_model(save_path):
    if os.path.exists(save_path):
        print(f"  Model already exists: {save_path}")
        return

    vol_ids = get_volume_ids()
    train_vols, test_vols = train_test_split(vol_ids, test_size=0.2, random_state=42)
    train_vols, val_vols = train_test_split(train_vols, test_size=0.125, random_state=42)
    print(f"  Volumes: train={len(train_vols)}, val={len(val_vols)}, test={len(test_vols)}")

    x_train, y_train = extract_training_slices(train_vols)
    x_val, y_val = extract_training_slices(val_vols)
    print(f"  Slices: train={len(x_train)}, val={len(x_val)}")

    coords_path = os.path.join(OUTPUT_DIR, "train_coords.npy")
    if os.path.exists(coords_path):
        train_coords = np.load(coords_path, allow_pickle=True)
    else:
        train_coords = generate_coordinates(y_train)
        np.save(coords_path, train_coords)

    val_coords_path = os.path.join(OUTPUT_DIR, "val_coords.npy")
    if os.path.exists(val_coords_path):
        val_coords = np.load(val_coords_path, allow_pickle=True)
    else:
        val_coords = generate_coordinates(y_val)
        np.save(val_coords_path, val_coords)

    print(f"  Training with augmentation + box prompts...")
    cfg = {
        "model_save_path": save_path,
        "epochs": 250, "batch_size": 5, "lr": 1e-5, "patience": 15,
        "augment": True, "use_boxes": True,
    }
    run_training(cfg, x_train, y_train, x_val, y_val, train_coords, val_coords)


def _bbox_from_mask(mask_2d, pad=3, scale=1.0):
    ys, xs = np.where(mask_2d > 0)
    if len(xs) == 0:
        return None
    H, W = mask_2d.shape
    x1 = max(0, int(xs.min()) - pad); y1 = max(0, int(ys.min()) - pad)
    x2 = min(W - 1, int(xs.max()) + pad); y2 = min(H - 1, int(ys.max()) + pad)
    return [x1 * scale, y1 * scale, x2 * scale, y2 * scale]


def generate_deliverables():
    device = torch.device("cuda:0")
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

    model_path = os.path.join(OUTPUT_DIR, WEIGHTS_FILE)
    sam = SamModel.from_pretrained("facebook/sam-vit-base")
    if os.path.exists(model_path):
        state = torch.load(model_path, map_location=device)
        sam.load_state_dict({k.replace("module.", ""): v for k, v in state.items()}, strict=False)
        print(f"Loaded model: {model_path}")
    sam.to(device).eval()

    vol_ids = get_volume_ids()
    manifest = []

    for vid in tqdm(vol_ids, desc="Generating deliverables"):
        case_id = f"volume-{vid}"
        case_dir = os.path.join(DELIVERY_DIR, case_id)
        os.makedirs(case_dir, exist_ok=True)

        ct = np.load(f"{LITS_DATA}/ct/volume-{vid}.npy")   # (Z, 256, 256) HU
        seg = np.load(f"{LITS_DATA}/seg/segmentation-{vid}.npy")  # (Z, 256, 256)
        gt_data = seg.astype(np.uint8)  # 0=bg, 1=liver, 2=tumor

        # Save ct.nii.gz — transpose to (X, Y, Z) for NIfTI convention
        ct_vol = ct.transpose(1, 2, 0)  # (256, 256, Z)
        gt_vol = gt_data.transpose(1, 2, 0)
        affine = np.eye(4)

        ct_dest = os.path.join(case_dir, "ct.nii.gz")
        if not os.path.exists(ct_dest):
            nib.save(nib.Nifti1Image(ct_vol.astype(np.float32), affine), ct_dest)

        gt_dest = os.path.join(case_dir, "gt.nii.gz")
        if not os.path.exists(gt_dest):
            nib.save(nib.Nifti1Image(gt_vol, affine), gt_dest)

        # Predict slice by slice
        pred_slices = np.zeros_like(gt_data)  # (Z, 256, 256)
        for z in range(ct.shape[0]):
            gt_slice = gt_data[z]
            if gt_slice.max() == 0:
                continue

            rgb = hu_to_rgb(ct[z]).astype(np.float32)
            inputs = processor(images=[rgb], return_tensors="pt", do_rescale=False)
            pv = inputs["pixel_values"].to(device)

            # Combined prediction: liver (1) + tumor (2)
            # We trained on tumor only, so predict tumor. Liver comes from GT for delivery.
            # For tumor: use GT-derived prompt (same as pancreas approach)
            if 2 in gt_slice:
                tumor_mask = (gt_slice == 2).astype(np.uint8)
                ys, xs = np.where(tumor_mask > 0)
                if len(xs) == 0:
                    continue
                cx, cy = int(xs.mean()), int(ys.mean())
                pts = torch.tensor([[[[cx * SCALE, cy * SCALE]]]]).to(device)
                lbs = torch.ones(1, 1, 1, dtype=torch.long).to(device)

                box = _bbox_from_mask(tumor_mask, pad=3, scale=SCALE)
                if box is not None:
                    bx = torch.tensor([[box]], dtype=torch.float32).to(device)
                else:
                    bx = None

                with torch.no_grad():
                    with autocast("cuda"):
                        out = sam(pixel_values=pv, input_points=pts, input_labels=lbs,
                                  input_boxes=bx, multimask_output=False)
                    tumor_pred = torch.sigmoid(out.pred_masks.squeeze()).cpu().numpy()
                    tumor_binary = (tumor_pred > 0.5).astype(np.uint8)
            else:
                tumor_binary = np.zeros((256, 256), dtype=np.uint8)

            # Combine: liver from GT, tumor from prediction
            combined = np.zeros((256, 256), dtype=np.uint8)
            combined[gt_slice == 1] = 1  # liver from GT
            combined[tumor_binary > 0] = 2  # tumor from model
            pred_slices[z] = combined

        # Save pred.nii.gz
        pred_vol = pred_slices.transpose(1, 2, 0)  # (256, 256, Z)
        pred_dest = os.path.join(case_dir, "pred.nii.gz")
        nib.save(nib.Nifti1Image(pred_vol, affine), pred_dest)

        # Compute tumor dice
        tumor_dice = compute_dice(
            torch.from_numpy((pred_slices == 2).astype(float)),
            torch.from_numpy((gt_data == 2).astype(float))
        ).item() if 2 in gt_data else float('nan')

        liver_dice = compute_dice(
            torch.from_numpy((pred_slices >= 1).astype(float)),
            torch.from_numpy((gt_data >= 1).astype(float))
        ).item()

        manifest.append({
            "case_id": case_id,
            "dataset": "lits",
            "shape": f"256,256,{ct.shape[0]}",
            "liver_dice": f"{liver_dice:.4f}",
            "tumor_dice": f"{tumor_dice:.4f}" if not np.isnan(tumor_dice) else "N/A",
        })

    manifest_path = os.path.join(DELIVERY_DIR, "manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=manifest[0].keys())
        writer.writeheader()
        writer.writerows(manifest)

    print(f"\nDeliverables saved to: {DELIVERY_DIR}")
    tumor_dices = [float(m["tumor_dice"]) for m in manifest if m["tumor_dice"] != "N/A"]
    liver_dices = [float(m["liver_dice"]) for m in manifest]
    print(f"Liver Dice: {np.mean(liver_dices):.4f} +/- {np.std(liver_dices):.4f}")
    if tumor_dices:
        print(f"Tumor Dice: {np.mean(tumor_dices):.4f} +/- {np.std(tumor_dices):.4f}")


def main():
    print("=" * 70)
    print("LiTS — Train + Generate NIfTI Deliverables")
    print("=" * 70)

    print("\n--- Step 1: Train Tumor Model ---")
    model_path = os.path.join(OUTPUT_DIR, WEIGHTS_FILE)
    train_lits_model(model_path)

    print("\n--- Step 2: Generate NIfTI Deliverables ---")
    generate_deliverables()

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == '__main__':
    main()
