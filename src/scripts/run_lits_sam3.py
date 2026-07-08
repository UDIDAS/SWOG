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

import nibabel as nib

# Reuse all SAM3 infrastructure from the Pancreas experiment
from run_pancreas_sam3 import (
    train_worker_v3, evaluate_finetuned,
    _sam3_infer_slice, _load_sam3_ckpt, bbox_from_mask,
    SAM3_MODEL_ID, HF_TOKEN,
)
from run_flare import find_free_port, WORLD_SIZE, compute_dice
from transformers import Sam3Processor
import torch
import csv

LITS_DELIVERY_DIR = "/scratch/ud3d4/acm_data/LiTS/sam3_delivery"

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


def deliver_worker_lits(rank, world_size):
    """Generate per-case NIfTI (liver from GT + SAM3 tumor pred) for a shard of LiTS volumes."""
    device = torch.device(f"cuda:{rank}")
    processor = Sam3Processor.from_pretrained(SAM3_MODEL_ID, token=HF_TOKEN)
    tumor_model = _load_sam3_ckpt(os.path.join(LITS_SAM3_DIR, "lits_sam3_v3_caselevel_tumor.pth"), device)
    if rank == 0:
        print(f"  [GPU{rank}] loaded SAM3 tumor model", flush=True)

    vol_ids = get_volume_ids()
    shard = vol_ids[rank::world_size]
    affine = np.eye(4)
    manifest = []

    for ci, vid in enumerate(shard):
        case_id = f"volume-{vid}"
        case_dir = os.path.join(LITS_DELIVERY_DIR, case_id)
        os.makedirs(case_dir, exist_ok=True)

        ct = np.load(f"{LITS_RAW}/ct/volume-{vid}.npy")            # (Z,256,256) HU
        seg = np.load(f"{LITS_RAW}/seg/segmentation-{vid}.npy").astype(np.uint8)  # 0/1/2
        Z = ct.shape[0]

        ct_vol = ct.transpose(1, 2, 0).astype(np.float32)         # (256,256,Z)
        gt_vol = seg.transpose(1, 2, 0)
        nib.save(nib.Nifti1Image(ct_vol, affine), os.path.join(case_dir, "ct.nii.gz"))
        nib.save(nib.Nifti1Image(gt_vol, affine), os.path.join(case_dir, "gt.nii.gz"))

        pred_slices = np.zeros_like(seg)                          # (Z,256,256)
        for z in range(Z):
            gt_slice = seg[z]
            combined = np.zeros((256, 256), dtype=np.uint8)
            combined[gt_slice == 1] = 1                           # liver from GT
            tumor256 = (gt_slice == 2).astype(np.uint8)
            if tumor256.sum() >= 50:
                rgb = lits_hu_to_rgb(ct[z]).astype(np.uint8)
                prob = _sam3_infer_slice(tumor_model, processor, rgb, tumor256, device)
                if prob is not None:
                    combined[(prob > 0.5)] = 2                     # tumor from SAM3 pred
            pred_slices[z] = combined

        nib.save(nib.Nifti1Image(pred_slices.transpose(1, 2, 0), affine),
                 os.path.join(case_dir, "pred.nii.gz"))

        tumor_dice = compute_dice(torch.from_numpy((pred_slices == 2).astype(float)),
                                  torch.from_numpy((seg == 2).astype(float))).item() if (seg == 2).any() else float("nan")
        liver_dice = compute_dice(torch.from_numpy((pred_slices >= 1).astype(float)),
                                  torch.from_numpy((seg >= 1).astype(float))).item()
        manifest.append({"case_id": case_id, "dataset": "lits", "shape": f"256,256,{Z}",
                         "liver_dice": f"{liver_dice:.4f}",
                         "tumor_dice": f"{tumor_dice:.4f}" if not np.isnan(tumor_dice) else "N/A"})
        if rank == 0:
            print(f"  [GPU{rank}] {ci+1}/{len(shard)} {case_id} tumor={manifest[-1]['tumor_dice']}", flush=True)

    part = os.path.join(LITS_DELIVERY_DIR, f"manifest_rank{rank}.csv")
    with open(part, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0].keys())); w.writeheader(); w.writerows(manifest)
    print(f"  [GPU{rank}] done: {len(manifest)} cases -> {part}", flush=True)


def generate_lits_deliverables():
    """Multi-GPU: split 131 LiTS volumes across GPUs, generate SAM3 NIfTI deliverables."""
    os.makedirs(LITS_DELIVERY_DIR, exist_ok=True)
    ckpt = os.path.join(LITS_SAM3_DIR, "lits_sam3_v3_caselevel_tumor.pth")
    assert os.path.exists(ckpt), f"Missing checkpoint: {ckpt}"
    print(f"Generating LiTS SAM3 deliverables across {WORLD_SIZE} GPUs -> {LITS_DELIVERY_DIR}", flush=True)
    mp.spawn(deliver_worker_lits, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)

    rows = []
    for rank in range(WORLD_SIZE):
        part = os.path.join(LITS_DELIVERY_DIR, f"manifest_rank{rank}.csv")
        if os.path.exists(part):
            rows.extend(list(csv.DictReader(open(part))))
    rows.sort(key=lambda r: int(r["case_id"].replace("volume-", "")))
    with open(os.path.join(LITS_DELIVERY_DIR, "manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    liver = [float(r["liver_dice"]) for r in rows]
    tumor = [float(r["tumor_dice"]) for r in rows if r["tumor_dice"] != "N/A"]
    print("\n" + "=" * 60)
    print(f"LiTS SAM3 DELIVERABLES: {len(rows)} cases -> {LITS_DELIVERY_DIR}")
    print(f"Liver Dice (from GT): {np.mean(liver):.4f} +/- {np.std(liver):.4f}")
    print(f"Tumor Dice (SAM3 v3): {np.mean(tumor):.4f} +/- {np.std(tumor):.4f}")
    print("=" * 60)
    print("DONE")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--v3", action="store_true", help="Run v3 on the surviving slice-level split")
    parser.add_argument("--caselevel", action="store_true", help="Run v3 case-level from restored raw volumes")
    parser.add_argument("--deliver", action="store_true", help="Generate NIfTI deliverables (multi-GPU)")
    args = parser.parse_args()
    if args.deliver:
        generate_lits_deliverables()
    elif args.caselevel:
        run_v3_caselevel()
    else:
        run_v3()
