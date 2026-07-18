#!/usr/bin/env python3
"""
FLARE SAM3 experiment + NIfTI mask delivery.

Applies the best SAM3 recipe (v3 partial-freeze) to the four FLARE structures
(liver, pancreas, tumor, duodenum) and exports ground-truth + predicted masks
as NIfTI stacks for sharing.

FLARE is distributed pre-sliced as per-class 2D arrays with NO patient IDs, so
only a SLICE-LEVEL split is possible (unavoidable leakage; same limitation as
the original AUSAM FLARE protocol). Deliverables are NIfTI stacks (unrelated
slices stacked into a pseudo-volume), not per-patient volumes.

Resumable: any structure whose checkpoint already exists is skipped, so a
session interruption only costs the in-flight structure.

Run:  HF_TOKEN=... python run_flare_sam3.py            # train all 4
      HF_TOKEN=... python run_flare_sam3.py --deliver   # export gt+pred NIfTI
"""
import sys, os, csv
sys.path.insert(0, "/home/ud3d4/Desktop/SWOG/src/scripts")

import numpy as np
import torch
import torch.multiprocessing as mp
import nibabel as nib
from sklearn.model_selection import train_test_split
from transformers import Sam3Processor

from run_pancreas_sam3 import (
    train_worker_v3, evaluate_finetuned,
    _sam3_infer_slice, _load_sam3_ckpt,
    SAM3_MODEL_ID, HF_TOKEN,
)
from run_flare import find_free_port, WORLD_SIZE

FLARE_DATA = "/scratch/ud3d4/acm_data/FLARE"
FLARE_SAM3_DIR = "/scratch/ud3d4/acm_data/FLARE/sam3"
FLARE_DELIVERY_DIR = "/scratch/ud3d4/acm_data/FLARE/sam3_delivery"
os.makedirs(FLARE_SAM3_DIR, exist_ok=True)

# class id -> structure name
# Paper's coverage-regime set: 5 organs + pan-cancer lesion. New organs first (train),
# then already-trained (skip+eval). Duodenum(12) is excluded (not in the paper's set).
STRUCTURES = [(2, "right_kidney"), (3, "spleen"), (13, "left_kidney"),
              (1, "liver"), (4, "pancreas"), (14, "tumor")]
MAX_TRAIN = 6000  # cap training slices per structure to keep epochs tractable


def load_class(c):
    im = np.load(f"{FLARE_DATA}/class_{c}_images.npy")
    lb = np.load(f"{FLARE_DATA}/class_{c}_labels.npy")
    return im.astype(np.uint8), (lb > 0).astype(np.uint8)


def split_class(im, lb):
    idx = np.arange(len(im))
    tr, te = train_test_split(idx, test_size=0.2, random_state=42)
    tr, va = train_test_split(tr, test_size=0.125, random_state=42)
    return tr, va, te


def v3_cfg(model_path):
    return {
        "model_save_path": model_path, "epochs": 45, "batch_size": 2, "patience": 12,
        "strong_augment": True, "text_prompt": "visual",
        "freeze_blocks": 20, "encoder_lr": 1e-5, "decoder_lr": 1e-4,
        "warmup_epochs": 5, "cosine_T0": 20, "dice_weight": 0.7, "focal_weight": 0.3,
    }


def run_all():
    results = {}
    for c, name in STRUCTURES:
        print(f"\n{'='*70}\nFLARE SAM3 v3 — {name} (class {c})\n{'='*70}", flush=True)
        im, lb = load_class(c)
        tr, va, te = split_class(im, lb)
        x_train, y_train = im[tr], lb[tr]
        x_val, y_val = im[va], lb[va]
        x_test, y_test = im[te], lb[te]
        if len(x_train) > MAX_TRAIN:
            sel = np.random.RandomState(42).choice(len(x_train), MAX_TRAIN, replace=False)
            x_train, y_train = x_train[sel], y_train[sel]
        print(f"Slices: train={len(x_train)} (of {len(tr)}), val={len(x_val)}, test={len(x_test)}", flush=True)

        model_path = os.path.join(FLARE_SAM3_DIR, f"flare_sam3_v3_{name}.pth")
        if not os.path.exists(model_path):
            port = find_free_port()
            mp.spawn(train_worker_v3, args=(WORLD_SIZE, port, v3_cfg(model_path),
                     x_train, y_train, x_val, y_val), nprocs=WORLD_SIZE, join=True)
        else:
            print(f"  checkpoint exists, skipping training: {model_path}", flush=True)

        ft = evaluate_finetuned(model_path, x_test, y_test, tag=f"flare_{name}_v3")
        results[name] = ft["dice"][0]

    print("\n" + "=" * 70)
    print("FLARE SAM3 V3 (SLICE-LEVEL — no patient IDs) vs original AUSAM")
    print("=" * 70)
    print(f"{'Structure':<15s} {'SAM3 v3':>10s} {'Orig AUSAM':>12s}")
    print("-" * 40)
    orig = {"liver": 0.965, "pancreas": 0.839, "tumor": 0.855, "duodenum": 0.890}
    for _, name in STRUCTURES:
        ob = orig.get(name)
        print(f"{name:<15s} {results.get(name, float('nan')):>10.4f} {(f'{ob:.3f}' if ob else '—'):>12s}")
    print("=" * 70)
    print("DONE")


def deliver_worker(rank, world_size):
    """Export gt + pred + ct NIfTI stacks for a shard of structures (test slices)."""
    device = torch.device(f"cuda:{rank}")
    processor = Sam3Processor.from_pretrained(SAM3_MODEL_ID, token=HF_TOKEN)
    shard = STRUCTURES[rank::world_size]
    affine = np.eye(4)

    for c, name in shard:
        ckpt = os.path.join(FLARE_SAM3_DIR, f"flare_sam3_v3_{name}.pth")
        if not os.path.exists(ckpt):
            print(f"  [GPU{rank}] SKIP {name}: no checkpoint", flush=True)
            continue
        model = _load_sam3_ckpt(ckpt, device)
        im, lb = load_class(c)
        _, _, te = split_class(im, lb)
        x_test, y_test = im[te], lb[te]

        out_dir = os.path.join(FLARE_DELIVERY_DIR, name)
        os.makedirs(out_dir, exist_ok=True)
        N = len(x_test)
        ct_stack = np.zeros((256, 256, N), dtype=np.float32)
        gt_stack = np.zeros((256, 256, N), dtype=np.uint8)
        pred_stack = np.zeros((256, 256, N), dtype=np.uint8)
        dices = []
        for i in range(N):
            rgb = x_test[i].astype(np.uint8)
            gt = y_test[i].astype(np.uint8)
            ct_stack[:, :, i] = rgb[..., 0]
            gt_stack[:, :, i] = gt
            if gt.sum() > 0:
                prob = _sam3_infer_slice(model, processor, rgb, gt, device)
                pred = (prob > 0.5).astype(np.uint8) if prob is not None else np.zeros_like(gt)
            else:
                pred = np.zeros_like(gt)
            pred_stack[:, :, i] = pred
            inter = (pred & gt).sum()
            dices.append((2 * inter + 1e-6) / (pred.sum() + gt.sum() + 1e-6))
            if rank == 0 and (i + 1) % 200 == 0:
                print(f"  [GPU{rank}] {name}: {i+1}/{N}", flush=True)

        nib.save(nib.Nifti1Image(ct_stack, affine), os.path.join(out_dir, "ct.nii.gz"))
        nib.save(nib.Nifti1Image(gt_stack, affine), os.path.join(out_dir, "gt.nii.gz"))
        nib.save(nib.Nifti1Image(pred_stack, affine), os.path.join(out_dir, "pred.nii.gz"))
        with open(os.path.join(out_dir, "info.txt"), "w") as f:
            f.write(f"structure={name} class={c} test_slices={N} "
                    f"mean_slice_dice={np.mean(dices):.4f}\n")
        print(f"  [GPU{rank}] {name}: saved {N} slices, mean Dice {np.mean(dices):.4f} -> {out_dir}", flush=True)


def deliver():
    os.makedirs(FLARE_DELIVERY_DIR, exist_ok=True)
    print(f"Exporting FLARE gt+pred NIfTI stacks across {WORLD_SIZE} GPUs -> {FLARE_DELIVERY_DIR}", flush=True)
    mp.spawn(deliver_worker, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)
    print("\n" + "=" * 60)
    print(f"FLARE SAM3 mask deliverables -> {FLARE_DELIVERY_DIR}")
    for _, name in STRUCTURES:
        info = os.path.join(FLARE_DELIVERY_DIR, name, "info.txt")
        if os.path.exists(info):
            print("  " + open(info).read().strip())
    print("=" * 60)
    print("DONE")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--deliver", action="store_true", help="Export gt+pred NIfTI stacks")
    args = parser.parse_args()
    if args.deliver:
        deliver()
    else:
        run_all()
