#!/usr/bin/env python3
"""
SAM3 v3 on FLARE-Task2 (2024) PER-PATIENT volumes — patient-level split, true 3D Dice.

Supersedes the pre-sliced per-class FLARE run: this uses intact per-patient CT volumes with
dense FLARE22 organ labels, so the split is at the PATIENT level (no slice leakage) and Dice
is computed volumetrically per patient. Five organs (liver, right/left kidney, spleen,
pancreas); FLARE22 has no tumour label (tumour work stays on Pancreas/LiTS).

Data: /scratch/ud3d4/acm_data/FLARE_Task2  (50 train_gt + 50 public-val = 100 volumes),
pooled and split 70/10/20 at the patient level (seed 42).

Run:  HF_TOKEN=... python run_flare_task2_sam3.py            # train all 5
      HF_TOKEN=... python run_flare_task2_sam3.py --deliver  # export per-patient ct/gt/pred NIfTI
Resumable: an organ whose checkpoint exists is skipped (eval still runs).
"""
import sys, os, glob
sys.path.insert(0, "/home/ud3d4/Desktop/SWOG/src/scripts")
import numpy as np
import torch
import torch.multiprocessing as mp
import nibabel as nib
from sklearn.model_selection import train_test_split
from transformers import Sam3Processor

from run_pancreas_sam3 import (
    train_worker_v3, _sam3_infer_slice, _load_sam3_ckpt,
    SAM3_MODEL_ID, HF_TOKEN,
)
from run_flare import find_free_port, WORLD_SIZE

DATA = "/scratch/ud3d4/acm_data/FLARE_Task2"
CKPT_DIR = "/scratch/ud3d4/acm_data/FLARE_Task2/sam3"
DELIVERY_DIR = "/scratch/ud3d4/acm_data/FLARE_Task2/sam3_delivery"
os.makedirs(CKPT_DIR, exist_ok=True)

ORGANS = [(1, "liver"), (2, "right_kidney"), (3, "spleen"), (4, "pancreas"), (13, "left_kidney")]
WINDOW = (-125, 225)          # abdominal soft-tissue HU window
MIN_ORGAN_PX = 20             # skip slices with a negligible organ footprint
MAX_TRAIN = 5000              # cap training slices/organ to keep epochs tractable


def list_pairs():
    """(image_path, label_path) for all 100 volumes, from both GT sources."""
    pairs = []
    for img_dir, lbl_dir in [("train_gt_label/imagesTr", "train_gt_label/labelsTr"),
                             ("validation/Validation-Public-Images", "validation/Validation-Public-Labels")]:
        for ip in sorted(glob.glob(f"{DATA}/{img_dir}/*.nii.gz")):
            base = os.path.basename(ip).replace("_0000.nii.gz", ".nii.gz")
            lp = f"{DATA}/{lbl_dir}/{base}"
            if os.path.exists(lp) and not os.path.basename(ip).startswith("._"):
                pairs.append((ip, lp))
    return pairs


def split_pairs():
    pairs = list_pairs()
    idx = np.arange(len(pairs))
    tr, te = train_test_split(idx, test_size=0.2, random_state=42)
    tr, va = train_test_split(tr, test_size=0.125, random_state=42)   # 0.125*0.8 = 0.10
    P = lambda ii: [pairs[i] for i in ii]
    return P(tr), P(va), P(te)


def window_rgb(sl):
    lo, hi = WINDOW
    x = np.clip(sl, lo, hi)
    x = ((x - lo) / (hi - lo) * 255.0).astype(np.uint8)
    return np.repeat(x[..., None], 3, axis=2)          # HxWx3


def _resize(a, order):
    from skimage.transform import resize
    return resize(a, (256, 256), order=order, preserve_range=True, anti_aliasing=(order > 0))


def build_organ_slices(pairs, organ_id, cap=None, seed=42):
    """Return X (N,256,256,3 uint8), Y (N,256,256 uint8), meta [(case,z)] for one organ."""
    X, Y, meta = [], [], []
    for ip, lp in pairs:
        vol = nib.load(ip).get_fdata()
        lab = nib.load(lp).get_fdata().astype(np.uint8)
        m = (lab == organ_id)
        zs = np.where(m.reshape(-1, m.shape[2]).sum(0) >= MIN_ORGAN_PX)[0]
        cid = os.path.basename(ip).replace("_0000.nii.gz", "")
        for z in zs:
            X.append(window_rgb(_resize(vol[:, :, z], 1)))
            Y.append((_resize(m[:, :, z].astype(np.uint8), 0) > 0.5).astype(np.uint8))
            meta.append((cid, int(z)))
    X = np.asarray(X, dtype=np.uint8); Y = np.asarray(Y, dtype=np.uint8)
    if cap and len(X) > cap:
        sel = np.random.RandomState(seed).choice(len(X), cap, replace=False)
        X, Y = X[sel], Y[sel]
    return X, Y, meta


def v3_cfg(model_path):
    # epochs=32 is a safety ceiling; early stopping (patience 6) decides. cosine_T0=60 >> epochs
    # makes the schedule a plain monotonic decay (NO warm restart), so the LR jump can't reset
    # the patience counter -> early stop actually triggers on plateau.
    return {"model_save_path": model_path, "epochs": 32, "batch_size": 2, "patience": 6,
            "strong_augment": True, "text_prompt": "visual", "freeze_blocks": 20,
            "encoder_lr": 1e-5, "decoder_lr": 1e-4, "warmup_epochs": 5, "cosine_T0": 60,
            "dice_weight": 0.7, "focal_weight": 0.3}


def dice3d(pred_vol, gt_vol):
    i = float((pred_vol & gt_vol).sum()); s = float(pred_vol.sum() + gt_vol.sum())
    return (2 * i / s) if s else float("nan")


def eval_3d(model_path, test_pairs, organ_id, name):
    """Volumetric Dice per patient (GT-box prompted, semi-oracle), mean over test volumes."""
    device = torch.device("cuda:0")
    processor = Sam3Processor.from_pretrained(SAM3_MODEL_ID, token=HF_TOKEN)
    model = _load_sam3_ckpt(model_path, device)
    dices = []
    for ip, lp in test_pairs:
        vol = nib.load(ip).get_fdata()
        lab = nib.load(lp).get_fdata().astype(np.uint8)
        gt = (lab == organ_id)
        if gt.sum() == 0:
            continue
        pred = np.zeros(gt.shape, dtype=np.uint8)
        for z in range(gt.shape[2]):
            gz = gt[:, :, z]
            if gz.sum() < MIN_ORGAN_PX:
                continue
            rgb = window_rgb(_resize(vol[:, :, z], 1)).astype(np.uint8)
            m256 = (_resize(gz.astype(np.uint8), 0) > 0.5).astype(np.uint8)
            prob = _sam3_infer_slice(model, processor, rgb, m256, device)
            if prob is None:
                continue
            H, W = gz.shape
            pred[:, :, z] = (_resize_to(prob, (H, W)) > 0.5).astype(np.uint8)
        dices.append(dice3d(pred.astype(bool), gt))
    return float(np.nanmean(dices)) if dices else float("nan"), len(dices)


def _resize_to(a, shape):
    from skimage.transform import resize
    return resize(a, shape, order=1, preserve_range=True)


def run_all():
    tr, va, te = split_pairs()
    print(f"Patient-level split: train={len(tr)} val={len(va)} test={len(te)} volumes", flush=True)
    results = {}
    for organ_id, name in ORGANS:
        print(f"\n{'='*70}\nFLARE-Task2 SAM3 v3 — {name} (label {organ_id})\n{'='*70}", flush=True)
        model_path = os.path.join(CKPT_DIR, f"flare_t2_sam3_v3_{name}.pth")
        if not os.path.exists(model_path):
            xtr, ytr, _ = build_organ_slices(tr, organ_id, cap=MAX_TRAIN)
            xva, yva, _ = build_organ_slices(va, organ_id)
            print(f"  slices: train={len(xtr)} val={len(xva)}", flush=True)
            port = find_free_port()
            mp.spawn(train_worker_v3, args=(WORLD_SIZE, port, v3_cfg(model_path),
                     xtr, ytr, xva, yva), nprocs=WORLD_SIZE, join=True)
        else:
            print(f"  checkpoint exists, skipping training: {model_path}", flush=True)
        d3, nvol = eval_3d(model_path, te, organ_id, name)
        results[name] = d3
        print(f"  >> {name}: 3D Dice = {d3:.4f} over {nvol} test volumes", flush=True)
        # completion marker: lets the requeuing resume job tell finished organs from
        # interrupted (partial-checkpoint) ones -> incomplete organs retrain cleanly.
        if d3 == d3:  # not NaN
            open(os.path.join(CKPT_DIR, f"{name}.done"), "w").close()

    print("\n" + "=" * 60)
    print("FLARE-Task2 (2024) SAM3 v3 — PER-PATIENT 3D Dice")
    print("=" * 60)
    for _, name in ORGANS:
        print(f"  {name:<14s} {results.get(name, float('nan')):.4f}")
    print("=" * 60 + "\nDONE", flush=True)


def deliver_worker(rank, world_size):
    device = torch.device(f"cuda:{rank}")
    processor = Sam3Processor.from_pretrained(SAM3_MODEL_ID, token=HF_TOKEN)
    _, _, te = split_pairs()
    shard = ORGANS[rank::world_size]
    for organ_id, name in shard:
        ckpt = os.path.join(CKPT_DIR, f"flare_t2_sam3_v3_{name}.pth")
        if not os.path.exists(ckpt):
            print(f"  [GPU{rank}] SKIP {name}: no checkpoint", flush=True); continue
        model = _load_sam3_ckpt(ckpt, device)
        out_dir = os.path.join(DELIVERY_DIR, name); os.makedirs(out_dir, exist_ok=True)
        for ip, lp in te:
            ni = nib.load(ip); vol = ni.get_fdata()
            lab = nib.load(lp).get_fdata().astype(np.uint8)
            gt = (lab == organ_id).astype(np.uint8)
            pred = np.zeros(gt.shape, dtype=np.uint8)
            for z in range(gt.shape[2]):
                if gt[:, :, z].sum() < MIN_ORGAN_PX:
                    continue
                rgb = window_rgb(_resize(vol[:, :, z], 1)).astype(np.uint8)
                m256 = (_resize(gt[:, :, z], 0) > 0.5).astype(np.uint8)
                prob = _sam3_infer_slice(model, processor, rgb, m256, device)
                if prob is not None:
                    H, W = gt[:, :, z].shape
                    pred[:, :, z] = (_resize_to(prob, (H, W)) > 0.5).astype(np.uint8)
            cid = os.path.basename(ip).replace("_0000.nii.gz", "")
            nib.save(nib.Nifti1Image(vol.astype(np.float32), ni.affine), f"{out_dir}/{cid}_ct.nii.gz")
            nib.save(nib.Nifti1Image(gt, ni.affine), f"{out_dir}/{cid}_gt.nii.gz")
            nib.save(nib.Nifti1Image(pred, ni.affine), f"{out_dir}/{cid}_pred.nii.gz")
        print(f"  [GPU{rank}] {name}: delivered {len(te)} volumes -> {out_dir}", flush=True)


def deliver():
    os.makedirs(DELIVERY_DIR, exist_ok=True)
    mp.spawn(deliver_worker, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--deliver", action="store_true")
    a = ap.parse_args()
    deliver() if a.deliver else run_all()
