#!/usr/bin/env python3
"""Clean LiTS-ONLY tumor model (no KG) — the 'original single-dataset paradigm' comparator for the
two-model paradigm figures. Same SAM3 backbone, same recipe, same patient-level split (seed 42) as the
pooled model's s1 stage, so the only variable vs sam3_tumor_generic.pth is the training-data breadth
(one dataset vs pooled). Box-free text='tumor'. -> tumor_pool/sam3_tumor_s1_lits_patientlevel.pth
"""
import os
import sys

import torch.multiprocessing as mp

sys.path.insert(0, "/home/ud3d4/Desktop/SWOG/src/scripts")
from train_tumor_incremental import load_all, patient_split, CKPT_DIR
from run_pancreas_sam3 import train_worker_v3, WORLD_SIZE, find_free_port

CKPT = f"{CKPT_DIR}/sam3_tumor_s1_lits_patientlevel.pth"   # NON-kg, matches the headline 34.8/0.004 curve


def main():
    X, Y, M = load_all()
    tr, va, te = patient_split(M)
    tr_s = [i for i in tr if M[i]["dataset"] == "lits"]
    va_s = [i for i in va if M[i]["dataset"] == "lits"]
    print(f"LiTS-only tumor (original single-dataset paradigm): tr/va = {len(tr_s)}/{len(va_s)}", flush=True)
    if os.path.exists(CKPT):
        print(f"exists: {CKPT}", flush=True); return
    cfg = {"model_save_path": CKPT, "pretrained_path": None, "epochs": 6, "batch_size": 4,
           "patience": 2, "strong_augment": True, "text_prompt": "tumor", "use_boxes": False,
           "freeze_blocks": 20, "encoder_lr": 1e-5, "decoder_lr": 1e-4, "warmup_epochs": 2,
           "cosine_T0": 40, "dice_weight": 0.7, "focal_weight": 0.3}
    port = find_free_port()
    mp.spawn(train_worker_v3, args=(WORLD_SIZE, port, cfg, X[tr_s], Y[tr_s], X[va_s], Y[va_s]),
             nprocs=WORLD_SIZE, join=True)
    print(f"-> {CKPT}", flush=True)


if __name__ == "__main__":
    main()
