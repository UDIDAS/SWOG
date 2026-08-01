#!/usr/bin/env python3
"""Decisive first experiment for the autonomous end-goal: on a held-out CT (FLARE23_0217, liver),
compare prompt modes for the SAM3 liver model —
  gtbox  : GT-derived box (SEMI-ORACLE ceiling)
  fullbox: full-image box (autonomous localization within slice)
  text   : SAM3 concept prompt 'liver', NO box (fully autonomous) — fine-tuned AND base SAM3
Dice is volume-level over the liver-containing slices, so all modes are compared on the same slices."""
import os
import sys

import nibabel as nib
import numpy as np
import torch
from skimage.transform import resize
from torch.amp import autocast

sys.path.insert(0, "/home/ud3d4/Desktop/SWOG/src/scripts")
import infer_sam3 as I

DEV = "cuda"
WIN = (-125, 225)
CT = "data/FLARE23_0217_0000.nii.gz"
GT = "data/FLARE23_0217.nii.gz"
LAB = 1  # liver
CKPT = "/scratch/ud3d4/acm_data/FLARE_Task2/sam3/flare_t2_sam3_v3_liver.pth"
os.chdir("/home/ud3d4/Desktop/SWOG")


def load_base():
    from transformers import Sam3Model, Sam3Processor
    proc = Sam3Processor.from_pretrained(I.SAM3_BASE, token=I.HF_TOKEN, revision=I.SAM3_REVISION)
    model = Sam3Model.from_pretrained(I.SAM3_BASE, token=I.HF_TOKEN, revision=I.SAM3_REVISION)
    return model.to(DEV).eval(), proc


def infer_text(model, proc, rgb, concept):
    inputs = proc(images=[rgb], text=[concept], return_tensors="pt")
    kw = {"pixel_values": inputs["pixel_values"].to(DEV)}
    for k in ("input_ids", "attention_mask"):
        if inputs.get(k) is not None:
            kw[k] = inputs[k].to(DEV)
    with torch.no_grad(), autocast("cuda"):
        out = model(**kw)
        m = torch.nn.functional.interpolate(I.best_mask(out).float(), size=(256, 256),
                                            mode="bilinear", align_corners=False)
    return m.sigmoid().squeeze().cpu().numpy()


ct = nib.load(CT).get_fdata()
gt = nib.load(GT).get_fdata().astype(np.uint8)
X, Y, Z = ct.shape
sl_idx = [z for z in range(Z) if (gt[:, :, z] == LAB).sum() >= 20]
print(f"CT {ct.shape}  liver slices={len(sl_idx)}  liver vox(GT)={(gt==LAB).sum()}", flush=True)


def run(mode, model, proc):
    pred = np.zeros(ct.shape, np.uint8)
    for z in sl_idx:
        rgb = I.hu_to_rgb(resize(ct[:, :, z], (256, 256), preserve_range=True, anti_aliasing=True),
                          *WIN).astype(np.uint8)
        if mode == "gtbox":
            g256 = resize((gt[:, :, z] == LAB).astype(np.uint8), (256, 256), order=0,
                          preserve_range=True).astype(np.uint8)
            box = I.bbox_from_mask(g256, 3)
            if box is None:
                continue
            prob = I.infer_slice(model, proc, rgb, box, DEV)
        elif mode == "fullbox":
            prob = I.infer_slice(model, proc, rgb, [0, 0, 255, 255], DEV)
        else:
            prob = infer_text(model, proc, rgb, "liver")
        m = (resize(prob, (X, Y), order=1, preserve_range=True) > 0.5).astype(np.uint8)
        pred[:, :, z][m > 0] = 1
    gm = (gt == LAB)
    s = int((pred > 0).sum()) + int(gm.sum())
    d = 2 * int(((pred > 0) & gm).sum()) / s if s else 0.0
    return d, int((pred > 0).sum()), int(gm.sum())


ft, proc = I.load_model(CKPT, DEV)
for mode in ["gtbox", "fullbox", "text"]:
    d, pv, gv = run(mode, ft, proc)
    print(f"FT   {mode:7s}  Dice={d:.3f}  pred_vox={pv}  gt_vox={gv}", flush=True)
del ft
torch.cuda.empty_cache()

base, bproc = load_base()
d, pv, gv = run("text", base, bproc)
print(f"BASE text     Dice={d:.3f}  pred_vox={pv}  gt_vox={gv}", flush=True)
print("DONE", flush=True)
