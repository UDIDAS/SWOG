#!/usr/bin/env python3
"""Extend the prompt bake-off to the HARD structures: pancreas organ + pancreas tumor.
Compare gtbox (semi-oracle ceiling) vs SAM3 concept text prompts (autonomous, no box).
Dice over the label-containing slices (fair mask-quality comparison)."""
import os
import sys

import nibabel as nib
import numpy as np
import torch
from skimage.transform import resize
from torch.amp import autocast

sys.path.insert(0, "/home/ud3d4/Desktop/SWOG/src/scripts")
import infer_sam3 as I

os.chdir("/home/ud3d4/Desktop/SWOG")
DEV = "cuda"
CKDIR = "/scratch/ud3d4/acm_data/Pancreas/sam3"
P_CT = "/scratch/ud3d4/acm_data/Pancreas/imagesTr/pancreas_001.nii.gz"
P_GT = "/scratch/ud3d4/acm_data/Pancreas/labelsTr/pancreas_001.nii.gz"

# (tag, ct, gt, label, checkpoint, HU window, [concepts to try])
CASES = [
    ("pancreas_organ", P_CT, P_GT, 1, f"{CKDIR}/sam3_v3_organ.pth", (-100, 300), ["pancreas"]),
    ("pancreas_tumor", P_CT, P_GT, 2, f"{CKDIR}/sam3_v3_tumor.pth", (-100, 300),
     ["tumor", "pancreatic tumor", "mass", "lesion"]),
]


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


def run(ct, gt, lab, model, proc, win, mode, concept=None):
    X, Y, Z = ct.shape
    idx = [z for z in range(Z) if (gt[:, :, z] == lab).sum() >= 15]
    pred = np.zeros(ct.shape, np.uint8)
    for z in idx:
        rgb = I.hu_to_rgb(resize(ct[:, :, z], (256, 256), preserve_range=True, anti_aliasing=True),
                          *win).astype(np.uint8)
        if mode == "gtbox":
            g256 = resize((gt[:, :, z] == lab).astype(np.uint8), (256, 256), order=0,
                          preserve_range=True).astype(np.uint8)
            box = I.bbox_from_mask(g256, 3)
            if box is None:
                continue
            prob = I.infer_slice(model, proc, rgb, box, DEV)
        else:
            prob = infer_text(model, proc, rgb, concept)
        m = (resize(prob, (X, Y), order=1, preserve_range=True) > 0.5).astype(np.uint8)
        pred[:, :, z][m > 0] = 1
    gm = (gt == lab)
    s = int((pred > 0).sum()) + int(gm.sum())
    d = 2 * int(((pred > 0) & gm).sum()) / s if s else 0.0
    return d, int((pred > 0).sum()), int(gm.sum()), len(idx)


for tag, ctp, gtp, lab, ckpt, win, concepts in CASES:
    ct = nib.load(ctp).get_fdata()
    gt = nib.load(gtp).get_fdata().astype(np.uint8)
    model, proc = I.load_model(ckpt, DEV)
    d, pv, gv, ns = run(ct, gt, lab, model, proc, win, "gtbox")
    print(f"{tag:16s} gtbox        Dice={d:.3f}  slices={ns}  gt_vox={gv}", flush=True)
    for c in concepts:
        d, pv, gv, ns = run(ct, gt, lab, model, proc, win, "text", c)
        print(f"{tag:16s} text[{c:16s}] Dice={d:.3f}  pred_vox={pv}", flush=True)
    del model
    torch.cuda.empty_cache()
print("DONE", flush=True)
