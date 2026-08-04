#!/usr/bin/env python3
"""End-to-end 3D reconstruction from a raw CT volume: run the generic ORGAN model (concept prompts
liver/kidney/pancreas) and the generic TUMOR model slice-by-slice, assemble a labelled 3D segmentation
aligned to the CT's affine, optionally KG-repair it, and mesh it to app-ready 3D surfaces.

Label scheme (matches FLARE / the mesh tools): 1 liver · 2 kidney · 4 pancreas · 14 tumor.

Usage (needs one GPU):
  python reconstruct_3d.py CT.nii.gz OUT_DIR [--no-kg] [--organ-ckpt X.pth] [--tumor-ckpt Y.pth]
"""
import argparse
import os
import sys

import numpy as np
import nibabel as nib
import torch
from torch.amp import autocast
from skimage.transform import resize

sys.path.insert(0, "/home/ud3d4/Desktop/SWOG/src/scripts")
from run_pancreas_sam3 import _load_sam3_ckpt, SAM3_MODEL_ID, HF_TOKEN, extract_best_mask_soft
from meshify import meshify

WIN = (-125, 225)
ORGAN_PROMPTS = {"liver": 1, "kidney": 2, "pancreas": 4}   # prompt -> output label
TUMOR_LABEL = 14
DEF_ORGAN = "/scratch/ud3d4/acm_data/organ_pool_lkp/sam3_organ_generic.pth"
DEF_TUMOR = "/scratch/ud3d4/acm_data/tumor_pool/sam3_tumor_generic.pth"


def hu_rgb(sl):
    lo, hi = WIN; x = np.clip(sl, lo, hi); x = ((x - lo) / (hi - lo) * 255).astype(np.uint8)
    return np.stack([x] * 3, -1)


def predict_slice(model, proc, rgb, prompt, dev, native_hw):
    inp = proc(images=[rgb], text=[prompt], return_tensors="pt")
    kw = {"pixel_values": inp["pixel_values"].to(dev)}
    for k in ("input_ids", "attention_mask"):
        if inp.get(k) is not None:
            kw[k] = inp[k].to(dev)
    with torch.no_grad(), autocast("cuda"):
        out = model(**kw)
        pm = extract_best_mask_soft(out.pred_masks, out.pred_logits)
        pm = torch.nn.functional.interpolate(pm.float(), size=(256, 256), mode="bilinear", align_corners=False)
    m = pm.sigmoid().squeeze().cpu().numpy() > 0.5
    return resize(m.astype(np.uint8), native_hw, order=0, preserve_range=True).astype(bool)


def reconstruct(ct_path, out_dir, organ_ckpt, tumor_ckpt, use_kg=True):
    from transformers import Sam3Processor
    os.makedirs(out_dir, exist_ok=True)
    dev = "cuda:0"
    proc = Sam3Processor.from_pretrained(SAM3_MODEL_ID, token=HF_TOKEN)
    organ_model = _load_sam3_ckpt(organ_ckpt, dev)
    tumor_model = _load_sam3_ckpt(tumor_ckpt, dev)
    ct_nii = nib.load(ct_path); ct = ct_nii.get_fdata()
    Z = ct.shape[2]; seg = np.zeros(ct.shape, np.uint8)
    for z in range(Z):
        sl = ct[:, :, z]
        if sl.max() - sl.min() < 10:
            continue
        rgb = hu_rgb(resize(sl, (256, 256), preserve_range=True, anti_aliasing=True).astype(np.float32))
        hw = sl.shape
        for prompt, lab in ORGAN_PROMPTS.items():
            m = predict_slice(organ_model, proc, rgb, prompt, dev, hw)
            seg[:, :, z][m] = lab
        t = predict_slice(tumor_model, proc, rgb, "tumor", dev, hw)
        seg[:, :, z][t] = TUMOR_LABEL
        if (z + 1) % 40 == 0:
            print(f"  slice {z+1}/{Z}", flush=True)
    if use_kg:
        try:
            from kg_guided_segment import repair
            import json
            atlas = json.load(open("/home/ud3d4/Desktop/SWOG/kg/data/kg_atlas.json"))
            seg = repair(seg, ct_nii.header.get_zooms()[:3], atlas)
            print("  KG repair applied", flush=True)
        except Exception as e:
            print(f"  KG repair skipped: {type(e).__name__}", flush=True)
    seg_path = f"{out_dir}/segmentation.nii.gz"
    nib.save(nib.Nifti1Image(seg, ct_nii.affine), seg_path)
    nib.save(ct_nii, f"{out_dir}/CT.nii.gz")
    r = meshify(seg_path, f"{out_dir}/meshes", tag="pred")
    print("reconstructed:", {o: d["volume_cm3"] for o, d in r["organs"].items()}, "-> ", r["glb"])
    return seg_path, r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ct"); ap.add_argument("out_dir")
    ap.add_argument("--organ-ckpt", default=DEF_ORGAN); ap.add_argument("--tumor-ckpt", default=DEF_TUMOR)
    ap.add_argument("--no-kg", action="store_true")
    a = ap.parse_args()
    reconstruct(a.ct, a.out_dir, a.organ_ckpt, a.tumor_ckpt, use_kg=not a.no_kg)
