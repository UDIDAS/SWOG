#!/usr/bin/env python3
"""AUTONOMOUS ensemble segmentation of an abdominal CT — NO labels, NO boxes.
  organs : SAM3 concept prompts (text='liver'/'spleen'/... ) — base or fine-tuned SAM3
  tumor  : the generic text-'tumor' model (train_tumor_generic.py)
  filter : a tumor voxel is kept only if it lands inside a segmented organ (KG rule)

Produces a multi-label volume + a KG phenotype record (organ volumes cm3, tumor per organ).

Usage:
  python infer_ensemble.py --ct data/FLARE23_0217_0000.nii.gz [--gt data/FLARE23_0217.nii.gz] \
      --out /scratch/.../pred.nii.gz
"""
import argparse
import json
import os
import sys

import nibabel as nib
import numpy as np
import torch
from skimage.transform import resize
from torch.amp import autocast

sys.path.insert(0, "/home/ud3d4/Desktop/SWOG/src/scripts")
import infer_sam3 as I
from run_pancreas_sam3 import _load_sam3_ckpt, SAM3_MODEL_ID, HF_TOKEN

WIN = (-125, 225)
TUMOR_CKPT = "/scratch/ud3d4/acm_data/tumor_pool/sam3_tumor_generic.pth"
# organ concept -> output label id (matches FLARE/our KG scheme)
ORGANS = [("liver", 1), ("right kidney", 2), ("spleen", 3), ("pancreas", 4), ("left kidney", 13)]
TUMOR_LABEL = 14
VOX_MIN = 15          # min mask pixels per slice to keep
CONF_MIN = 0.15       # min top-detection confidence (sigmoid of pred_logit) to accept a slice


def load_base():
    from transformers import Sam3Model, Sam3Processor
    proc = Sam3Processor.from_pretrained(SAM3_MODEL_ID, token=HF_TOKEN)
    model = Sam3Model.from_pretrained(SAM3_MODEL_ID, token=HF_TOKEN)
    return model.to("cuda").eval(), proc


def concept_slice(model, proc, rgb, concept):
    """Return (prob_map 256x256, confidence) for a text concept, no box."""
    inputs = proc(images=[rgb], text=[concept], return_tensors="pt")
    kw = {"pixel_values": inputs["pixel_values"].to("cuda")}
    for k in ("input_ids", "attention_mask"):
        if inputs.get(k) is not None:
            kw[k] = inputs[k].to("cuda")
    with torch.no_grad(), autocast("cuda"):
        out = model(**kw)
        pl = out.pred_logits[0].sigmoid()
        i = int(pl.argmax())
        conf = float(pl[i])
        m = torch.nn.functional.interpolate(out.pred_masks[0:1, i:i + 1].float(),
                                            size=(256, 256), mode="bilinear", align_corners=False)
    return m.sigmoid().squeeze().cpu().numpy(), conf


def seg_concept(ct, model, proc, concept):
    X, Y, Z = ct.shape
    out = np.zeros(ct.shape, np.uint8)
    for z in range(Z):
        rgb = I.hu_to_rgb(resize(ct[:, :, z], (256, 256), preserve_range=True, anti_aliasing=True),
                          *WIN).astype(np.uint8)
        prob, conf = concept_slice(model, proc, rgb, concept)
        if conf < CONF_MIN:
            continue
        m = (resize(prob, (X, Y), order=1, preserve_range=True) > 0.5).astype(np.uint8)
        if m.sum() >= VOX_MIN:
            out[:, :, z] = m
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ct", required=True)
    ap.add_argument("--gt", default=None)
    ap.add_argument("--out", default="/scratch/ud3d4/acm_data/tumor_pool/ensemble_pred.nii.gz")
    ap.add_argument("--organ_ckpt", default=None, help="optional per-run fine-tuned organ ckpt; default base SAM3")
    args = ap.parse_args()

    ct_nii = nib.load(args.ct)
    ct = ct_nii.get_fdata()
    sp = float(np.prod(ct_nii.header.get_zooms()[:3])) / 1000.0
    gt = nib.load(args.gt).get_fdata().astype(np.uint8) if args.gt else None
    pred = np.zeros(ct.shape, np.uint8)

    # organs via base SAM3 concept prompting
    organ_model, proc = load_base()
    dice = {}
    for concept, lab in ORGANS:
        m = seg_concept(ct, organ_model, proc, concept)
        pred[m > 0] = lab
        if gt is not None:
            gm = (gt == lab)
            s = int((m > 0).sum()) + int(gm.sum())
            dice[concept] = round(2 * int(((m > 0) & gm).sum()) / s, 3) if s else None
        print(f"  organ {concept:13s} vox={int((m>0).sum())}  Dice={dice.get(concept)}", flush=True)
    del organ_model
    torch.cuda.empty_cache()

    # tumor via the generic tumor model, then KG filter: keep only inside an organ
    organ_mask = pred > 0
    tmodel = _load_sam3_ckpt(TUMOR_CKPT, "cuda") if os.path.exists(TUMOR_CKPT) else None
    if tmodel is not None:
        tm = seg_concept(ct, tmodel, proc, "tumor")
        tm_kept = (tm > 0) & organ_mask                 # KG rule: tumor must be inside an organ
        pred[tm_kept] = TUMOR_LABEL
        if gt is not None:
            gm = (gt == TUMOR_LABEL)
            s = int(tm_kept.sum()) + int(gm.sum())
            dice["tumor(filtered)"] = round(2 * int((tm_kept & gm).sum()) / s, 3) if s else None
        print(f"  tumor raw_vox={int((tm>0).sum())} kept_in_organ={int(tm_kept.sum())} "
              f"Dice={dice.get('tumor(filtered)')}", flush=True)

    nib.save(nib.Nifti1Image(pred, ct_nii.affine, ct_nii.header), args.out)
    # phenotype record
    rec = {"organs": {c: {"organ_volume_cm3": round(int((pred == lab).sum()) * sp, 2)}
                      for c, lab in ORGANS if (pred == lab).sum() > 0}}
    print("PHENOTYPES:", json.dumps(rec), flush=True)
    print("DICE:", json.dumps(dice), flush=True)
    print("-> saved", args.out, flush=True)


if __name__ == "__main__":
    main()
