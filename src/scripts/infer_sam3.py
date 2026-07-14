#!/usr/bin/env python3
"""
Standalone SAM3 inference for the segmentation hand-off.

CT volume in -> per-slice preprocess -> box prompt -> SAM3 -> multi-label mask out.
Self-contained: only needs torch + transformers + nibabel + numpy + scikit-image.

Base model : facebook/sam3  (revision 3c879f39826c281e95690f02c7821c4de09afae7)
Framework  : PyTorch 2.5.1+cu121, transformers 5.8.1
Fine-tuned checkpoints: sam3_pancreas_organ_v3.pt, sam3_pancreas_tumor_v3.pt, sam3_lits_tumor_v3.pt

Preprocessing (must match training):
  - 2D per-axial-slice, resized to 256x256, HU-windowed to uint8 RGB
  - HU window: pancreas [-100, 300], LiTS [-100, 400]
  - prediction = sigmoid(best-query mask) > 0.5, resized back to native, no post-processing

Prompt modes:
  --prompt gtbox   : GT-derived bounding box (padded +/-3px) -- SEMI-ORACLE (localization given)
  --prompt fullbox : full-image box -- AUTONOMOUS w.r.t. localization (no GT hint within slice)
  --prompt box  x1 y1 x2 y2 (in native pixels, per slice) -- external/generated prompt

Examples:
  # semi-oracle (reproduces the delivered masks)
  python infer_sam3.py --ct Task07/imagesTr/pancreas_001.nii.gz --gt Task07/labelsTr/pancreas_001.nii.gz \
      --checkpoint sam3_pancreas_tumor_v3.pt --dataset pancreas --label 2 --prompt gtbox --out pred.nii.gz
  # autonomous (GT box removed) -- the true detection+segmentation-within-slice number
  python infer_sam3.py --ct pancreas_001.nii.gz --gt pancreas_001_gt.nii.gz \
      --checkpoint sam3_pancreas_tumor_v3.pt --dataset pancreas --label 2 --prompt fullbox --out pred.nii.gz
"""
import argparse, os
import numpy as np
import nibabel as nib
import torch
from torch.amp import autocast
from skimage.transform import resize
from transformers import Sam3Model, Sam3Processor

SAM3_BASE = "facebook/sam3"
SAM3_REVISION = "3c879f39826c281e95690f02c7821c4de09afae7"
HF_TOKEN = os.environ.get("HF_TOKEN") or (open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
                                          if os.path.exists(os.path.expanduser("~/.cache/huggingface/token")) else None)
HU_WINDOW = {"pancreas": (-100, 300), "lits": (-100, 400)}


def hu_to_rgb(slice_hu, lo, hi):
    clip = np.clip(slice_hu, lo, hi)
    norm = ((clip - lo) / (hi - lo) * 255).astype(np.uint8)
    return np.stack([norm] * 3, axis=-1)


def bbox_from_mask(mask2d, pad=3):
    ys, xs = np.where(mask2d > 0)
    if len(xs) == 0:
        return None
    H, W = mask2d.shape
    return [max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad),
            min(W - 1, int(xs.max()) + pad), min(H - 1, int(ys.max()) + pad)]


def best_mask(outputs):
    pm, pl = outputs.pred_masks, outputs.pred_logits
    i = pl[0].sigmoid().argmax()
    return pm[0:1, i:i + 1]  # [1,1,h,w]


def load_model(checkpoint, device):
    proc = Sam3Processor.from_pretrained(SAM3_BASE, token=HF_TOKEN, revision=SAM3_REVISION)
    model = Sam3Model.from_pretrained(SAM3_BASE, token=HF_TOKEN, revision=SAM3_REVISION)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()}, strict=False)
    model.to(device).eval()
    return model, proc


def infer_slice(model, proc, rgb256, box256, device):
    """box256 in 256-space (xyxy) or None. Returns 256x256 float prob map."""
    inputs = proc(images=[rgb256], text=["visual"],
                  input_boxes=[[box256]] if box256 is not None else None,
                  input_boxes_labels=[[1]] if box256 is not None else None,
                  return_tensors="pt")
    kw = {"pixel_values": inputs["pixel_values"].to(device)}
    for k in ("input_ids", "attention_mask", "input_boxes", "input_boxes_labels"):
        if inputs.get(k) is not None:
            kw[k] = inputs[k].to(device)
    with torch.no_grad(), autocast("cuda"):
        out = model(**kw)
        m = torch.nn.functional.interpolate(best_mask(out).float(), size=(256, 256),
                                            mode="bilinear", align_corners=False)
    return m.sigmoid().squeeze().cpu().numpy()


def run_volume(ct_path, gt_path, checkpoint, dataset, label, prompt, out_path, device):
    lo, hi = HU_WINDOW[dataset]
    model, proc = load_model(checkpoint, device)
    ct_nii = nib.load(ct_path); ct = ct_nii.get_fdata()
    gt = nib.load(gt_path).get_fdata().astype(np.uint8) if gt_path else None
    Z = ct.shape[2] if ct.ndim == 3 else ct.shape[0]
    get = (lambda z: ct[:, :, z]) if ct.ndim == 3 else (lambda z: ct[z])
    getg = (lambda z: (gt[:, :, z] if gt.ndim == 3 else gt[z])) if gt is not None else (lambda z: None)
    pred = np.zeros(ct.shape, dtype=np.uint8)

    for z in range(Z):
        sl = get(z); H, W = sl.shape
        gsl = getg(z)
        # slice selection: run only where GT has the label (matches delivered protocol); for fully
        # autonomous detection across all slices, drop this guard.
        if gsl is not None and (gsl == label).sum() < 5 and prompt in ("gtbox", "fullbox"):
            continue
        rgb = hu_to_rgb(resize(sl, (256, 256), preserve_range=True, anti_aliasing=True), lo, hi).astype(np.uint8)
        if prompt == "gtbox":
            g256 = resize((gsl == label).astype(np.uint8), (256, 256), order=0, preserve_range=True).astype(np.uint8)
            box = bbox_from_mask(g256, pad=3)
            if box is None:
                continue
        elif prompt == "fullbox":
            box = [0, 0, 255, 255]
        else:  # explicit box in native -> scale to 256
            x1, y1, x2, y2 = prompt
            box = [x1 / W * 256, y1 / H * 256, x2 / W * 256, y2 / H * 256]
        prob = infer_slice(model, proc, rgb, box, device)
        m = (resize(prob, (H, W), order=1, preserve_range=True) > 0.5).astype(np.uint8)
        if ct.ndim == 3:
            pred[:, :, z][m > 0] = label
        else:
            pred[z][m > 0] = label

    nib.save(nib.Nifti1Image(pred, ct_nii.affine, ct_nii.header), out_path)
    return pred


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Standalone SAM3 inference")
    ap.add_argument("--ct", required=True)
    ap.add_argument("--gt", default=None, help="GT NIfTI (needed for --prompt gtbox and slice selection)")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", required=True, choices=["pancreas", "lits"])
    ap.add_argument("--label", type=int, required=True, help="target label id (1=organ, 2=tumor)")
    ap.add_argument("--prompt", nargs="+", default=["gtbox"],
                    help="gtbox | fullbox | four ints x1 y1 x2 y2")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    prompt = args.prompt[0] if len(args.prompt) == 1 else [int(v) for v in args.prompt]
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    run_volume(args.ct, args.gt, args.checkpoint, args.dataset, args.label, prompt, args.out, dev)
    print(f"saved {args.out}")
