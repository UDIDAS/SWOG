#!/usr/bin/env python3
"""
New-CT ingestion backend: CT volume -> SAM3 segmentation -> validation -> KG patient record.

Segmentation modes (per the agreed scope):
  • autonomous (no labels): full-image-box SAM3 for the 5 FLARE organs (liver/pancreas/spleen/
    kidneys). Degraded but usable for organ volumes; largest-3D-component cleanup. Tumors are NOT
    segmented autonomously (SAM3 can't localize them without a box — Dice ~0.11).
  • semi-oracle (a label mask is provided): GT-box-prompted SAM3 -> accurate organs AND tumor.

Reuses the vetted per-slice inference from src/scripts/infer_sam3.py. Streamlit-free.
"""
import os
import sys

import numpy as np
from scipy import ndimage
from skimage.transform import resize

# heavy deps (torch, transformers/SAM3, infer_sam3) are imported lazily inside Segmenter so that
# `import ingest` (for the constants / phenotypes / validate / overlay) stays cheap.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "scripts")

_CK = "/scratch/ud3d4/acm_data"
WINDOW = (-125, 225)     # FLARE abdominal soft-tissue HU window

# structure -> (checkpoint, output label id, label id in an uploaded GT mask)
ORGAN_MODELS = {
    "liver":        (f"{_CK}/FLARE_Task2/sam3/flare_t2_sam3_v3_liver.pth", 1, 1),
    "right_kidney": (f"{_CK}/FLARE_Task2/sam3/flare_t2_sam3_v3_right_kidney.pth", 2, 2),
    "spleen":       (f"{_CK}/FLARE_Task2/sam3/flare_t2_sam3_v3_spleen.pth", 3, 3),
    "pancreas":     (f"{_CK}/FLARE_Task2/sam3/flare_t2_sam3_v3_pancreas.pth", 4, 4),
    "left_kidney":  (f"{_CK}/FLARE_Task2/sam3/flare_t2_sam3_v3_left_kidney.pth", 13, 13),
}
TUMOR_MODEL = (f"{_CK}/FLARE/sam3/flare_sam3_v3_tumor.pth", 14, 14)   # semi-oracle only
LABEL_ORGAN = {1: "liver", 2: "right_kidney", 3: "spleen", 4: "pancreas", 13: "left_kidney"}
PLAUSIBLE_CM3 = {"liver": (600, 3000), "pancreas": (30, 220), "spleen": (40, 600),
                 "right_kidney": (60, 320), "left_kidney": (60, 320)}


class Segmenter:
    """Loads the SAM3 base once; swaps fine-tuned weights per structure."""

    def __init__(self, device="cuda"):
        if _SRC not in sys.path:
            sys.path.insert(0, _SRC)
        import infer_sam3 as I
        import torch
        self.I, self.torch, self.device = I, torch, device
        self.proc = I.Sam3Processor.from_pretrained(I.SAM3_BASE, token=I.HF_TOKEN,
                                                    revision=I.SAM3_REVISION)
        self.model = I.Sam3Model.from_pretrained(I.SAM3_BASE, token=I.HF_TOKEN,
                                                 revision=I.SAM3_REVISION).to(device).eval()
        self._cur = None

    def _apply(self, ckpt):
        if self._cur == ckpt:
            return
        state = self.torch.load(ckpt, map_location=self.device, weights_only=False)
        self.model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()},
                                   strict=False)
        self.model.to(self.device).eval()
        self._cur = ckpt

    def _seg_structure(self, ct, ckpt, out_label, gt, gt_label, stride, progress, si, ns, name):
        self._apply(ckpt)
        I = self.I
        lo, hi = WINDOW
        Z = ct.shape[2]
        out = np.zeros(ct.shape[:2] + (Z,), np.uint8)
        semi = gt is not None and int((gt == gt_label).sum()) >= 5
        zs = ([z for z in range(Z) if (gt[:, :, z] == gt_label).sum() >= 5] if semi
              else list(range(0, Z, stride)))
        for zi, z in enumerate(zs):
            sl = ct[:, :, z]
            H, W = sl.shape
            rgb = I.hu_to_rgb(resize(sl, (256, 256), preserve_range=True, anti_aliasing=True),
                              lo, hi).astype(np.uint8)
            if semi:
                g256 = resize((gt[:, :, z] == gt_label).astype(np.uint8), (256, 256), order=0,
                              preserve_range=True).astype(np.uint8)
                box = I.bbox_from_mask(g256, pad=3)
                if box is None:
                    continue
            else:
                box = [0, 0, 255, 255]
            prob = I.infer_slice(self.model, self.proc, rgb, box, self.device)
            pm = (resize(prob, (H, W), order=1, preserve_range=True) > 0.5).astype(np.uint8)
            out[:, :, z][pm > 0] = out_label
            if progress:
                progress(si, ns, zi, len(zs), name, "semi-oracle" if semi else "autonomous")
        if not semi and stride > 1:                       # fill strided z-gaps (hold nearest)
            for z in range(Z):
                if not out[:, :, z].any() and z > 0 and out[:, :, z - 1].any():
                    out[:, :, z] = out[:, :, z - 1]
        if not semi:                                      # keep largest 3D component (denoise)
            m = out == out_label
            lab, n = ndimage.label(m)
            if n > 1:
                keep = int(np.argmax(ndimage.sum(m, lab, range(1, n + 1)))) + 1
                out[m & (lab != keep)] = 0
        return out, semi

    def run(self, ct, organs, gt=None, want_tumor=False, stride=2, progress=None):
        """Return (multi_label_mask, modes) — modes[name] = 'semi-oracle' | 'autonomous'."""
        jobs = [(n,) + ORGAN_MODELS[n] for n in organs]
        if want_tumor and gt is not None:
            jobs.append(("tumor",) + TUMOR_MODEL)
        out = np.zeros(ct.shape, np.uint8)
        modes = {}
        for si, (name, ckpt, out_label, gt_label) in enumerate(jobs):
            seg, semi = self._seg_structure(ct, ckpt, out_label, gt, gt_label, stride,
                                            progress, si, len(jobs), name)
            out[seg > 0] = seg[seg > 0]
            modes[name] = "semi-oracle" if semi else "autonomous"
        return out, modes


def phenotypes(mask, spacing_cm3, burden_thresh=(2.0, 10.0)):
    """Build a corpus-style patient record from a multi-label mask."""
    organs, observed = {}, []
    tumor_mask = mask == 14
    tvox = int(tumor_mask.sum())
    # attach a tumor to the organ it overlaps most (semi-oracle only; autonomous has no tumor)
    tumor_organ = None
    if tvox:
        overlaps = {o: int((tumor_mask & ndimage.binary_dilation(mask == lab, iterations=3)).sum())
                    for lab, o in LABEL_ORGAN.items() if (mask == lab).any()}
        tumor_organ = max(overlaps, key=overlaps.get) if overlaps else "liver"
    for lab, organ in LABEL_ORGAN.items():
        om = mask == lab
        if om.sum() < 20:
            continue
        observed.append(organ)
        has_tumor = organ == tumor_organ and tvox > 0
        tvol = round(tvox * spacing_cm3, 2) if has_tumor else 0.0
        burden = ("high" if tvol >= burden_thresh[1] else "low") if has_tumor else "none"
        mult = "none"
        contain = "none"
        loc = "na"
        if has_tumor:
            mult = "multifocal" if ndimage.label(tumor_mask)[1] >= 2 else "solitary"
            frac = (tumor_mask & ndimage.binary_dilation(om, iterations=3)).sum() / tvox
            contain = "contained" if frac >= 0.90 else "boundary"
            if organ == "pancreas":
                loc = "unknown"
        organs[organ] = {
            "present": True, "organ_volume_cm3": round(int(om.sum()) * spacing_cm3, 2),
            "has_tumor": has_tumor, "tumor_volume_cm3": tvol, "tumor_voxels": tvox if has_tumor else 0,
            "burden_cat": burden, "multiplicity": mult, "containment": contain,
            "anatomic_location": loc, "size_cat": "unknown",
        }
    return {"dataset": "ingested", "observed_organs": observed, "n_observed": len(observed),
            "organs": organs, "granularity": "volume"}


def validate(record, gt=None, mask=None, spacing_cm3=None):
    """Plausibility per organ (+ Dice vs GT if a label mask was provided)."""
    rows = []
    inv = {v: k for k, v in {"liver": 1, "right_kidney": 2, "spleen": 3, "pancreas": 4,
                             "left_kidney": 13}.items()}
    for organ, od in record["organs"].items():
        v = od["organ_volume_cm3"]
        lo, hi = PLAUSIBLE_CM3.get(organ, (0, 1e9))
        dice = None
        if gt is not None and mask is not None:
            lab = {"liver": 1, "right_kidney": 2, "spleen": 3, "pancreas": 4, "left_kidney": 13}[organ]
            pm, gm = mask == lab, gt == lab
            if pm.sum() + gm.sum():
                dice = round(2 * int((pm & gm).sum()) / int(pm.sum() + gm.sum()), 3)
        rows.append({"organ": organ, "volume_cm3": v,
                     "status": "✓ plausible" if lo <= v <= hi else "⚠️ out of range",
                     "expected cm³": f"{lo}–{hi}", "Dice vs GT": dice})
    return rows


def overlay_png(ct, mask):
    """PNG bytes: the busiest axial slice, CT grayscale + colored mask overlay."""
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    z = int(np.argmax([(mask[:, :, k] > 0).sum() for k in range(mask.shape[2])]))
    lo, hi = WINDOW
    ctn = np.clip((ct[:, :, z] - lo) / (hi - lo), 0, 1)
    cmap = {1: (1, 0, 0), 2: (0, 1, 0), 3: (0, 0.4, 1), 4: (1, 1, 0), 13: (0, 1, 1), 14: (1, 0, 1)}
    rgba = np.zeros(mask[:, :, z].shape + (4,))
    for lab, col in cmap.items():
        m = mask[:, :, z] == lab
        for i, c in enumerate(col):
            rgba[..., i][m] = c
        rgba[..., 3][m] = 0.45
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.imshow(ctn.T, cmap="gray", origin="lower")
    ax.imshow(np.transpose(rgba, (1, 0, 2)), origin="lower")
    ax.axis("off"); ax.set_title(f"axial slice z={z}")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=90)
    plt.close(fig)
    return buf.getvalue()
