#!/usr/bin/env python3
"""
Tri-planar preview PNGs for the FLARE per-patient delivery: axial / coronal / sagittal CT
slices with the multi-label mask overlaid, GT (top row) vs prediction (bottom row), one PNG
per case. Also serves as visual proof the volumes assemble correctly in all three planes.
"""
import os, glob
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

O = "/scratch/ud3d4/acm_data/FLARE_Task2/flare_handoff_v2"
OUT = f"{O}/previews"
os.makedirs(OUT, exist_ok=True)
ID2NAME = {1: "liver", 2: "right kidney", 3: "spleen", 4: "pancreas", 13: "left kidney"}
COLORS = {1: (0.90, 0.28, 0.28), 2: (0.25, 0.60, 0.95), 3: (0.35, 0.82, 0.42),
          4: (0.96, 0.80, 0.20), 13: (0.70, 0.40, 0.90)}
WIN = (-125, 275)  # abdominal soft-tissue window


def win(ct2d):
    lo, hi = WIN
    return np.clip((ct2d - lo) / (hi - lo), 0, 1)


def show(ax, ct2d, msk2d, title, aspect=1.0):
    ax.imshow(win(ct2d).T, cmap="gray", origin="lower", aspect=aspect)
    rgba = np.zeros(msk2d.shape + (4,))
    for lid, col in COLORS.items():
        m = msk2d == lid
        for c in range(3):
            rgba[..., c][m] = col[c]
        rgba[..., 3][m] = 0.45
    ax.imshow(np.transpose(rgba, (1, 0, 2)), origin="lower", aspect=aspect)
    ax.set_title(title, fontsize=9); ax.axis("off")


def main():
    cases = sorted(os.path.basename(f)[:-7] for f in glob.glob(f"{O}/ct/flare/*.nii.gz"))
    print(f"{len(cases)} cases")
    for cid in cases:
        cimg = nib.as_closest_canonical(nib.load(f"{O}/ct/flare/{cid}.nii.gz"))
        ct = cimg.get_fdata()
        sx, sy, sz = [float(z) for z in cimg.header.get_zooms()[:3]]
        gt = nib.as_closest_canonical(nib.load(f"{O}/ground_truth/flare/{cid}.nii.gz")).get_fdata().astype(int)
        pr = nib.as_closest_canonical(nib.load(f"{O}/ssl_predictions/flare/{cid}.nii.gz")).get_fdata().astype(int)
        # slice indices at the centroid of the observed organs (so the planes cut through anatomy)
        idx = np.argwhere(gt > 0)
        cx, cy, cz = [int(round(v)) for v in idx.mean(0)]
        # anatomically-correct aspect per plane (vertical-mm / horizontal-mm)
        a_ax, a_cor, a_sag = sy / sx, sz / sx, sz / sy
        fig, axes = plt.subplots(2, 3, figsize=(10.5, 7.6))
        for row, (msk, tag) in enumerate([(gt, "GT"), (pr, "Pred")]):
            show(axes[row][0], ct[:, :, cz], msk[:, :, cz], f"{tag} — axial z={cz}", a_ax)
            show(axes[row][1], ct[:, cy, :], msk[:, cy, :], f"{tag} — coronal y={cy}", a_cor)
            show(axes[row][2], ct[cx, :, :], msk[cx, :, :], f"{tag} — sagittal x={cx}", a_sag)
        handles = [Patch(color=COLORS[l], label=ID2NAME[l]) for l in ID2NAME if (gt == l).any()]
        fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8, frameon=False)
        fig.suptitle(f"FLARE {cid}  —  CT + multi-label overlay (GT top, prediction bottom)", fontsize=11)
        fig.tight_layout(rect=[0, 0.04, 1, 0.96])
        fig.savefig(f"{OUT}/{cid}_triplanar.png", dpi=110); plt.close(fig)
        print(f"  {cid} -> {cid}_triplanar.png")
    print(f"\nSaved {len(cases)} previews -> {OUT}")


if __name__ == "__main__":
    main()
