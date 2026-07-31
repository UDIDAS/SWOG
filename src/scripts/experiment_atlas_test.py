#!/usr/bin/env python3
"""
Unsupervised test-KG experiment: can we segment a held-out CT with NO GT box, using an organ-box
ATLAS built from the train masks (the train-KG prior)?  Compare autonomous-atlas vs full-box vs GT.

atlas = per organ, the average fractional z-range + in-slice bbox over a sample of train masks.
For a test CT we place that atlas box (scaled to the test volume) as SAM3's prompt — no GT used.
"""
import json
import struct
import sys
import time
import zlib

import nibabel as nib
import numpy as np
from skimage.transform import resize

sys.path.insert(0, "/home/ud3d4/Desktop/SWOG/src/scripts")
import infer_sam3 as I

LAB = "/scratch/ud3d4/acm_data/flare23_labels"
CK = "/scratch/ud3d4/acm_data/FLARE_Task2/sam3"
ORGANS = {"liver": (1, f"{CK}/flare_t2_sam3_v3_liver.pth"),
          "spleen": (3, f"{CK}/flare_t2_sam3_v3_spleen.pth"),
          "pancreas": (4, f"{CK}/flare_t2_sam3_v3_pancreas.pth"),
          "right_kidney": (2, f"{CK}/flare_t2_sam3_v3_right_kidney.pth"),
          "left_kidney": (13, f"{CK}/flare_t2_sam3_v3_left_kidney.pth")}
WIN = (-125, 225)


def iter_train_masks(n):
    idx = json.load(open(f"{LAB}/label_index.json"))
    first, labs = idx["first_offset"], idx["labels"]
    buf = open(f"{LAB}/flare_labels.bin", "rb").read()
    step = max(1, len(labs) // n)
    for name, off, cs in labs[::step][:n]:
        p = off - first
        if buf[p:p + 4] != b"PK\x03\x04":
            continue
        nl = struct.unpack("<H", buf[p + 26:p + 28])[0]
        el = struct.unpack("<H", buf[p + 28:p + 30])[0]
        try:
            raw = zlib.decompress(buf[p + 30 + nl + el:p + 30 + nl + el + cs], -15)
            open("/dev/shm/_a.nii.gz", "wb").write(raw)
            yield np.asarray(nib.load("/dev/shm/_a.nii.gz").dataobj).astype(np.uint8)
        except Exception:
            continue


def build_atlas(n=120):
    """Per organ: mean fractional [z_lo,z_hi, x_lo,y_lo,x_hi,y_hi]."""
    acc = {o: [] for o in ORGANS}
    for mask in iter_train_masks(n):
        X, Y, Z = mask.shape
        for o, (lab, _) in ORGANS.items():
            m = mask == lab
            if m.sum() < 50:
                continue
            xs, ys, zs = np.where(m)
            acc[o].append([zs.min() / Z, zs.max() / Z, xs.min() / X, ys.min() / Y,
                           xs.max() / X, ys.max() / Y])
    return {o: (np.mean(v, axis=0).tolist() if v else None) for o, v in acc.items()}


def seg_organ(model, proc, ct, box_or_full, zrange):
    lo, hi = WIN
    X, Y, Z = ct.shape
    out = np.zeros(ct.shape, np.uint8)
    z0, z1 = zrange
    for z in range(max(0, z0), min(Z, z1)):
        sl = ct[:, :, z]
        rgb = I.hu_to_rgb(resize(sl, (256, 256), preserve_range=True, anti_aliasing=True),
                          lo, hi).astype(np.uint8)
        prob = I.infer_slice(model, proc, rgb, box_or_full, "cuda")
        pm = (resize(prob, (X, Y), order=1, preserve_range=True) > 0.5).astype(np.uint8)
        out[:, :, z] = pm
    return out


def main():
    t0 = time.time()
    atlas = build_atlas(120)
    print(f"atlas built in {time.time()-t0:.0f}s:")
    for o, b in atlas.items():
        print(f"  {o}: {'none' if b is None else [round(x, 2) for x in b]}")
    ct_nii = nib.load("data/FLARE23_0217_0000.nii.gz")
    ct = ct_nii.get_fdata()
    gt = nib.load("data/FLARE23_0217.nii.gz").get_fdata().astype(np.uint8)
    X, Y, Z = ct.shape
    sp = float(np.prod(ct_nii.header.get_zooms()[:3])) / 1000.0
    print(f"\ntest CT FLARE23_0217 {ct.shape}  voxel {sp*1000:.3f} mm^3\n")
    print(f"{'organ':13s} {'GT vol':>9s} {'atlas vol':>10s} {'atlasDice':>10s} "
          f"{'fullbox vol':>12s} {'fullDice':>9s}")
    for o, (lab, ckpt) in ORGANS.items():
        if atlas[o] is None:
            continue
        model, proc = I.load_model(ckpt, "cuda")
        zlo, zhi, xlo, ylo, xhi, yhi = atlas[o]
        box256 = [xlo * 256, ylo * 256, xhi * 256, yhi * 256]           # atlas box in 256-space
        zr = (int(zlo * Z), int(zhi * Z) + 1)
        a = seg_organ(model, proc, ct, box256, zr)
        f = seg_organ(model, proc, ct, [0, 0, 255, 255], zr)           # full-box baseline, same z
        gm = (gt == lab)
        dv = lambda m: 2 * int((m & gm).sum()) / (int(m.sum()) + int(gm.sum()) + 1e-9)
        print(f"{o:13s} {gm.sum()*sp:9.1f} {a.sum()*sp:10.1f} {dv(a):10.3f} "
              f"{f.sum()*sp:12.1f} {dv(f):9.3f}")


if __name__ == "__main__":
    main()
