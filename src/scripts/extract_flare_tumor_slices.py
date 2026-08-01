#!/usr/bin/env python3
"""Byte-range extract the 270 FLARE23 tumor-imaged cases from the remote Metadata.zip (one at a time,
never keeping more than one image on disk), slice the tumor (label 14) regions, and pool them for the
generic tumor model. Labels come from the LOCAL flare_labels.bin (no download). Checkpoints the pool
every 20 cases so a network hiccup never loses progress.

out: /scratch/ud3d4/acm_data/flare_tumor_pool/{images.npy, masks.npy, meta.json}
"""
import json
import os
import struct
import subprocess
import sys
import zlib

import nibabel as nib
import numpy as np
from skimage.transform import resize

LAB = "/scratch/ud3d4/acm_data/flare23_labels"
ZIP = "drive_UD:data/VKG datasets/Metadata.zip"
OUT = "/scratch/ud3d4/acm_data/flare_tumor_pool"
WIN = (-125, 225)
TUMOR, MINPX = 14, 15
os.makedirs(OUT, exist_ok=True)

li = json.load(open(f"{LAB}/label_index.json"))
LFIRST = li["first_offset"]
LIDX = {os.path.basename(n).replace(".nii.gz", ""): (off, cs) for n, off, cs in li["labels"]}
IIDX = {os.path.basename(n).replace("_0000.nii.gz", ""): (off, cs)
        for n, off, cs in json.load(open(f"{LAB}/image_index.json"))["images"]}
CASES = json.load(open(f"{LAB}/tumor_imaged_cases.json"))
LBIN = f"{LAB}/flare_labels.bin"


def inflate_local(buf):
    assert buf[:4] == b"PK\x03\x04", repr(buf[:4])
    nl = struct.unpack("<H", buf[26:28])[0]
    el = struct.unpack("<H", buf[28:30])[0]
    return zlib.decompress(buf[30 + nl + el:], -15)


def get_label(cid):
    off, cs = LIDX[cid]
    with open(LBIN, "rb") as f:
        f.seek(off - LFIRST)
        buf = f.read(cs + 300)
    open("/dev/shm/_lb.nii.gz", "wb").write(inflate_local(buf))
    return np.asarray(nib.load("/dev/shm/_lb.nii.gz").dataobj).astype(np.uint8)


def get_image(cid):
    off, cs = IIDX[cid]
    r = subprocess.run(["rclone", "cat", ZIP, "--offset", str(off), "--count", str(cs + 400)],
                       capture_output=True)
    open("/dev/shm/_im.nii.gz", "wb").write(inflate_local(r.stdout))
    return nib.load("/dev/shm/_im.nii.gz").get_fdata()


def hu_rgb(sl):
    lo, hi = WIN
    x = np.clip(sl, lo, hi)
    x = ((x - lo) / (hi - lo) * 255).astype(np.uint8)
    return np.stack([x] * 3, -1)


def save(imgs, masks, meta):
    np.save(f"{OUT}/images.npy", np.stack(imgs).astype(np.uint8))
    np.save(f"{OUT}/masks.npy", np.stack(masks).astype(np.uint8))
    json.dump(meta, open(f"{OUT}/meta.json", "w"))


limit = int(sys.argv[1]) if len(sys.argv) > 1 else len(CASES)
imgs, masks, meta, done = [], [], [], set()
for i, cid in enumerate(CASES[:limit]):
    try:
        lab = get_label(cid)
        if int((lab == TUMOR).sum()) < MINPX:
            continue
        ct = get_image(cid)
        if ct.shape != lab.shape:
            print(f"[{i+1}] {cid} shape mismatch {ct.shape} vs {lab.shape}", flush=True)
            continue
        n = 0
        for z in range(lab.shape[2]):
            if int((lab[:, :, z] == TUMOR).sum()) < MINPX:
                continue
            imgs.append(hu_rgb(resize(ct[:, :, z], (256, 256), preserve_range=True,
                                      anti_aliasing=True)).astype(np.uint8))
            masks.append(resize((lab[:, :, z] == TUMOR).astype(np.uint8), (256, 256), order=0,
                                preserve_range=True).astype(np.uint8))
            meta.append({"dataset": "flare", "case": cid})
            n += 1
        done.add(cid)
        print(f"[{i+1}/{limit}] {cid}: +{n} tumor slices (pool {len(imgs)}, {len(done)} cases)", flush=True)
        if len(done) % 20 == 0 and imgs:
            save(imgs, masks, meta)
            print(f"  ...checkpointed {len(imgs)} slices", flush=True)
    except Exception as e:
        print(f"[{i+1}] {cid} ERR {type(e).__name__}: {str(e)[:80]}", flush=True)

if imgs:
    save(imgs, masks, meta)
print(f"DONE FLARE tumor pool: {len(imgs)} slices from {len(done)} cases -> {OUT}", flush=True)
