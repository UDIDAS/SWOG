#!/usr/bin/env python3
"""Byte-range extract FLARE23 ORGAN slices (liver / kidney / pancreas ONLY) from the remote Metadata.zip,
one case at a time (never keeping a full volume on disk). Labels come from the LOCAL flare_labels.bin.
Samples up to SPP evenly-spaced slices per organ per case so the pool spans all cases without bloating.
Checkpoints every 25 cases. Storage-safe.

out: /scratch/ud3d4/acm_data/flare_organ_pool/{images.npy, masks.npy, meta.json}
"""
import json
import os
import struct
import subprocess
import sys
import zlib

import numpy as np
import nibabel as nib
from skimage.transform import resize

LAB = "/scratch/ud3d4/acm_data/flare23_labels"
ZIP = "drive_UD:data/VKG datasets/Metadata.zip"
OUT = "/scratch/ud3d4/acm_data/flare_organ_pool"
WIN = (-125, 225)
MINPX, SPP = 50, 8                       # min organ px/slice ; slices per organ per case
ORGANS = {"liver": [1], "kidney": [2, 13], "pancreas": [4]}
os.makedirs(OUT, exist_ok=True)

li = json.load(open(f"{LAB}/label_index.json")); LFIRST = li["first_offset"]
LIDX = {os.path.basename(n).replace(".nii.gz", ""): (off, cs) for n, off, cs in li["labels"]}
IIDX = {os.path.basename(n).replace("_0000.nii.gz", ""): (off, cs)
        for n, off, cs in json.load(open(f"{LAB}/image_index.json"))["images"]}
LBIN = f"{LAB}/flare_labels.bin"
CASES = sorted(c for c in IIDX if c in LIDX)
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else len(CASES)
CASES = CASES[:LIMIT]


def inflate(buf):
    assert buf[:4] == b"PK\x03\x04", repr(buf[:4])
    nl = struct.unpack("<H", buf[26:28])[0]; el = struct.unpack("<H", buf[28:30])[0]
    return zlib.decompress(buf[30 + nl + el:], -15)


def get_label(cid):
    off, cs = LIDX[cid]
    with open(LBIN, "rb") as f:
        f.seek(off - LFIRST); buf = f.read(cs + 300)
    open("/dev/shm/_ol.nii.gz", "wb").write(inflate(buf))
    return np.asarray(nib.load("/dev/shm/_ol.nii.gz").dataobj).astype(np.uint8)


def get_image(cid):
    off, cs = IIDX[cid]
    r = subprocess.run(["rclone", "cat", ZIP, "--offset", str(off), "--count", str(cs + 400)], capture_output=True)
    open("/dev/shm/_oi.nii.gz", "wb").write(inflate(r.stdout))
    return nib.load("/dev/shm/_oi.nii.gz").get_fdata()


def hu_rgb(sl):
    lo, hi = WIN; x = np.clip(sl, lo, hi); x = ((x - lo) / (hi - lo) * 255).astype(np.uint8)
    return np.stack([x] * 3, -1)


def save(imgs, masks, meta):
    np.save(f"{OUT}/images.npy", np.stack(imgs).astype(np.uint8))
    np.save(f"{OUT}/masks.npy", np.stack(masks).astype(np.uint8))
    json.dump(meta, open(f"{OUT}/meta.json", "w"))


imgs, masks, meta, done = [], [], [], 0
print(f"extracting liver/kidney/pancreas from {len(CASES)} FLARE23 cases (SPP={SPP})", flush=True)
for i, cid in enumerate(CASES):
    try:
        lab = get_label(cid); ct = None
        for organ, labs in ORGANS.items():
            zs = [z for z in range(lab.shape[2]) if int(np.isin(lab[:, :, z], labs).sum()) >= MINPX]
            if not zs:
                continue
            pick = zs if len(zs) <= SPP else [zs[k] for k in np.linspace(0, len(zs) - 1, SPP).astype(int)]
            if ct is None:
                ct = get_image(cid)
                if ct.shape != lab.shape:
                    print(f"[{i+1}] {cid} shape mismatch — skip", flush=True); break
            for z in pick:
                m = np.isin(lab[:, :, z], labs).astype(np.uint8)
                imgs.append(hu_rgb(resize(ct[:, :, z], (256, 256), preserve_range=True, anti_aliasing=True)).astype(np.uint8))
                masks.append(resize(m, (256, 256), order=0, preserve_range=True).astype(np.uint8))
                meta.append({"organ": organ, "dataset": "flare", "case": cid})
        done += 1
        if (i + 1) % 25 == 0:
            save(imgs, masks, meta)
            print(f"[{i+1}/{len(CASES)}] pool {len(imgs)} slices ({done} cases)", flush=True)
    except Exception as e:
        print(f"[{i+1}] {cid} ERR {type(e).__name__}: {str(e)[:70]}", flush=True)

save(imgs, masks, meta)
import collections
print(f"DONE: {len(imgs)} organ slices from {done} cases | {dict(collections.Counter(m['organ'] for m in meta))} -> {OUT}", flush=True)
