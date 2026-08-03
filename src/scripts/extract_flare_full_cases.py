#!/usr/bin/env python3
"""Byte-range extract FULL FLARE23 volumes (whole CT + full 13-organ+tumor label) for N cases that are
(a) in the 1,312 KG, (b) already in the tumor pool (so their tumor slices are in the 4-stage training),
and (c) organ-rich. These full volumes unlock organ segmentation, the predicted KG, and CT/pred/GT
reconstruction — the things tumor-slices-only couldn't. Storage-safe: one case in flight, checkpointed.

out: /scratch/ud3d4/acm_data/flare_full_cases/{cid}_ct.nii.gz, {cid}_label.nii.gz  (+ manifest.json)
"""
import json
import os
import struct
import subprocess
import sys
import zlib

import nibabel as nib
import numpy as np

LAB = "/scratch/ud3d4/acm_data/flare23_labels"
ZIP = "drive_UD:data/VKG datasets/Metadata.zip"
OUT = "/scratch/ud3d4/acm_data/flare_full_cases"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
os.makedirs(OUT, exist_ok=True)

li = json.load(open(f"{LAB}/label_index.json"))
LFIRST = li["first_offset"]
LIDX = {os.path.basename(n).replace(".nii.gz", ""): (off, cs) for n, off, cs in li["labels"]}
IIDX = {os.path.basename(n).replace("_0000.nii.gz", ""): (off, cs)
        for n, off, cs in json.load(open(f"{LAB}/image_index.json"))["images"]}
LBIN = f"{LAB}/flare_labels.bin"

# ---- select N cases: in KG ∩ image-available ∩ tumor-pool, richest organs first ----
corpus = {r["case_id"]: r for r in json.load(open("/home/ud3d4/Desktop/SWOG/kg/data/corpus_flare_train.json"))["records"]}
pool = set(r["case"] for r in json.load(open("/scratch/ud3d4/acm_data/flare_tumor_pool/meta.json")))
cand = [c for c in corpus if c in IIDX and c in pool]
cand.sort(key=lambda c: corpus[c].get("n_observed", 0), reverse=True)
SEL = cand[:N]
print(f"candidates (KG ∩ image ∩ tumor-pool): {len(cand)} | selecting {len(SEL)}", flush=True)
print("organ counts of selection:", sorted({corpus[c].get('n_observed') for c in SEL}, reverse=True), flush=True)


def inflate(buf):
    assert buf[:4] == b"PK\x03\x04", repr(buf[:4])
    nl = struct.unpack("<H", buf[26:28])[0]; el = struct.unpack("<H", buf[28:30])[0]
    return zlib.decompress(buf[30 + nl + el:], -15)


def get_label(cid):
    off, cs = LIDX[cid]
    with open(LBIN, "rb") as f:
        f.seek(off - LFIRST); buf = f.read(cs + 300)
    open("/dev/shm/_flb.nii.gz", "wb").write(inflate(buf))
    return nib.load("/dev/shm/_flb.nii.gz")


def get_image(cid):
    off, cs = IIDX[cid]
    r = subprocess.run(["rclone", "cat", ZIP, "--offset", str(off), "--count", str(cs + 400)],
                       capture_output=True)
    open("/dev/shm/_fim.nii.gz", "wb").write(inflate(r.stdout))
    return nib.load("/dev/shm/_fim.nii.gz")


manifest, done = [], set(os.path.basename(f).replace("_ct.nii.gz", "")
                         for f in os.listdir(OUT) if f.endswith("_ct.nii.gz"))
for i, cid in enumerate(SEL):
    if cid in done:
        continue
    try:
        lb = get_label(cid); im = get_image(cid)
        L = np.asarray(lb.dataobj).astype(np.uint8)
        I = np.asarray(im.dataobj)
        if I.shape != L.shape:
            print(f"[{i+1}] {cid} shape mismatch {I.shape} vs {L.shape} — skip", flush=True)
            continue
        nib.save(nib.Nifti1Image(I.astype(np.int16), im.affine, im.header), f"{OUT}/{cid}_ct.nii.gz")
        nib.save(nib.Nifti1Image(L, lb.affine, lb.header), f"{OUT}/{cid}_label.nii.gz")
        organs = sorted(int(v) for v in np.unique(L) if v != 0)
        manifest.append({"case": cid, "shape": list(I.shape), "labels": organs,
                         "has_tumor": 14 in organs})
        json.dump(manifest, open(f"{OUT}/manifest.json", "w"), indent=1)
        print(f"[{i+1}/{len(SEL)}] {cid}: saved CT+label {I.shape}, labels={organs}", flush=True)
    except Exception as e:
        print(f"[{i+1}] {cid} ERR {type(e).__name__}: {str(e)[:80]}", flush=True)

print(f"DONE: {len(manifest)} full cases -> {OUT}", flush=True)
