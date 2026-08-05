#!/usr/bin/env python3
"""Tumor-focused 3D pack for the app handoff — Krishna's priority: more tumor-bearing cases WITH CT.
For each tumor-bearing FLARE23 case that has a retrievable CT (270 of them, all in the KG), extract CT + label,
mesh per-label STLs (volume-preserving Taubin, kidneys split, spleen, NO GLB), and STREAM each case straight
to Drive (upload then delete local) so local disk stays tiny — it never uses /scratch, so it can't clash with
the training checkpoints. CPU + network only.

Drive: drive_UD:data/ssl_handoff_for_krishna/3d_tumor_focused/<case>/{CT.nii.gz, seg.nii.gz, meshes/*.stl}
"""
import json
import os
import shutil
import struct
import subprocess
import sys
import zlib

import numpy as np
import nibabel as nib

sys.path.insert(0, "/home/ud3d4/Desktop/SWOG/src/scripts")
from meshify import meshify

LAB = "/scratch/ud3d4/acm_data/flare23_labels"
ZIP = "drive_UD:data/VKG datasets/Metadata.zip"
DRIVE = "drive_UD:data/ssl_handoff_for_krishna/3d_tumor_focused"
TMP = "/dev/shm/_tf_pack"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 270

li = json.load(open(f"{LAB}/label_index.json")); LFIRST = li["first_offset"]
LIDX = {os.path.basename(n).replace(".nii.gz", ""): (off, cs) for n, off, cs in li["labels"]}
IIDX = {os.path.basename(n).replace("_0000.nii.gz", ""): (off, cs)
        for n, off, cs in json.load(open(f"{LAB}/image_index.json"))["images"]}
LBIN = f"{LAB}/flare_labels.bin"

# tumor-bearing + imaged + in-KG cases (KG-aligned; tumor-only-no-organ cases are excluded by construction)
recs = json.load(open("/home/ud3d4/Desktop/SWOG/kg/data/corpus_perpatient.json"))["records"]
tb = [r["case_id"] for r in recs if r["dataset"] == "flare"
      and any(o.get("has_tumor") for o in r["organs"].values())]
CASES = [c for c in tb if c in IIDX and c in LIDX][:N]


def inflate(buf):
    assert buf[:4] == b"PK\x03\x04", repr(buf[:4])
    nl = struct.unpack("<H", buf[26:28])[0]; el = struct.unpack("<H", buf[28:30])[0]
    return zlib.decompress(buf[30 + nl + el:], -15)


def get_label(cid, dst):
    off, cs = LIDX[cid]
    with open(LBIN, "rb") as f:
        f.seek(off - LFIRST); buf = f.read(cs + 300)
    open(dst, "wb").write(inflate(buf))


def get_ct(cid, dst):
    off, cs = IIDX[cid]
    r = subprocess.run(["rclone", "cat", ZIP, "--offset", str(off), "--count", str(cs + 400)],
                       capture_output=True)
    open(dst, "wb").write(inflate(r.stdout))


os.makedirs(TMP, exist_ok=True)
manifest, done = [], 0
print(f"tumor-focused pack: {len(CASES)} tumor-bearing imaged cases -> {DRIVE}", flush=True)
for i, cid in enumerate(CASES):
    cdir = f"{TMP}/{cid}"; os.makedirs(f"{cdir}/meshes", exist_ok=True)
    try:
        lp = f"{cdir}/_lab.nii.gz"; get_label(cid, lp); get_ct(cid, f"{cdir}/CT.nii.gz")
        ctn = nib.load(f"{cdir}/CT.nii.gz"); labn = nib.load(lp)
        if ctn.shape != labn.shape:
            print(f"[{i+1}] {cid} shape mismatch — skip", flush=True); shutil.rmtree(cdir); continue
        nib.save(nib.Nifti1Image(np.asarray(labn.dataobj).astype(np.uint8), ctn.affine), f"{cdir}/seg.nii.gz")
        os.remove(lp)
        r = meshify(f"{cdir}/seg.nii.gz", f"{cdir}/meshes", tag="gt", glb=False)
        if "tumor" not in r["organs"]:
            print(f"[{i+1}] {cid} tumor below mesh threshold — skip", flush=True); shutil.rmtree(cdir); continue
        subprocess.run(["rclone", "copy", cdir, f"{DRIVE}/{cid}", "--transfers", "4"], capture_output=True)
        manifest.append({"case": cid, "organs": r["organs"]}); done += 1
        if (i + 1) % 10 == 0:
            print(f"[{i+1}/{len(CASES)}] uploaded {done} cases", flush=True)
    except Exception as e:
        print(f"[{i+1}] {cid} ERR {type(e).__name__}: {str(e)[:70]}", flush=True)
    finally:
        shutil.rmtree(cdir, ignore_errors=True)   # keep local disk tiny

open(f"{TMP}/MANIFEST.json", "w").write(json.dumps(
    {"n_cases": done, "note": "tumor-bearing FLARE23 with CT; per-label STLs (kidneys split, spleen), "
     "volume-preserving Taubin, no GLB", "cases": manifest}, indent=1))
subprocess.run(["rclone", "copy", f"{TMP}/MANIFEST.json", DRIVE], capture_output=True)
print(f"DONE: {done} tumor-bearing cases with CT uploaded -> {DRIVE}", flush=True)
