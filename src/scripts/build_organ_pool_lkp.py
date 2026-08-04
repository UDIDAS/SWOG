#!/usr/bin/env python3
"""Build the focused ORGAN pool — liver / kidney / pancreas ONLY (spleen + other FLARE organs dropped),
pooled across every local source that has them, patient-tagged for leakage-free splits. Mirrors the
tumor pool: each entry = 256x256 uint8 RGB (HU-windowed) + binary organ mask + meta{organ,dataset,case}.
The model will be prompted with the organ name ("liver"/"kidney"/"pancreas").

Sources (all local): LiTS(liver) · KiTS(kidney) · MSD(pancreas) · FLARE-Task2(l/k/p) · FLARE23 40 full(l/k/p).
out: /scratch/ud3d4/acm_data/organ_pool_lkp/{images.npy, masks.npy, meta.json}
"""
import glob
import json
import os
from collections import Counter

import numpy as np
import nibabel as nib
from skimage.transform import resize

OUT = "/scratch/ud3d4/acm_data/organ_pool_lkp"
os.makedirs(OUT, exist_ok=True)
WIN = (-125, 225)
MINPX, CAP = 50, 5000                      # min organ px/slice ; cap per (organ,dataset) for balance
# unified organ -> per-dataset label(s)
FLARE_LAB = {"liver": [1], "kidney": [2, 13], "pancreas": [4]}   # FLARE / FLARE-Task2 13-organ scheme
imgs, masks, meta = [], [], []


def hu_rgb(sl):
    lo, hi = WIN
    x = np.clip(sl, lo, hi); x = ((x - lo) / (hi - lo) * 255).astype(np.uint8)
    return np.stack([x] * 3, -1)


def add(ct2d, m2d, organ, dataset, case):
    if int(m2d.sum()) < MINPX:
        return 0
    X, Y = ct2d.shape
    imgs.append(hu_rgb(resize(ct2d, (256, 256), preserve_range=True, anti_aliasing=True)).astype(np.uint8))
    masks.append(resize(m2d.astype(np.uint8), (256, 256), order=0, preserve_range=True).astype(np.uint8))
    meta.append({"organ": organ, "dataset": dataset, "case": case})
    return 1


def from_volume(ct, seg, labels, axial, organ, dataset, case, budget):
    """Extract slices where organ present. axial = which axis is the slice axis."""
    n = 0
    Z = ct.shape[axial]
    for z in range(Z):
        if budget[0] <= 0:
            break
        s = (seg[z] if axial == 0 else seg[:, :, z])
        m = np.isin(s, labels)
        if int(m.sum()) < MINPX:
            continue
        c = (ct[z] if axial == 0 else ct[:, :, z])
        n += add(c, m, organ, dataset, case); budget[0] -= 1
    return n


def run_source(name, files, loader, labels, axial, organ):
    budget = [CAP]
    kept_cases = 0
    for f in files:
        if budget[0] <= 0:
            break
        try:
            ct, seg, case = loader(f)
            if ct.shape != seg.shape:
                continue
            if from_volume(ct, seg, labels, axial, organ, name, case, budget) > 0:
                kept_cases += 1
        except Exception as e:
            print(f"  {name}/{organ} {os.path.basename(str(f))} ERR {type(e).__name__}", flush=True)
    print(f"{name:12s} {organ:9s}: {CAP-budget[0]:5d} slices from {kept_cases} cases", flush=True)


# ---- LiTS liver (raw npy volumes, axial=0, liver=1) ----
def lits_loader(vid):
    ct = np.load(f"/scratch/ud3d4/acm_data/Data/ct/volume-{vid}.npy")
    seg = np.load(f"/scratch/ud3d4/acm_data/Data/seg/segmentation-{vid}.npy")
    return ct, seg, f"volume-{vid}"
lits_ids = sorted(int(os.path.basename(f).replace("volume-", "").replace(".npy", ""))
                  for f in glob.glob("/scratch/ud3d4/acm_data/Data/ct/volume-*.npy"))
run_source("lits", lits_ids, lits_loader, [1], 0, "liver")

# ---- MSD pancreas (nii, axial=2, pancreas=1) ----
def msd_loader(lp):
    cid = os.path.basename(lp).replace(".nii.gz", "")
    ct = nib.load(f"/scratch/ud3d4/acm_data/Pancreas/imagesTr/{cid}.nii.gz").get_fdata()
    seg = np.asarray(nib.load(lp).dataobj).astype(np.uint8)
    return ct, seg, cid
run_source("msd", sorted(glob.glob("/scratch/ud3d4/acm_data/Pancreas/labelsTr/pancreas_*.nii.gz")),
           msd_loader, [1], 2, "pancreas")

# ---- FLARE-Task2 (nii, axial=2, l/k/p) ----
def ft2_loader(lp):
    cid = os.path.basename(lp).replace(".nii.gz", "")
    base = lp.replace("labelsTr", "imagesTr").replace("Validation-Public-Labels", "Validation-Public-Images")
    img = base.replace(".nii.gz", "_0000.nii.gz")
    ct = nib.load(img).get_fdata(); seg = np.asarray(nib.load(lp).dataobj).astype(np.uint8)
    return ct, seg, cid
ft2_labels = (sorted(glob.glob("/scratch/ud3d4/acm_data/FLARE_Task2/train_gt_label/labelsTr/*.nii.gz")) +
              sorted(glob.glob("/scratch/ud3d4/acm_data/FLARE_Task2/validation/Validation-Public-Labels/*.nii.gz")))
for organ, labs in FLARE_LAB.items():
    run_source("flare_task2", ft2_labels, ft2_loader, labs, 2, organ)

# ---- FLARE23: reuse the byte-extracted flare_organ_pool (950 cases, l/k/p slices already prepared) ----
fop = "/scratch/ud3d4/acm_data/flare_organ_pool"
if os.path.exists(f"{fop}/meta.json"):
    fi = np.load(f"{fop}/images.npy"); fm = np.load(f"{fop}/masks.npy"); fmeta = json.load(open(f"{fop}/meta.json"))
    cnt = Counter()
    for i in range(len(fmeta)):
        o = fmeta[i]["organ"]
        if cnt[o] >= CAP:
            continue
        imgs.append(fi[i]); masks.append((fm[i] > 0).astype(np.uint8))
        meta.append({"organ": o, "dataset": "flare23", "case": fmeta[i]["case"]})
        cnt[o] += 1
    for o in ("liver", "kidney", "pancreas"):
        print(f"{'flare23':12s} {o:9s}: {cnt[o]:5d} slices (reused flare_organ_pool)", flush=True)
else:
    print("flare23: flare_organ_pool not built yet — skipped (run extract_flare_organ_slices.py first)", flush=True)

# ---- KiTS kidney: reuse the already-extracted kidney pool ----
kp = "/scratch/ud3d4/acm_data/kits_kidney_pool"
if os.path.exists(f"{kp}/meta.json"):
    ki = np.load(f"{kp}/images.npy"); km = np.load(f"{kp}/masks.npy"); kmeta = json.load(open(f"{kp}/meta.json"))
    keep = min(len(kmeta), CAP)
    for i in range(keep):
        imgs.append(ki[i]); masks.append((km[i] > 0).astype(np.uint8))
        meta.append({"organ": "kidney", "dataset": "kits", "case": kmeta[i]["case"]})
    print(f"{'kits':12s} {'kidney':9s}: {keep} slices (reused kits_kidney_pool)", flush=True)

# ---- save ----
np.save(f"{OUT}/images.npy", np.stack(imgs).astype(np.uint8))
np.save(f"{OUT}/masks.npy", np.stack(masks).astype(np.uint8))
json.dump(meta, open(f"{OUT}/meta.json", "w"))
print("\n=== ORGAN POOL (liver/kidney/pancreas) ===", flush=True)
by = Counter((m["organ"], m["dataset"]) for m in meta)
for (o, d), c in sorted(by.items()):
    print(f"  {o:9s} {d:12s} {c:6d}", flush=True)
print(f"TOTAL {len(meta)} slices | per organ: {dict(Counter(m['organ'] for m in meta))} | "
      f"patients: {len({(m['dataset'],m['case']) for m in meta})}", flush=True)
print(f"-> {OUT}", flush=True)
