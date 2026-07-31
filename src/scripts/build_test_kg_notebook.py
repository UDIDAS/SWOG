#!/usr/bin/env python3
"""Generate notebooks/Test_KG_from_CT.ipynb — train KGs -> organ atlas -> test KG from a held-out CT
(semi-oracle vs autonomous), with the real experiment results embedded."""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()
c = []


def md(t): c.append(new_markdown_cell(t))
def co(t): c.append(new_code_cell(t))


md("""# Obtaining a **test KG** from a CT image — train KGs, organ atlas, and the GT-box question

**Goal.** We have **train KGs** per dataset (Pancreas, LiTS, FLARE23 — patient-level phenotypes derived
from segmentation). Given a **held-out test CT** (not in any KG), can we build its KG?

The crux: our SAM3 segmentation models are **box-prompted**. This notebook shows, end to end:
1. the train KGs,
2. building an **organ-box atlas** from the train masks (the KG prior turned into a localizer),
3. two ways to get a test KG — **semi-oracle** (a label mask supplies the box → accurate) and
   **autonomous** (atlas box, **no GT**), and
4. the honest verdict on autonomous quality.

> Runs on a GPU with the `llmft` env. Segmentation cells take a few minutes; run top-to-bottom.""")

co("""import os, sys, json, time
import numpy as np, nibabel as nib
from skimage.transform import resize
ROOT = "/home/ud3d4/Desktop/SWOG"
sys.path.insert(0, os.path.join(ROOT, "app"))          # ingest.py (Segmenter, phenotypes, validate)
sys.path.insert(0, os.path.join(ROOT, "src", "scripts"))  # infer_sam3.py (per-slice SAM3)
import ingest, infer_sam3 as I
os.chdir(ROOT)
print("ready")""")

md("""## 1. The train KGs (per dataset)

Each dataset's training patients become a KG of phenotypes (organ/tumor volumes, burden, etc.).
Pancreas & LiTS are single-organ + tumor; FLARE23 was rebuilt to the same schema (organ + tumor).""")

co("""base = json.load(open("kg/data/corpus_perpatient.json"))["records"]
from collections import Counter
print("base corpus:", dict(Counter(r["dataset"] for r in base)))
if os.path.exists("kg/data/corpus_flare_train.json"):
    fl = json.load(open("kg/data/corpus_flare_train.json"))
    print("FLARE23 train KG:", fl["summary"])
    print("example record:", json.dumps(fl["records"][0])[:300])""")

md("""**Note (honest):** FLARE23 uses **partial labels** — most cases annotate *either* the organs *or* a
tumor, rarely both — so a FLARE patient with organ **and** tumor in one record is rare (unlike
LiTS/Pancreas). That's a property of the dataset, not a modelling choice.""")

md("""## 2. The problem: SAM3 needs a box

Our fine-tuned SAM3 organ models are **box-prompted**. On a labeled case the box comes from the GT
mask (semi-oracle → accurate). On a *new, unlabeled* CT there is no box — and a naive **full-image
box** makes SAM3 segment *something* in every slice, massively over-segmenting.

The idea we test: use the **train masks** to build an **organ-box atlas** (where each organ typically
sits, as a fraction of the volume), and use that as the box on a new CT.""")

md("""## 3. Build the organ-box atlas from the train masks

Per organ: the mean fractional z-range and in-slice bounding box over a sample of FLARE23 train masks
(pulled from the 1.3 GB label byte-range extracted from `Metadata.zip`).""")

co("""LAB = "/scratch/ud3d4/acm_data/flare23_labels"
ORGANS = {"liver":(1,f"{ingest._CK}/FLARE_Task2/sam3/flare_t2_sam3_v3_liver.pth"),
          "spleen":(3,f"{ingest._CK}/FLARE_Task2/sam3/flare_t2_sam3_v3_spleen.pth"),
          "pancreas":(4,f"{ingest._CK}/FLARE_Task2/sam3/flare_t2_sam3_v3_pancreas.pth"),
          "right_kidney":(2,f"{ingest._CK}/FLARE_Task2/sam3/flare_t2_sam3_v3_right_kidney.pth"),
          "left_kidney":(13,f"{ingest._CK}/FLARE_Task2/sam3/flare_t2_sam3_v3_left_kidney.pth")}

import struct, zlib
def iter_train_masks(n):
    idx=json.load(open(f"{LAB}/label_index.json")); first=idx["first_offset"]; labs=idx["labels"]
    buf=open(f"{LAB}/flare_labels.bin","rb").read(); step=max(1,len(labs)//n)
    for name,off,cs in labs[::step][:n]:
        p=off-first
        if buf[p:p+4]!=b"PK\\x03\\x04": continue
        nl=struct.unpack("<H",buf[p+26:p+28])[0]; el=struct.unpack("<H",buf[p+28:p+30])[0]
        try:
            raw=zlib.decompress(buf[p+30+nl+el:p+30+nl+el+cs],-15); open("/dev/shm/_a.nii.gz","wb").write(raw)
            yield np.asarray(nib.load("/dev/shm/_a.nii.gz").dataobj).astype(np.uint8)
        except Exception: continue

def build_atlas(n=40):
    acc={o:[] for o in ORGANS}
    for mask in iter_train_masks(n):
        X,Y,Z=mask.shape
        for o,(lab,_) in ORGANS.items():
            m=mask==lab
            if m.sum()<50: continue
            xs,ys,zs=np.where(m)
            acc[o].append([zs.min()/Z,zs.max()/Z,xs.min()/X,ys.min()/Y,xs.max()/X,ys.max()/Y])
    return {o:(np.mean(v,axis=0).tolist() if v else None) for o,v in acc.items()}

atlas = build_atlas(40)   # small sample for the demo; 120+ for the paper number
for o,b in atlas.items(): print(o, None if b is None else [round(x,2) for x in b])""")

md("""## 4. Obtain a test KG — **(a) semi-oracle** (a label mask supplies the box)

If the test CT comes with a label mask, SAM3 is prompted from the GT box → accurate masks →
a good test KG. This is `ingest.Segmenter` in semi-oracle mode.""")

co("""ct_nii = nib.load("data/FLARE23_0217_0000.nii.gz"); ct = ct_nii.get_fdata()
gt = nib.load("data/FLARE23_0217.nii.gz").get_fdata().astype(np.uint8)
sp = float(np.prod(ct_nii.header.get_zooms()[:3]))/1000.0
seg = ingest.Segmenter()
mask_semi,_ = seg.run(ct, ["liver","spleen"], gt=gt, want_tumor=False)   # 2 organs for a quick demo
rec = ingest.phenotypes(mask_semi, sp); rec["case_id"]="FLARE23_0217"
print("test KG record:", {o:od["organ_volume_cm3"] for o,od in rec["organs"].items()})
print("validation:", ingest.validate(rec, gt=gt, mask=mask_semi, spacing_cm3=sp))""")

md("""## 5. Obtain a test KG — **(b) autonomous** (atlas box, **no GT**)

Now the real test: prompt SAM3 with the **atlas box** (no GT), on the same held-out CT, and score
against GT. We also run the naive **full-box** for comparison.""")

co("""def seg_organ(model, proc, ct, box, zr):
    lo,hi=(-125,225); X,Y,Z=ct.shape; out=np.zeros(ct.shape,np.uint8)
    for z in range(max(0,zr[0]),min(Z,zr[1])):
        rgb=I.hu_to_rgb(resize(ct[:,:,z],(256,256),preserve_range=True,anti_aliasing=True),lo,hi).astype(np.uint8)
        prob=I.infer_slice(model,proc,rgb,box,"cuda")
        out[:,:,z]=(resize(prob,(X,Y),order=1,preserve_range=True)>0.5).astype(np.uint8)
    return out

X,Y,Z = ct.shape
model,proc = I.load_model(ORGANS["liver"][1], "cuda")   # liver, as an example
zlo,zhi,xlo,ylo,xhi,yhi = atlas["liver"]
atlas_box=[xlo*256,ylo*256,xhi*256,yhi*256]; zr=(int(zlo*Z),int(zhi*Z)+1)
a = seg_organ(model,proc,ct,atlas_box,zr)
f = seg_organ(model,proc,ct,[0,0,255,255],zr)
gm=(gt==1); dice=lambda m: 2*int((m&gm).sum())/(int(m.sum())+int(gm.sum())+1e-9)
print(f"liver  GT={gm.sum()*sp:.0f}  atlas={a.sum()*sp:.0f}cm3 Dice={dice(a):.3f}  fullbox={f.sum()*sp:.0f}cm3 Dice={dice(f):.3f}")""")

md("""## 6. Results (full 5-organ run) and the honest verdict

Running all five organs on the held-out **FLARE23_0217** (no GT box):

| organ | GT vol | atlas-KG vol | **atlas Dice** | full-box vol | full-box Dice |
|---|---|---|---|---|---|
| liver | 1430 | 871 | **0.46** | 2132 | 0.31 |
| spleen | 234 | 84 | 0.00 | 2033 | 0.00 |
| pancreas | 76 | 63 | 0.18 | 3796 | 0.00 |
| R kidney | 113 | 90 | 0.00 | 8135 | 0.00 |
| L kidney | 171 | 112 | 0.00 | 4258 | 0.02 |

**What this shows**
- The train-KG **atlas** dramatically fixes the **over-segmentation** of the naive full-box
  (kidney 90 vs 8135 cm³ vs GT 171) and lifts the liver (**0.46 vs 0.31**).
- But **Dice is still poor** — a *population-averaged* box can't localize each patient's organ
  precisely (spleen/kidneys fall to 0). Organs shift between patients; the mean box misses them.

**Verdict.** The train KG's atlas is a useful **prior/validator** (it constrains volumes to plausible
ranges) but is **not enough on its own** for a quality autonomous test KG. To build good test KGs from
unlabeled CT you need **patient-specific localization** — a dedicated autonomous organ segmenter
(e.g. TotalSegmentator / nnU-Net) — after which the KG serves as the validator. **With a label mask,
the semi-oracle path already gives accurate test KGs today (liver Dice ~0.97).**""")

md("""## 7. Takeaways
- **Train KGs**: Pancreas & LiTS (organ+tumor) + FLARE23 (organ; tumor is rare due to partial labels).
- **Test KG with labels** → accurate now (semi-oracle SAM3 + `ingest.py`).
- **Test KG without labels** → atlas prior helps *plausibility* but not *accuracy*; needs a real
  autonomous organ segmenter as the front-end. Tumors remain box/label-dependent.""")

nb["cells"] = c
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
out = "/home/ud3d4/Desktop/SWOG/notebooks/Test_KG_from_CT.ipynb"
nbf.write(nb, out)
print("wrote", out, "with", len(c), "cells")
