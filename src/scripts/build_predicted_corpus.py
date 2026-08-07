#!/usr/bin/env python3
"""PREDICTED-phenotype corpus from the AUSAM pipeline — the input to the OAKG retrieval-on-predicted experiment.

For each tumor-test patient: run the per-dataset ORGAN + TUMOR AUSAM models over the full volume (semi-oracle
GT box), assemble 3-D organ + tumor masks, and derive the SAME phenotype schema the GT corpus uses:
  organ_volume_cm3, has_tumor, tumor_volume_cm3, burden_cat, multiplicity, containment, anatomic_location, size_cat.
Mirrors kg_extract_phenotypes.py (voxel-tertile burden bins, ndimage.label multiplicity, containment, centroid
location) so predicted records are directly comparable to corpus_perpatient.json.

  python build_predicted_corpus.py --dataset msd|lits|kits [--limit N]
-> results/corpus_predicted_<dataset>.json
"""
import argparse
import json
import sys
from collections import defaultdict

import numpy as np
import torch
from torch.amp import autocast
from scipy import ndimage
from skimage.transform import resize

sys.path.insert(0, "/home/ud3d4/Desktop/SWOG/src/scripts")
from eval_ausam_3d import hu_rgb, _slice, WIN, MINPX, load_msd, load_lits, load_kits
from train_tumor_incremental import patient_split as tumor_split, load_all as tumor_load_all
from run_pancreas_sam3 import bbox_from_mask, extract_best_mask_soft, _load_sam3_ckpt, SAM3_MODEL_ID, HF_TOKEN

POOL = "/scratch/ud3d4/acm_data/organ_pool_lkp"
TPOOL = "/scratch/ud3d4/acm_data/tumor_pool"
DEV = "cuda:0"

# single-organ tumor datasets: organ + tumor label in the volume, models, loader, slice axis
# "tds" = dataset key in the TUMOR pool (MSD tumors are keyed "pancreas" there, not "msd")
CFG = {
    "msd":  {"organ": "pancreas", "tds": "pancreas", "olab": [1], "tlab": [2], "ax": 2, "load": load_msd,
             "om": f"{POOL}/sam3_organ_generic_ausam_msd.pth", "tm": f"{TPOOL}/sam3_tumor_ausam_pancreas.pth"},
    "lits": {"organ": "liver", "tds": "lits", "olab": [1], "tlab": [2], "ax": 0, "load": load_lits,
             "om": f"{POOL}/sam3_organ_generic_ausam_lits.pth", "tm": f"{TPOOL}/sam3_tumor_ausam_lits.pth"},
    "kits": {"organ": "kidney", "tds": "kits", "olab": [1], "tlab": [2], "ax": 0, "load": load_kits,
             "om": f"{POOL}/sam3_organ_generic_ausam_kits.pth", "tm": f"{TPOOL}/sam3_tumor_ausam_kits.pth"},
}


def burden_cat(v, t1, t2):
    return "low" if v < t1 else ("high" if v >= t2 else "medium")


def predict_volume(model, proc, ct, seg, labs, ax, text):
    """Run a model over the labeled slices (semi-oracle GT box) -> 3-D binary mask at native resolution."""
    pred = np.zeros(ct.shape, np.uint8)
    for z in range(ct.shape[ax]):
        gm = np.isin(_slice(seg, ax, z), labs)
        gm256 = resize(gm.astype(float), (256, 256), order=0, preserve_range=True) > 0.5
        if gm256.sum() < MINPX:
            continue
        rgb = hu_rgb(resize(_slice(ct, ax, z), (256, 256), preserve_range=True, anti_aliasing=True))
        box = bbox_from_mask(gm256.astype(np.uint8), pad=3) or [0, 0, 255, 255]
        inp = proc(images=[rgb], text=[text], input_boxes=[[box]], input_boxes_labels=[[1]], return_tensors="pt")
        kw = {"pixel_values": inp["pixel_values"].to(DEV)}
        for k in ("input_ids", "attention_mask", "input_boxes", "input_boxes_labels"):
            if inp.get(k) is not None:
                kw[k] = inp[k].to(DEV)
        with torch.no_grad(), autocast("cuda"):
            out = model(**kw)
            pm = extract_best_mask_soft(out.pred_masks, out.pred_logits)
            pm = torch.nn.functional.interpolate(pm.float(), size=(256, 256), mode="bilinear", align_corners=False)
        pr = pm.sigmoid().squeeze().cpu().numpy() > 0.5
        s = resize(pr.astype(float), _slice(ct, ax, z).shape, order=0, preserve_range=True) > 0.5
        if ax == 0:
            pred[z] = s
        else:
            pred[:, :, z] = s
    return pred.astype(bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gt", action="store_true", help="use GT masks (identical phenotype method) for the matched index")
    a = ap.parse_args()
    cfg = CFG[a.dataset]
    from transformers import Sam3Processor
    proc = Sam3Processor.from_pretrained(SAM3_MODEL_ID, token=HF_TOKEN)

    # tumor-test patients for this dataset (same seed-42 tumor split)
    Mt = tumor_load_all()[2]
    _, _, te = tumor_split(Mt)
    cases = sorted({Mt[i]["case"] for i in te if Mt[i]["dataset"] == cfg["tds"]})
    if a.limit:
        cases = cases[:a.limit]
    print(f"predicted corpus [{a.dataset}] {len(cases)} tumor-test patients", flush=True)

    om = tm = None
    if not a.gt:
        om = _load_sam3_ckpt(cfg["om"], DEV); tm = _load_sam3_ckpt(cfg["tm"], DEV)
    organ = cfg["organ"]; ax = cfg["ax"]

    raw = []                             # (case, spacing_cm3, organ_vox, tumor_mask, organ_mask, seg)
    for ci, case in enumerate(cases):
        try:
            ct, seg, sp = cfg["load"](case)
        except Exception as e:
            print(f"  [{ci+1}] {case} SKIP ({type(e).__name__})", flush=True); continue
        if a.gt:
            omask = np.isin(seg, cfg["olab"]); tmask = np.isin(seg, cfg["tlab"])
        else:
            omask = predict_volume(om, proc, ct, seg, cfg["olab"], ax, organ)
            tmask = predict_volume(tm, proc, ct, seg, cfg["tlab"], ax, "tumor")
        vox_cm3 = (sp[0] * sp[1] * sp[2] / 1000.0) if sp else 1.0 / 1000     # voxel tertiles are spacing-invariant
        raw.append([case, vox_cm3, int(omask.sum()), tmask, omask])
        if (ci + 1) % 5 == 0 or ci == 0:
            print(f"  [{ci+1}/{len(cases)}] {case}: organ {int(omask.sum())}vox tumor {int(tmask.sum())}vox", flush=True)
    if not a.gt:
        del om, tm; torch.cuda.empty_cache()

    # burden bins from predicted-tumor-volume tertiles (voxel-count, spacing-invariant within dataset)
    tvols = sorted(int(r[3].sum()) for r in raw if r[3].sum() > 0)
    t1 = tvols[len(tvols) // 3] if tvols else 1
    t2 = tvols[2 * len(tvols) // 3] if tvols else 2

    records = []
    for case, vox_cm3, ovox, tmask, omask in raw:
        tvox = int(tmask.sum()); has_tumor = tvox > 0
        mult = "multifocal" if (has_tumor and ndimage.label(tmask)[1] >= 2) else ("solitary" if has_tumor else "none")
        inside = int((tmask & omask).sum())
        contain = ("contained" if (has_tumor and tvox and inside / tvox >= 0.90) else ("boundary" if has_tumor else "none"))
        # anatomic location from tumor centroid offset along the organ's long axis
        loc = "na"
        if has_tumor:
            zc = np.where(tmask.any(axis=tuple(i for i in range(3) if i != ax)))[0]
            oc = np.where(omask.any(axis=tuple(i for i in range(3) if i != ax)))[0]
            if len(zc) and len(oc):
                off = (zc.mean() - oc.mean()) / (oc.max() - oc.min() + 1e-6)
                loc = "head" if off > 0.12 else ("tail" if off < -0.12 else "body")
        records.append({"case_id": case, "dataset": a.dataset, "observed_organs": [organ], "granularity": "volume",
                        "organs": {organ: {"present": True, "organ_volume_cm3": round(ovox * vox_cm3, 2),
                                           "has_tumor": has_tumor, "tumor_volume_cm3": round(tvox * vox_cm3, 2),
                                           "tumor_voxels": tvox,
                                           "burden_cat": burden_cat(tvox, t1, t2) if has_tumor else "none",
                                           "multiplicity": mult, "containment": contain,
                                           "anatomic_location": loc, "size_cat": "unknown"}}})
    fp = f"/home/ud3d4/Desktop/SWOG/results/corpus_{'gt' if a.gt else 'predicted'}_{a.dataset}.json"
    json.dump({"n": len(records), "dataset": a.dataset, "records": records}, open(fp, "w"), indent=1)
    from collections import Counter
    print(f"burden {dict(Counter(r['organs'][organ]['burden_cat'] for r in records))} | "
          f"mult {dict(Counter(r['organs'][organ]['multiplicity'] for r in records))} | "
          f"contain {dict(Counter(r['organs'][organ]['containment'] for r in records))}", flush=True)
    print(f"-> {fp}", flush=True)


if __name__ == "__main__":
    main()
