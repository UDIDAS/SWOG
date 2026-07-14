#!/usr/bin/env python3
"""
Hand-off metrics v2 — addresses reviewer items #1, #4, #5, #6.

On the HELD-OUT TEST split (57 Pancreas + 27 LiTS cases), run SAM3 in two prompt
modes and compute the full metric set per case:
  - gtbox   : GT-derived box (SEMI-ORACLE, reproduces the delivered masks)
  - fullbox : full-image box (AUTONOMOUS w.r.t. localization -- the "GT box removed" number)
Metrics: Dice, HD95 (mm), NSD@2mm, sensitivity, specificity.  Pancreas = organ(1)+tumor(2);
LiTS = tumor(2) only (liver is not predicted -> tumor-only, per reviewer item #2).
"""
import os, sys, json
sys.path.insert(0, "/home/ud3d4/Desktop/SWOG/src/scripts")
import numpy as np, torch, nibabel as nib
from torch.amp import autocast
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from transformers import Sam3Model, Sam3Processor
from monai.metrics import compute_hausdorff_distance, compute_surface_dice

from infer_sam3 import SAM3_BASE, SAM3_REVISION, HF_TOKEN, hu_to_rgb, bbox_from_mask, best_mask, HU_WINDOW

PANC_IMG = "/scratch/ud3d4/acm_data/Pancreas/imagesTr"
PANC_LBL = "/scratch/ud3d4/acm_data/Pancreas/labelsTr"
LITS = "/scratch/ud3d4/acm_data/Data"
CKPT = "/scratch/ud3d4/acm_data/sam3_ckpts"
OUT = "/home/ud3d4/Desktop/SWOG/handoff/handoff_metrics_v2.json"
DEV = torch.device("cuda:0")


def dice(p, g):
    i = float((p & g).sum()); s = float(p.sum() + g.sum())
    return (2 * i / s) if s else float("nan")


def sens_spec(p, g):
    tp = float((p & g).sum()); fn = float((~p & g).sum())
    tn = float((~p & ~g).sum()); fp = float((p & ~g).sum())
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    return sens, spec


def surface(p, g, spacing):
    if p.sum() == 0 or g.sum() == 0:
        return float("nan"), float("nan")
    pt = torch.from_numpy(p[None, None].astype(np.uint8))
    gt = torch.from_numpy(g[None, None].astype(np.uint8))
    try:
        hd = float(compute_hausdorff_distance(pt, gt, percentile=95, spacing=spacing))
        nsd = float(compute_surface_dice(pt, gt, class_thresholds=[2.0], spacing=spacing, include_background=False))
    except Exception:
        hd, nsd = float("nan"), float("nan")
    return hd, nsd


def load_models():
    proc = Sam3Processor.from_pretrained(SAM3_BASE, token=HF_TOKEN, revision=SAM3_REVISION)
    models = {}
    for name, f in [("panc_organ", "sam3_pancreas_organ_v3.pt"),
                    ("panc_tumor", "sam3_pancreas_tumor_v3.pt"),
                    ("lits_tumor", "sam3_lits_tumor_v3.pt")]:
        m = Sam3Model.from_pretrained(SAM3_BASE, token=HF_TOKEN, revision=SAM3_REVISION)
        st = torch.load(f"{CKPT}/{f}", map_location=DEV, weights_only=False)
        m.load_state_dict({k.replace("module.", ""): v for k, v in st.items()}, strict=False)
        models[name] = m.to(DEV).eval()
    return proc, models


def infer(model, proc, rgb, box):
    inp = proc(images=[rgb], text=["visual"], input_boxes=[[box]], input_boxes_labels=[[1]], return_tensors="pt")
    kw = {"pixel_values": inp["pixel_values"].to(DEV)}
    for k in ("input_ids", "attention_mask", "input_boxes", "input_boxes_labels"):
        if inp.get(k) is not None:
            kw[k] = inp[k].to(DEV)
    with torch.no_grad(), autocast("cuda"):
        out = model(**kw)
        m = torch.nn.functional.interpolate(best_mask(out).float(), (256, 256), mode="bilinear", align_corners=False)
    return m.sigmoid().squeeze().cpu().numpy()


def pred_volume(model, proc, ct, gt, label, lo, hi, mode):
    """Return predicted binary volume for one label, in ct-shape."""
    pred = np.zeros(ct.shape, dtype=np.uint8)
    Z = ct.shape[2]
    for z in range(Z):
        g = (gt[:, :, z] == label)
        if g.sum() < 5:
            continue  # slice-selection (both modes) -- matches delivered protocol
        H, W = ct[:, :, z].shape
        rgb = hu_to_rgb(resize(ct[:, :, z], (256, 256), preserve_range=True, anti_aliasing=True), lo, hi).astype(np.uint8)
        if mode == "gtbox":
            g256 = resize(g.astype(np.uint8), (256, 256), order=0, preserve_range=True).astype(np.uint8)
            box = bbox_from_mask(g256, 3)
            if box is None:
                continue
        else:
            box = [0, 0, 255, 255]
        prob = infer(model, proc, rgb, box)
        pred[:, :, z] = (resize(prob, (H, W), order=1, preserve_range=True) > 0.5).astype(np.uint8)
    return pred


def case_metrics(pred, gt_bin, spacing):
    p, g = pred.astype(bool), gt_bin.astype(bool)
    d = dice(p, g); se, sp = sens_spec(p, g); hd, nsd = surface(p, g, spacing)
    return {"dice": None if d != d else round(d, 4),
            "hd95_mm": None if hd != hd else round(hd, 2),
            "nsd_2mm": None if nsd != nsd else round(nsd, 4),
            "sensitivity": None if se != se else round(se, 4),
            "specificity": None if sp != sp else round(sp, 6)}


def run():
    proc, models = load_models()
    results = {"pancreas": {"gtbox": {}, "fullbox": {}}, "lits": {"gtbox": {}, "fullbox": {}}}

    # ---- Pancreas test split ----
    cids = sorted(f.replace(".nii.gz", "") for f in os.listdir(PANC_IMG)
                  if f.endswith(".nii.gz") and not f.startswith("._"))
    _, test = train_test_split(cids, test_size=0.2, random_state=42)
    print(f"Pancreas test: {len(test)} cases", flush=True)
    for i, cid in enumerate(test):
        ni = nib.load(f"{PANC_IMG}/{cid}.nii.gz"); ct = ni.get_fdata()
        gt = nib.load(f"{PANC_LBL}/{cid}.nii.gz").get_fdata().astype(np.uint8)
        sp = tuple(float(s) for s in ni.header.get_zooms()[:3])
        lo, hi = HU_WINDOW["pancreas"]
        for mode in ("gtbox", "fullbox"):
            po = pred_volume(models["panc_organ"], proc, ct, gt, 1, lo, hi, mode)
            pt = pred_volume(models["panc_tumor"], proc, ct, gt, 2, lo, hi, mode)
            results["pancreas"][mode][cid] = {
                "organ": case_metrics(po, gt == 1, sp),
                "tumor": case_metrics(pt, gt == 2, sp)}
        if (i + 1) % 10 == 0:
            print(f"  panc {i+1}/{len(test)}", flush=True)

    # ---- LiTS test split (tumor-only) ----
    vids = sorted(int(f.replace("volume-", "").replace(".npy", "")) for f in os.listdir(f"{LITS}/ct")
                  if f.startswith("volume-") and not f.startswith("._"))
    _, testv = train_test_split(vids, test_size=0.2, random_state=42)
    print(f"LiTS test: {len(testv)} cases", flush=True)
    for i, vid in enumerate(testv):
        ct = np.load(f"{LITS}/ct/volume-{vid}.npy").transpose(1, 2, 0)   # (256,256,Z)
        seg = np.load(f"{LITS}/seg/segmentation-{vid}.npy").transpose(1, 2, 0).astype(np.uint8)
        lo, hi = HU_WINDOW["lits"]; sp = (1.0, 1.0, 1.0)
        for mode in ("gtbox", "fullbox"):
            pt = pred_volume(models["lits_tumor"], proc, ct, seg, 2, lo, hi, mode)
            results["lits"][mode][f"LiTs-{vid:03d}"] = {"tumor": case_metrics(pt, seg == 2, sp)}
        if (i + 1) % 10 == 0:
            print(f"  lits {i+1}/{len(testv)}", flush=True)

    # ---- aggregate ----
    def agg(cases, key):
        vals = {m: [] for m in ["dice", "hd95_mm", "nsd_2mm", "sensitivity", "specificity"]}
        for c in cases.values():
            for m in vals:
                v = c[key].get(m)
                if v is not None and v == v:  # filter None and NaN
                    vals[m].append(v)
        return {m: (round(float(np.mean(v)), 4) if v else None) for m, v in vals.items()}

    summary = {}
    for ds, labels in [("pancreas", ["organ", "tumor"]), ("lits", ["tumor"])]:
        summary[ds] = {}
        for mode in ("gtbox", "fullbox"):
            summary[ds][mode] = {lab: agg(results[ds][mode], lab) for lab in labels}

    out = {"note": "Held-out TEST split. gtbox=semi-oracle (delivered); fullbox=autonomous (GT box removed). "
                   "LiTS tumor-only (liver not predicted). HD95/NSD via monai with native spacing.",
           "test_summary": summary, "per_case": results}
    json.dump(out, open(OUT, "w"), indent=2)
    print("\n=== TEST-SPLIT SUMMARY (Dice) ===")
    for ds in ("pancreas", "lits"):
        for lab in (["organ", "tumor"] if ds == "pancreas" else ["tumor"]):
            g = summary[ds]["gtbox"][lab]["dice"]; f = summary[ds]["fullbox"][lab]["dice"]
            print(f"  {ds} {lab:6s}: oracle(gtbox)={g}   autonomous(fullbox)={f}")
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    run()
