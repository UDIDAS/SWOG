#!/usr/bin/env python3
"""Turn the patient-level 3-D AUSAM results into the KG-stage findings — no mask re-run, no GPU.

Reads results/ausam_3d_<dataset>.json and produces three things (the KG contribution, lightweight):
  1. Patient-level 3-D Dice per dataset/organ (the honest whole-volume, semi-oracle number).
  2. Node fidelity — predicted vs GT organ volume (correlation + MAE), i.e. is a KG built from AUSAM's
     predicted phenotypes as trustworthy as one from GT?
  3. GT-free validation — flag predicted volumes that fall outside the KG atlas plausibility band
     [p2.5, p97.5], and test whether "flagged" tracks "actually-low 3-D Dice" (AUROC when both classes exist).

-> results/ausam_3d_summary.json  (+ printed table)
"""
import json
import glob
import numpy as np

RES = "/home/ud3d4/Desktop/SWOG/results"
ATLAS = json.load(open("/home/ud3d4/Desktop/SWOG/kg/data/kg_atlas.json"))["organs"]
LOWDICE = 0.85          # a predicted mask below this 3-D Dice is "actually poor"


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and np.std(a) > 0 and np.std(b) > 0 else None


def main():
    files = sorted(glob.glob(f"{RES}/ausam_3d_*.json"))
    files = [f for f in files if "summary" not in f]
    out = {"datasets": {}, "node_fidelity": {}, "gt_free_validation": {}}
    # collect per (dataset, organ)
    pairs = {}          # (ds,organ) -> list of (dice, vol_pred_cm3|None, vox_pred, vox_gt)
    for f in files:
        d = json.load(open(f)); ds = d["dataset"]
        out["datasets"][ds] = {"n_patients": d["n_patients"], "mean_3d_dice": d["mean_3d_dice"],
                               "mean_nsd_2mm": d.get("mean_nsd_2mm", {})}
        for c in d["cases"]:
            for organ, o in c["organs"].items():
                pairs.setdefault((ds, organ), []).append(
                    (o["dice3d"], o.get("vol_pred_cm3"), o.get("vol_gt_cm3"),
                     o.get("vox_pred", 0), o.get("vox_gt", 0)))

    # 2. node fidelity (pred vs GT volume) — cm3 if available else voxels
    for (ds, organ), rows in sorted(pairs.items()):
        has_cm3 = all(r[1] is not None for r in rows)
        vp = [r[1] if has_cm3 else r[3] for r in rows]
        vg = [r[2] if has_cm3 else r[4] for r in rows]
        unit = "cm3" if has_cm3 else "voxels"
        mae = float(np.mean(np.abs(np.array(vp) - np.array(vg))))
        mape = float(np.mean(np.abs(np.array(vp) - np.array(vg)) / (np.array(vg) + 1e-6)) * 100)
        out["node_fidelity"][f"{organ}/{ds}"] = {
            "n": len(rows), "unit": unit, "volume_corr": round(corr(vp, vg), 4) if corr(vp, vg) else None,
            "volume_MAE": round(mae, 2), "volume_MAPE_pct": round(mape, 1)}

    # 3. GT-free validation via atlas band [p2.5, p97.5] on predicted cm3 volume
    for (ds, organ), rows in sorted(pairs.items()):
        band = ATLAS.get(organ, {}).get("volume_cm3")
        rows_cm3 = [r for r in rows if r[1] is not None]
        if not band or not rows_cm3:
            continue
        lo, hi = band["p2.5"], band["p97.5"]
        flagged = np.array([not (lo <= r[1] <= hi) for r in rows_cm3])   # implausible predicted volume
        poor = np.array([r[0] < LOWDICE for r in rows_cm3])              # actually-low 3-D Dice
        auroc = None
        if poor.any() and (~poor).any():
            try:
                from sklearn.metrics import roc_auc_score
                # score = distance outside band (0 if inside) — higher = more implausible
                score = np.array([max(lo - r[1], 0, r[1] - hi) for r in rows_cm3], float)
                auroc = round(float(roc_auc_score(poor, score)), 3)
            except Exception:
                pass
        out["gt_free_validation"][f"{organ}/{ds}"] = {
            "n": len(rows_cm3), "band_cm3": [lo, hi], "n_flagged": int(flagged.sum()),
            "n_poor_dice(<%.2f)" % LOWDICE: int(poor.sum()),
            "flagged_and_poor": int((flagged & poor).sum()), "auroc_flag_vs_poor": auroc}

    json.dump(out, open(f"{RES}/ausam_3d_summary.json", "w"), indent=2)
    print("=== patient-level 3-D Dice ===")
    for ds, v in out["datasets"].items():
        print(f"  {ds:12s} n={v['n_patients']:3d}  {v['mean_3d_dice']}")
    print("\n=== node fidelity (predicted vs GT organ volume) ===")
    for k, v in out["node_fidelity"].items():
        print(f"  {k:20s} n={v['n']:3d}  corr={v['volume_corr']}  MAPE={v['volume_MAPE_pct']}% ({v['unit']})")
    print("\n=== GT-free validation (atlas-band flag vs actual low-Dice) ===")
    for k, v in out["gt_free_validation"].items():
        print(f"  {k:20s} n={v['n']:3d}  flagged={v['n_flagged']}  poor={v['n_poor_dice(<0.85)']}  auroc={v['auroc_flag_vs_poor']}")
    print(f"\n-> {RES}/ausam_3d_summary.json")


if __name__ == "__main__":
    main()
