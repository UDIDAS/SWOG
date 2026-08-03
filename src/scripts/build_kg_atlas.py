#!/usr/bin/env python3
"""Build the KG anatomical atlas — per-organ phenotype priors learned from the whole cohort. This is the
bridge that lets the knowledge graph HELP segmentation (not just validate/evolve): plausible size/diameter
ranges per organ, pooled across every dataset that observes that organ. KiTS enriches the kidney priors,
LiTS the liver, MSD the pancreas, FLARE all of them — the more the KG grows, the sharper the priors.

Used by KG-guided segmentation to (a) reject/repair implausible masks and (b) set expected extent.
out: kg/data/kg_atlas.json
"""
import json
from collections import defaultdict

import numpy as np

D = "/home/ud3d4/Desktop/SWOG/kg/data"


def pct(v, p):
    return round(float(np.percentile(v, p)), 1)


def main():
    recs = json.load(open(f"{D}/corpus_perpatient.json"))["records"]
    vol, dia, srcs = defaultdict(list), defaultdict(list), defaultdict(set)
    tum = defaultdict(list)
    for r in recs:
        for o, od in r["organs"].items():
            if od.get("organ_volume_cm3"):
                vol[o].append(od["organ_volume_cm3"]); srcs[o].add(r["dataset"])
            if od.get("organ_max_diameter_mm"):
                dia[o].append(od["organ_max_diameter_mm"])
            if od.get("has_tumor") and od.get("tumor_volume_cm3"):
                tum[o].append(od["tumor_volume_cm3"])

    atlas = {}
    for o in sorted(vol):
        v = np.array(vol[o])
        entry = {"n": len(v), "sources": sorted(srcs[o]),
                 "volume_cm3": {"p1": pct(v, 1), "p2.5": pct(v, 2.5), "median": pct(v, 50),
                                "p97.5": pct(v, 97.5), "p99": pct(v, 99),
                                "mean": round(float(v.mean()), 1), "std": round(float(v.std()), 1)}}
        if dia[o]:
            dd = np.array(dia[o])
            entry["max_diameter_mm"] = {"p2.5": pct(dd, 2.5), "median": pct(dd, 50), "p97.5": pct(dd, 97.5)}
        if tum[o]:
            entry["tumor_cm3"] = {"n": len(tum[o]), "median": pct(np.array(tum[o]), 50),
                                  "p97.5": pct(np.array(tum[o]), 97.5)}
        atlas[o] = entry

    json.dump({"n_patients": len(recs), "organs": atlas}, open(f"{D}/kg_atlas.json", "w"), indent=1)
    print(f"KG atlas over {len(recs)} patients -> kg/data/kg_atlas.json\n")
    print(f"{'organ':16s} {'n':>5s}  {'plausible volume cm3 (p2.5..p97.5)':36s} sources")
    for o, e in atlas.items():
        vb = e["volume_cm3"]
        print(f"{o:16s} {e['n']:5d}  {vb['p2.5']:>8.1f} .. {vb['p97.5']:<8.1f} (med {vb['median']:.0f})   "
              f"{','.join(e['sources'])}")


if __name__ == "__main__":
    main()
