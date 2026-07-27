#!/usr/bin/env python3
"""
Build the 3-regime retrieval corpus for the JBI IPKG evaluation.

Merges:
  - Pancreas (single-organ, single-tumor regime)   : 281 cases, observe {pancreas}
  - LiTS     (single-organ, multifocal regime)      : 131 cases, observe {liver}
  - FLARE    (multi-organ regime)                   : content-matched slice-cases
                                                       observing 1-5 of {liver, RK, spleen,
                                                       pancreas, LK}, from build_flare_multiorgan.

The FLARE cases are SLICE-LEVEL pseudo-cases (a physical CT slice matched across the
per-class arrays by content hash), NOT per-patient volumes. They supply the genuine
partial-observability overlap the coverage-corrected retrieval needs; granularity is
recorded on every record ("granularity": "slice" vs "volume") and reported honestly.

FLARE is sampled to a tractable, stratified subset (all cross-organ cases + tumor cases +
a coverage-stratified tumor-free remainder) so pairwise retrieval stays O(1e6), and the
per-organ phenotype schema is harmonized to the Pancreas/LiTS record shape.

Out: kg/data/corpus_3regime.json
"""
import json, os
from collections import Counter, defaultdict

RES = "/home/ud3d4/Desktop/SWOG/kg/data"
OUT = f"{RES}/corpus_3regime.json"
FLARE_TARGET = 600          # tractable multi-organ sample size
SEED = 42


def _rng(seed):
    import random
    r = random.Random(seed)
    return r


def harmonize_flare(c):
    """Map a FLARE multi-organ case to the Pancreas/LiTS per-organ record schema."""
    organs = {}
    for name, o in c["organs"].items():
        has_t = o["has_tumor"]
        organs[name] = {
            "present": True,
            "organ_area_px": o["organ_area_px"],
            "has_tumor": has_t,
            "tumor_area_px": o.get("tumor_area_px", 0),
            "burden_cat": o.get("burden_cat", "none"),
            # case-level multiplicity assigned to tumor-bearing organs; "none" otherwise
            "multiplicity": c["multiplicity"] if has_t else "none",
            "containment": o.get("containment", "none"),
            # FLARE sub-site (head/body/tail) is not computed -> unknown (onto_sim handles it)
            "anatomic_location": "unknown",
        }
    return {
        "case_id": c["case_id"],
        "dataset": "flare",
        "granularity": "slice",
        "observed_organs": c["observed_organs"],
        "n_observed": c["n_observed"],
        "organs": organs,
        "has_tumor": c["has_tumor"],
        "cross_organ": c["cross_organ"],
        "multiplicity": c["multiplicity"],
    }


def sample_flare(cases, target, seed):
    """Stratified sample: all cross-organ, then tumor cases, then coverage-stratified TN."""
    rng = _rng(seed)
    cross = [c for c in cases if c["cross_organ"]]
    tumor = [c for c in cases if c["has_tumor"] and not c["cross_organ"]]
    clean = [c for c in cases if not c["has_tumor"]]

    picked = list(cross)                                   # keep all cross-organ (informative, rare)
    # tumor cases: sample up to 40% of target, balanced across which organ carries the tumor
    by_org = defaultdict(list)
    for c in tumor:
        for name, o in c["organs"].items():
            if o["has_tumor"]:
                by_org[name].append(c); break
    tumor_quota = max(0, int(0.40 * target))
    per = max(1, tumor_quota // max(1, len(by_org)))
    for name, lst in by_org.items():
        rng.shuffle(lst)
        picked += lst[:per]
    # remainder: coverage-stratified tumor-free cases across n_observed
    seen = {c["case_id"] for c in picked}
    remain = target - len(picked)
    if remain > 0:
        by_n = defaultdict(list)
        for c in clean:
            if c["case_id"] not in seen:
                by_n[c["n_observed"]].append(c)
        ks = sorted(by_n)
        per_n = max(1, remain // max(1, len(ks)))
        for k in ks:
            lst = by_n[k]; rng.shuffle(lst)
            picked += lst[:per_n]
    # de-dupe, cap, deterministic order
    uniq = {}
    for c in picked:
        uniq[c["case_id"]] = c
    out = list(uniq.values())
    rng.shuffle(out)
    return out[:target] if len(out) > target else out


def main():
    base = json.load(open(f"{RES}/case_phenotypes.json"))["records"]
    for r in base:                                         # tag granularity on the single-organ sets
        r.setdefault("granularity", "volume")
    flare_all = json.load(open(f"{RES}/flare_multiorgan_cases.json"))["cases"]
    flare_s = sample_flare(flare_all, FLARE_TARGET, SEED)
    flare_rec = [harmonize_flare(c) for c in flare_s]

    corpus = base + flare_rec
    json.dump({"n": len(corpus),
               "organ_universe": ["liver", "right_kidney", "spleen", "pancreas", "left_kidney"],
               "records": corpus}, open(OUT, "w"))

    # report
    print(f"Corpus: {len(corpus)} cases")
    print("  by dataset:", dict(Counter(r["dataset"] for r in corpus)))
    print("  by granularity:", dict(Counter(r["granularity"] for r in corpus)))
    nobs = Counter(r["n_observed"] if "n_observed" in r else len(r["observed_organs"]) for r in corpus)
    print("  observed-organ count:", {k: nobs[k] for k in sorted(nobs)})
    print("  FLARE sampled:", len(flare_rec),
          f"(cross-organ {sum(c['cross_organ'] for c in flare_s)}, tumor {sum(c['has_tumor'] for c in flare_s)})")
    # partial-overlap check: FLARE cases whose observed set partially overlaps a single-organ query
    po = sum(1 for r in flare_rec
             if ("pancreas" in r["observed_organs"] or "liver" in r["observed_organs"])
             and r["n_observed"] >= 2)
    print(f"  FLARE cases overlapping pancreas/liver AND multi-organ (decomposition support): {po}")
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
