#!/usr/bin/env python3
"""
Table 12 — coverage matrix of the Nq=20 expert-evaluation query set.

Tags each of the 20 designed queries (Section 6: A within-control, B cross-dataset,
C decomposition, D adversarial/semantics) by the phenotype axes it exercises, the
coverage regime(s) it touches, and the ontology / three-valued-semantics / incomparability
behaviours it stresses — then verifies the set spans the phenotype x stratum space
(the claim Table 12 supports). Pure design table; no retrieval results or raters needed.

Out: JBI_submission/results/table12_coverage_matrix.json
"""
import json, os
from collections import defaultdict

RES = "/home/ud3d4/Desktop/SWOG/JBI_submission/results"
OUT = f"{RES}/table12_coverage_matrix.json"

# phenotype axes + semantics/feature axes (columns of the matrix)
PHENO = ["burden", "multiplicity", "containment", "cross_organ", "sub_site"]
FEAT = ["ontology_grounding", "three_valued", "incomparability"]
STRATA = {"A": "within (control)", "B": "cross-dataset (target)",
          "C": "decomposition (target)", "D": "adversarial / semantics"}

# (code, title, stratum, regimes touched, phenotypes, features)
Q = [
    ("A1", "Pancreas contained high-burden tumor", "A", ["pancreas"], ["burden", "containment"], []),
    ("A2", "Pancreas boundary-spanning containment", "A", ["pancreas"], ["containment", "sub_site"], []),
    ("A3", "LiTS multifocal hepatic disease", "A", ["lits"], ["multiplicity"], []),
    ("A4", "LiTS solitary high-burden lesion", "A", ["lits"], ["burden", "multiplicity"], []),
    ("A5", "FLARE single-organ tumor in multi-organ context", "A", ["flare"], ["burden", "containment", "cross_organ"], ["three_valued"]),
    ("B1", "Pancreas query -> FLARE match, containment", "B", ["pancreas", "flare"], ["containment"], []),
    ("B2", "LiTS query -> FLARE match, multifocality", "B", ["lits", "flare"], ["multiplicity"], []),
    ("B3", "FLARE query -> LiTS match, high-burden multifocal", "B", ["flare", "lits"], ["burden", "multiplicity"], []),
    ("B4", "FLARE query -> Pancreas match, burden + containment", "B", ["flare", "pancreas"], ["burden", "containment"], []),
    ("B5", "Pancreas query -> FLARE match, burden + containment", "B", ["pancreas", "flare"], ["burden", "containment"], []),
    ("B6", "Cross-vocabulary hepatic concept", "B", ["lits", "flare"], [], ["ontology_grounding"]),
    ("B7", "Graded-ontology organ-hierarchy query", "B", ["flare"], ["sub_site"], ["ontology_grounding"]),
    ("C1", "Multi-organ query decomposing to two sources", "C", ["flare", "pancreas", "lits"], ["burden"], ["incomparability"]),
    ("C2", "Cross-organ distribution / boundary-spanning tumor", "C", ["flare"], ["cross_organ", "containment"], []),
    ("C3", "Two phenotypes across two sources", "C", ["flare", "pancreas", "lits"], ["burden", "multiplicity"], []),
    ("C4", "Per-organ numerical burden match", "C", ["flare"], ["burden"], []),
    ("D1", "Burden-matched multiplicity confound", "D", ["lits"], ["multiplicity", "burden"], []),
    ("D2", "Containment / cross-organ adversarial pair", "D", ["flare", "pancreas"], ["containment", "cross_organ"], []),
    ("D3", "Indeterminate-handling query", "D", ["pancreas", "lits"], [], ["three_valued"]),
    ("D4", "Incomparability control query", "D", ["pancreas", "lits"], [], ["incomparability"]),
]


def main():
    rows = []
    for code, title, stratum, regimes, phenos, feats in Q:
        rows.append({"code": code, "title": title, "stratum": stratum,
                     "stratum_name": STRATA[stratum], "regimes": regimes,
                     **{p: (p in phenos) for p in PHENO},
                     **{f: (f in feats) for f in FEAT}})

    # coverage completeness: every phenotype and every feature covered in >=1 target stratum
    by_col = defaultdict(lambda: defaultdict(int))
    for r in rows:
        for col in PHENO + FEAT:
            if r[col]:
                by_col[col][r["stratum"]] += 1
    coverage = {col: {"total": sum(by_col[col].values()),
                      "by_stratum": dict(by_col[col])} for col in PHENO + FEAT}
    # regime coverage
    regime_hits = defaultdict(int)
    for r in rows:
        for rg in r["regimes"]:
            regime_hits[rg] += 1

    complete = all(coverage[c]["total"] >= 1 for c in PHENO + FEAT)
    # each phenotype exercised in at least one TARGET stratum (B or C)
    target_ok = {c: (by_col[c].get("B", 0) + by_col[c].get("C", 0)) >= 1 for c in PHENO}

    out = {"table": "Table 12 - Coverage matrix of the Nq=20 expert-evaluation query set",
           "n_queries": len(rows),
           "strata": {k: sum(1 for r in rows if r["stratum"] == k) for k in STRATA},
           "phenotype_axes": PHENO, "feature_axes": FEAT,
           "rows": rows, "coverage": coverage, "regime_hits": dict(regime_hits),
           "phenotype_covered_in_target_stratum": target_ok,
           "coverage_complete": bool(complete and all(target_ok.values()))}
    os.makedirs(RES, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2)

    print(f"Table 12: {len(rows)} queries  strata={out['strata']}")
    print("Phenotype coverage (by stratum):")
    for c in PHENO:
        print(f"  {c:12s} total={coverage[c]['total']:2d}  {dict(by_col[c])}  target-covered={target_ok[c]}")
    print("Feature coverage:")
    for c in FEAT:
        print(f"  {c:18s} total={coverage[c]['total']:2d}  {dict(by_col[c])}")
    print(f"Regime hits: {dict(regime_hits)}")
    print(f"COVERAGE COMPLETE (all phenotypes+features present, all phenotypes in a target stratum): {out['coverage_complete']}")
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
