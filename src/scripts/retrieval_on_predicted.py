#!/usr/bin/env python3
"""OAKG retrieval on the pipeline's PREDICTED phenotypes — the system-level claim.

Rank candidate patients by similarity computed from PREDICTED phenotypes, but score RELEVANCE against GROUND-TRUTH
tumor features — i.e. does predicted-phenotype retrieval still surface the truly-relevant patients? Compares:
  (1) GT phenotypes, γ (proposed)      — the upper bound
  (2) PREDICTED phenotypes, γ          — the real pipeline result
  (3) PREDICTED, no-γ (coverage_blind) — γ ablation (should invite spurious cross-organ matches)
  (4) PREDICTED, base (organ-agnostic) — weakest baseline
Reuses kg_retrieval_v2 (similarity + relevance). -> results/retrieval_on_predicted.json
"""
import glob
import json
import sys

import numpy as np

sys.path.insert(0, "/home/ud3d4/Desktop/SWOG/src/scripts")
from kg_retrieval_v2 import similarity, relevant, _dcg

RES = "/home/ud3d4/Desktop/SWOG/results"


def load(kind):
    recs = []
    for f in glob.glob(f"{RES}/corpus_{kind}_*.json"):
        recs += json.load(open(f))["records"]
    return {r["case_id"]: r for r in recs}


def eval_retrieval(sim, rel, ids, mode, gamma_min=1):
    """sim/rel: case_id -> record (phenotypes for ranking / for relevance). Rank by sim, score relevance by rel."""
    P5, P10, AP, ND, SPUR = [], [], [], [], []
    for qi in ids:
        cands = []
        for bi in ids:
            if bi == qi:
                continue
            s = similarity(sim[qi], sim[bi], mode, gamma_min=gamma_min)
            if s is None:
                continue
            cands.append((s, bi))
        if not cands:
            continue
        cands.sort(key=lambda x: -x[0])
        rels = [1 if relevant(rel[qi], rel[bi]) else 0 for _, bi in cands]
        if sum(rels) == 0:
            continue
        P5.append(float(np.mean(rels[:5]))); P10.append(float(np.mean(rels[:10])))
        hit = ap = 0.0
        for i, r in enumerate(rels):
            if r:
                hit += 1; ap += hit / (i + 1)
        AP.append(ap / sum(rels))
        ND.append(_dcg(rels[:10]) / (_dcg(sorted(rels, reverse=True)[:10]) or 1))
        # spurious @10: retrieved a patient sharing NO observed organ with the query (thin-overlap match)
        SPUR.append(float(np.mean([0.0 if (set(sim[qi]["observed_organs"]) & set(sim[bi]["observed_organs"]))
                                    else 1.0 for _, bi in cands[:10]])))
    r = lambda x: round(float(np.mean(x)), 3) if x else None
    return {"P@5": r(P5), "P@10": r(P10), "mAP": r(AP), "nDCG": r(ND), "spurious@10": r(SPUR), "n_queries": len(ND)}


def main():
    predB, gtB = load("predicted"), load("gt")
    ids = [i for i in predB if i in gtB]
    print(f"aligned {len(ids)} patients (datasets: "
          f"{sorted({predB[i]['dataset'] for i in ids})})", flush=True)
    out = {
        "1_GT_phenotypes_gamma (upper bound)":  eval_retrieval(gtB, gtB, ids, "proposed"),
        "2_PREDICTED_phenotypes_gamma":         eval_retrieval(predB, gtB, ids, "proposed"),
        "3_PREDICTED_no_gamma (coverage_blind)": eval_retrieval(predB, gtB, ids, "coverage_blind"),
        "4_PREDICTED_base (organ-agnostic)":    eval_retrieval(predB, gtB, ids, "base"),
    }
    json.dump({"n_aligned": len(ids), "results": out}, open(f"{RES}/retrieval_on_predicted.json", "w"), indent=2)
    print("\n=== OAKG retrieval on predicted phenotypes (relevance scored vs GT tumor features) ===")
    for k, v in out.items():
        print(f"  {k:38s} P@5={v['P@5']} P@10={v['P@10']} mAP={v['mAP']} nDCG={v['nDCG']} spurious@10={v['spurious@10']} (n={v['n_queries']})")
    print("-> results/retrieval_on_predicted.json")


if __name__ == "__main__":
    main()
