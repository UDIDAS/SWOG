#!/usr/bin/env python3
"""
Execute the cross-dataset expert queries (B1-B7, Section 6 target stratum) against the
3-regime corpus and record the REALIZED retrievals — the concrete top-k each method returns.

Each B-query is instantiated as a real query case from the corpus; we run the proposed
(observability-aware) retrieval and its coverage-blind ablation, and report the top-k
candidates with similarity, the shared observed anatomy, and whether the retrieval crosses
datasets as intended. This turns the query DESIGN (Table 12) into shown execution results.

Corpus: corpus_3regime.json (slice-level FLARE carries tumour phenotypes, so pancreas/LiTS
queries can genuinely match FLARE cases on the shared organ). GT-derived phenotypes.

Out: kg/data/crossdataset_query_results.json
"""
import json
import kg_retrieval_v2 as R

RES = "/home/ud3d4/Desktop/SWOG/kg/data"
REC = json.load(open(f"{RES}/corpus_3regime.json"))["records"]
BY_ID = {r["case_id"]: r for r in REC}


def organ_has(rec, organ, **kv):
    o = rec["organs"].get(organ)
    if not o:
        return False
    return all(o.get(k) == v for k, v in kv.items())


def pick(pred, prefer=None):
    """First case matching pred; if prefer(key) given, pick the max by it for a vivid example."""
    cands = [r for r in REC if pred(r)]
    if not cands:
        return None
    if prefer:
        cands.sort(key=prefer, reverse=True)
    return cands[0]


# ---- instantiate each cross-dataset query as a concrete query case + intended target dataset ----
def q_pancreas_contained():
    return pick(lambda r: r["dataset"] == "pancreas" and organ_has(r, "pancreas", has_tumor=True, containment="contained"),
               prefer=lambda r: r["organs"]["pancreas"].get("tumor_voxels", 0))

def q_pancreas_highburden():
    return pick(lambda r: r["dataset"] == "pancreas" and organ_has(r, "pancreas", has_tumor=True, burden_cat="high"),
               prefer=lambda r: r["organs"]["pancreas"].get("tumor_voxels", 0))

def q_lits_multifocal():
    return pick(lambda r: r["dataset"] == "lits" and organ_has(r, "liver", has_tumor=True, multiplicity="multifocal"),
               prefer=lambda r: r["organs"]["liver"].get("tumor_voxels", 0))

def q_flare_liver_tumor():
    return pick(lambda r: r["dataset"] == "flare" and organ_has(r, "liver", has_tumor=True),
               prefer=lambda r: r["organs"]["liver"].get("tumor_area_px", 0))

def q_flare_pancreas_tumor():
    return pick(lambda r: r["dataset"] == "flare" and organ_has(r, "pancreas", has_tumor=True),
               prefer=lambda r: r["organs"]["pancreas"].get("tumor_area_px", 0))

QUERIES = [
    ("B1", "Pancreas query -> FLARE match (containment)", q_pancreas_contained, "flare", "pancreas", "containment"),
    ("B2", "LiTS query -> FLARE match (multifocality)", q_lits_multifocal, "flare", "liver", "multiplicity"),
    ("B3", "FLARE query -> LiTS match (high-burden multifocal)", q_flare_liver_tumor, "lits", "liver", "burden_cat"),
    ("B4", "FLARE query -> Pancreas match (burden + containment)", q_flare_pancreas_tumor, "pancreas", "pancreas", "containment"),
    ("B5", "Pancreas query -> FLARE match (burden + containment)", q_pancreas_highburden, "flare", "pancreas", "burden_cat"),
    ("B6", "Cross-vocabulary hepatic concept (LiTS<->FLARE liver)", q_lits_multifocal, "flare", "liver", "multiplicity"),
    ("B7", "Graded-ontology organ-hierarchy (FLARE liver tumor)", q_flare_liver_tumor, "lits", "liver", "burden_cat"),
]


def _agree(query, c, organ, pheno):
    a, b = query["organs"].get(organ, {}), c["organs"].get(organ, {})
    return a.get(pheno) == b.get(pheno) and a.get(pheno) not in (None, "none", "na", "unknown")

def topk(query, mode, k=5, target=None, restrict_to_target=False, organ=None, pheno=None):
    scored = []
    for c in REC:
        if c["case_id"] == query["case_id"]:
            continue
        if restrict_to_target and c["dataset"] != target:
            continue
        s = R.similarity(query, c, mode)
        if s is None:
            continue
        scored.append((round(s, 3), c))
    scored.sort(key=lambda t: -t[0])
    out = []
    for s, c in scored[:k]:
        shared = sorted(set(query["observed_organs"]) & set(c["observed_organs"]))
        row = {"case_id": c["case_id"], "dataset": c["dataset"], "sim": s,
               "shared_organs": shared, "cross_dataset": c["dataset"] != query["dataset"]}
        if organ and pheno:
            row["phenotype_match"] = _agree(query, c, organ, pheno)
        out.append(row)
    tgt_rank = next((i + 1 for i, (_, c) in enumerate(scored) if c["dataset"] == target), None)
    return out, tgt_rank


def main():
    results = []
    for code, title, qfn, target, organ, pheno in QUERIES:
        q = qfn()
        if q is None:
            results.append({"code": code, "title": title, "error": "no matching query case"})
            print(f"{code}: NO QUERY CASE"); continue
        qp = q["organs"].get(organ, {})
        qdesc = {"case_id": q["case_id"], "dataset": q["dataset"],
                 "observed_organs": q["observed_organs"],
                 organ: {k: qp.get(k) for k in ("has_tumor", "burden_cat", "multiplicity", "containment")}}
        prop, prop_rank = topk(q, "proposed", target=target, organ=organ, pheno=pheno)
        cb, cb_rank = topk(q, "coverage_blind", target=target, organ=organ, pheno=pheno)
        # the cross-dataset execution: top matches restricted to the intended target source
        xprop, _ = topk(q, "proposed", target=target, restrict_to_target=True, organ=organ, pheno=pheno)
        xcb, _ = topk(q, "coverage_blind", target=target, restrict_to_target=True, organ=organ, pheno=pheno)
        results.append({"code": code, "title": title, "target_dataset": target,
                        "shared_organ": organ, "phenotype": pheno, "query": qdesc,
                        "proposed_top5": prop, "coverage_blind_top5": cb,
                        "cross_dataset_top5_proposed": xprop, "cross_dataset_top5_coverage_blind": xcb,
                        "target_first_rank": {"proposed": prop_rank, "coverage_blind": cb_rank}})
        pt = prop[0] if prop else {}
        print(f"{code} [{q['dataset']}->{target}] q={q['case_id']}  "
              f"proposed top1={pt.get('case_id')}({pt.get('dataset')},{pt.get('sim')})  "
              f"target@rank prop={prop_rank} cblind={cb_rank}")

    json.dump({"corpus": "3-regime (slice-level FLARE carries tumour phenotypes)",
               "note": "Realized cross-dataset retrievals for target-stratum queries B1-B7. "
                       "'target_first_rank' = rank at which the intended cross-dataset source first "
                       "appears; lower is better. Proposed vs coverage-blind ablation on identical pools.",
               "queries": results}, open(f"{RES}/crossdataset_query_results.json", "w"), indent=2)
    print(f"\nSaved: {RES}/crossdataset_query_results.json")


if __name__ == "__main__":
    main()
