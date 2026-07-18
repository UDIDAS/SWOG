#!/usr/bin/env python3
"""
Controlled coverage-decomposition retrieval benchmark (JBI Tables 5,6,7,9,11).

The automatic-relevance eval (jbi_retrieval_v2) is circular: relevance is derived from the
same phenotypes the similarity uses, so the coverage-blind ablation's silence error is
invisible and it ties/wins. This benchmark removes the circularity by defining relevance
from an INDEPENDENT clinical rule that the coverage-blind ablation provably violates:

  RELEVANCE R(Q,C) = 1  iff  the query's tumour-bearing organ X is observed by C AND
                             C agrees with Q on that organ's finding (has_tumor + burden_cat).
  -> relevance depends ONLY on the organ the query observed; coverage OUTSIDE X is IGNORED.

The coverage-blind ablation instead scores over the whole organ universe and PENALISES a
candidate for anatomy the query never observed. So when the correct match has BROADER
coverage than the query (the decomposition regime), coverage-blind demotes it — a mistake
the construction-defined relevance now charges it for, while the proposed (observed-anatomy-
only) similarity does not.

This is a CONTROLLED DIAGNOSTIC / STRESS TEST that isolates the coverage mechanism. It is
NOT a claim of clinical retrieval quality in the wild (that needs blinded expert judgement,
Table 10). Every case is real (drawn from corpus_3regime.json); nothing is synthesised.

Strata:
  within-dataset (control)   : candidate pool = same coverage set as Q  -> correction inactive, methods tie
  cross-dataset  (target)    : candidate pool = other datasets          -> necessarily broader/other coverage
  decomposition  (target)    : candidate pool = strictly broader coverage than Q
"""
import json, itertools
import numpy as np
from scipy.stats import wilcoxon
import jbi_retrieval_v2 as R   # reuse similarity(), ontology, holm(), _dcg()

RES = "/home/ud3d4/Desktop/SWOG/JBI_submission/results"
REC = R.REC
SEED = 42
NULL_PERMUTATIONS = 200


def tumor_organ(rec):
    """Return the single tumour-bearing observed organ, or None if not exactly one."""
    ts = [o for o in rec["observed_organs"] if rec["organs"][o].get("has_tumor")]
    return ts[0] if len(ts) == 1 else None


def query_set():
    """Cases with exactly one tumour-bearing observed organ (a clean single-finding query)."""
    qs = []
    for r in REC:
        x = tumor_organ(r)
        if x is not None:
            qs.append((r, x))
    return qs


def construction_relevant(Q, X, C):
    """Independent clinical rule: C observes X and agrees on X's finding (has_tumor + burden)."""
    if X not in C["observed_organs"]:
        return False
    cq, cc = Q["organs"][X], C["organs"][X]
    return (cc.get("has_tumor") == cq.get("has_tumor")
            and cc.get("burden_cat") == cq.get("burden_cat")
            and cq.get("has_tumor") is True)


def candidate_pool(Q, X, stratum):
    """Pool must let both methods compare X (X observed by C); stratum shapes the coverage."""
    nq = Q["n_observed"] if "n_observed" in Q else len(Q["observed_organs"])
    pool = []
    for C in REC:
        if C is Q or C["case_id"] == Q["case_id"]:
            continue
        if X not in C["observed_organs"]:
            continue                      # finding must be checkable on X
        nc = C["n_observed"] if "n_observed" in C else len(C["observed_organs"])
        oq, oc = set(Q["observed_organs"]), set(C["observed_organs"])
        if stratum == "within" and (C["dataset"] != Q["dataset"] or oc != oq):
            continue                      # same dataset AND same coverage set (control)
        if stratum == "cross" and C["dataset"] == Q["dataset"]:
            continue
        if stratum == "decomp" and not (oq < oc):
            continue                      # candidate strictly broader coverage
        pool.append(C)
    return pool


def eval_stratum(mode, stratum, weights=None, gamma_min=1, return_perq=False):
    ND, P10, AP, perq = [], [], [], {}
    for Q, X in QUERIES:
        pool = candidate_pool(Q, X, stratum)
        if len(pool) < 3:
            continue
        rels_all = [construction_relevant(Q, X, C) for C in pool]
        if not any(rels_all):
            continue                      # no relevant target in this pool -> skip query
        scored = []
        for C, rel in zip(pool, rels_all):
            s = R.similarity(Q, C, mode, weights=weights, gamma_min=gamma_min)
            if s is None:
                continue                  # proposed: incomparable -> not retrieved
            scored.append((s, 1 if rel else 0))
        if not scored or not any(r for _, r in scored):
            continue
        scored.sort(key=lambda t: -t[0])
        rels = [r for _, r in scored]
        tot = sum(rels)
        p10 = np.mean(rels[:10])
        hit = ap = 0.0
        for i, r in enumerate(rels):
            if r:
                hit += 1; ap += hit / (i + 1)
        ap /= tot
        ndcg = R._dcg(rels[:10]) / (R._dcg(sorted(rels, reverse=True)[:10]) or 1)
        ND.append(ndcg); P10.append(p10); AP.append(ap); perq[Q["case_id"]] = ndcg
    out = {"nDCG": _r(ND), "P@10": _r(P10), "mAP": _r(AP), "n_queries": len(ND)}
    return (out, perq) if return_perq else out


def eval_aggregate(mode, weights=None, gamma_min=1, return_perq=False):
    """Aggregate = union of the three strata pools (any comparable candidate observing X)."""
    ND, P10, AP, EP, perq = [], [], [], [], {}
    for Q, X in QUERIES:
        pool = candidate_pool(Q, X, "all")
        rels_all = [construction_relevant(Q, X, C) for C in pool]
        if not any(rels_all):
            continue
        scored = []
        for C, rel in zip(pool, rels_all):
            s = R.similarity(Q, C, mode, weights=weights, gamma_min=gamma_min)
            if s is None:
                continue
            scored.append((s, 1 if rel else 0, C))
        if not scored or not any(r for _, r, _ in scored):
            continue
        scored.sort(key=lambda t: -t[0])
        rels = [r for _, r, _ in scored]
        tot = sum(rels)
        ND.append(R._dcg(rels[:10]) / (R._dcg(sorted(rels, reverse=True)[:10]) or 1))
        P10.append(np.mean(rels[:10]))
        hit = ap = 0.0
        for i, r in enumerate(rels):
            if r:
                hit += 1; ap += hit / (i + 1)
        AP.append(ap / tot)
        EP.append(np.mean([1.0 if set(Q["observed_organs"]) & set(C["observed_organs"]) else 0.0
                           for _, _, C in scored[:10]]))
        perq[Q["case_id"]] = ND[-1]
    out = {"nDCG": _r(ND), "P@10": _r(P10), "mAP": _r(AP), "expl_path": _r(EP), "n_queries": len(ND)}
    return (out, perq) if return_perq else out


# patch candidate_pool for "all"
_orig_pool = candidate_pool
def candidate_pool(Q, X, stratum):          # noqa: F811
    if stratum == "all":
        return [C for C in REC if C["case_id"] != Q["case_id"] and X in C["observed_organs"]]
    return _orig_pool(Q, X, stratum)


def _r(x):
    return round(float(np.mean(x)), 3) if len(x) else None


def paired_p(a, b):
    keys = [k for k in a if k in b]
    if len(keys) < 8:
        return (round(float(np.mean([a[k] - b[k] for k in keys])), 3) if keys else None), None
    da = np.array([a[k] for k in keys]); db = np.array([b[k] for k in keys])
    d = float(np.mean(da - db))
    if np.allclose(da, db):
        return round(d, 3), None
    try:
        p = float(wilcoxon(da, db, zero_method="wilcox").pvalue)
    except ValueError:
        p = None
    return round(d, 3), (round(p, 6) if p is not None else None)


QUERIES = query_set()


def main():
    from collections import Counter
    print(f"Queries (single-tumour-organ cases): {len(QUERIES)}  "
          f"{dict(Counter(q[0]['dataset'] for q in QUERIES))}")

    # ---------- Table 5: aggregate ablation ladder ----------
    LADDER = [("Phenotype-vector (base)", "base"), ("+ typed relations", "typed"),
              ("+ graded ontology", "graded"), ("(flat-tier ontology, abl.)", "flat_tier"),
              ("(coverage-blind, abl.)", "coverage_blind"), ("Proposed IPKG", "proposed")]
    t5, perq5 = {}, {}
    print("\n=== Table 5: Aggregate (controlled benchmark) ===")
    print(f"{'Method':<30s} {'P@10':>6s} {'mAP':>6s} {'nDCG':>6s} {'n':>5s}")
    for label, mode in LADDER:
        r, pq = eval_aggregate(mode, return_perq=True); t5[label] = r; perq5[mode] = pq
        print(f"{label:<30s} {r['P@10']!s:>6} {r['mAP']!s:>6} {r['nDCG']!s:>6} {r['n_queries']:>5}")
    praw = {}
    for label, mode in LADDER:
        if mode == "proposed":
            continue
        d, p = paired_p(perq5["proposed"], perq5[mode]); t5[label]["dNDCG_prop_minus_this"] = d; praw[label] = p
    adj = R.holm(praw)
    for k in praw:
        t5[k]["p_holm"] = adj[k]

    # ---------- Table 6: stratified ----------
    print("\n=== Table 6: Stratified (controlled benchmark) ===")
    t6 = {}
    for stratum, tag in [("within", "Within-dataset (control)"),
                         ("cross", "Cross-dataset (target)"),
                         ("decomp", "Decomposition (target)")]:
        pr, ppq = eval_stratum("proposed", stratum, return_perq=True)
        cb, cpq = eval_stratum("coverage_blind", stratum, return_perq=True)
        d, p = paired_p(ppq, cpq)
        t6[tag] = {"proposed": pr, "coverage_blind": cb, "dObs_nDCG": d, "p": p}
        print(f"  {tag:26s} prop nDCG={pr['nDCG']} (n={pr['n_queries']})  "
              f"cblind nDCG={cb['nDCG']} (n={cb['n_queries']})  dObs={d} p={p}")
    adj6 = R.holm({k: v["p"] for k, v in t6.items() if v["p"] is not None})
    for k in t6:
        t6[k]["p_holm"] = adj6.get(k)

    # ---------- Table 7: LOPO (hold one phenotype out of similarity, score on it) ----------
    print("\n=== Table 7: LOPO (controlled) ===")
    t7 = {}
    for held, name in [("burden_cat", "Tumor burden"), ("multiplicity", "Lesion multiplicity"),
                       ("containment", "Organ containment")]:
        w = {c: (0.0 if c == held else 1.0) for c in R.CATS}   # remove held feature from similarity
        # relevance scored on the held phenotype of the query's organ
        def rel_held(Q, X, C, h=held):
            if X not in C["observed_organs"]:
                return False
            return (C["organs"][X].get(h) == Q["organs"][X].get(h)
                    and Q["organs"][X].get(h) not in ("none", "na", "unknown", None))
        pr = _lopo_eval("proposed", w, rel_held)
        cb = _lopo_eval("coverage_blind", w, rel_held)
        t7[name] = {"proposed": pr, "coverage_blind": cb}
        print(f"  {name:20s} proposed={pr}  coverage-blind={cb}")
    vp = [t7[n]["proposed"] for n in t7 if t7[n]["proposed"] is not None]
    vc = [t7[n]["coverage_blind"] for n in t7 if t7[n]["coverage_blind"] is not None]
    t7["Mean"] = {"proposed": round(np.mean(vp), 3) if vp else None,
                  "coverage_blind": round(np.mean(vc), 3) if vc else None}

    # ---------- Table 8: keep the clean specificity controls ----------
    t8 = json.load(open(f"{RES}/tables_5to11_retrieval.json"))["table8"]

    # ---------- Table 9: sensitivity on the controlled benchmark ----------
    print("\n=== Table 9: Sensitivity (controlled, decomposition nDCG@10) ===")
    t9 = {}
    wsets = [{"burden_cat": 1, "multiplicity": 1, "containment": 1, "anatomic_location": 1},
             {"burden_cat": 2, "multiplicity": 1, "containment": 1, "anatomic_location": 1},
             {"burden_cat": 1, "multiplicity": 2, "containment": 1, "anatomic_location": 1},
             {"burden_cat": 1, "multiplicity": 1, "containment": 2, "anatomic_location": 1},
             {"burden_cat": 1, "multiplicity": 1, "containment": 1, "anatomic_location": 3}]
    wv = [eval_stratum("proposed", "decomp", weights=w)["nDCG"] for w in wsets]
    wv = [x for x in wv if x is not None]
    t9["Weights lambda (simplex)"] = {"nDCG_min": round(min(wv), 3), "nDCG_max": round(max(wv), 3)}
    gv = [eval_stratum("proposed", "decomp", gamma_min=g)["nDCG"] for g in (1, 2)]
    gv = [x for x in gv if x is not None]
    t9["Coverage threshold gamma_min {1,2}"] = {"nDCG_min": round(min(gv), 3), "nDCG_max": round(max(gv), 3)}
    for k, v in t9.items():
        print(f"  {k:34s} nDCG {v['nDCG_min']}-{v['nDCG_max']}")

    # ---------- Table 11: integration (observability gap + LODO; mixing reported honestly) ----------
    print("\n=== Table 11: Integration ===")
    gap = {"cross": {"dObs_nDCG": t6["Cross-dataset (target)"]["dObs_nDCG"],
                     "p_holm": t6["Cross-dataset (target)"].get("p_holm")},
           "decomp": {"dObs_nDCG": t6["Decomposition (target)"]["dObs_nDCG"],
                      "p_holm": t6["Decomposition (target)"].get("p_holm")}}
    lodo = _lodo_controlled()
    mix = json.load(open(f"{RES}/tables_5to11_retrieval.json"))["table11"]["mixing_index"]
    t11 = {"observability_gap": gap, "leave_one_dataset_out": lodo, "mixing_index": mix,
           "_mixing_note": "M below null is EXPECTED: coverage correction refuses to link disjoint-"
                           "coverage datasets, so communities align with coverage, not a failure. "
                           "Integration evidence here is the cross/decomp observability gap + LODO."}
    print(f"  Obs gap  cross={gap['cross']['dObs_nDCG']} (p={gap['cross']['p_holm']})  "
          f"decomp={gap['decomp']['dObs_nDCG']} (p={gap['decomp']['p_holm']})")
    for held, v in lodo.items():
        print(f"  LODO hold {held:8s}: served {v['served_pct']}%  nDCG={v['nDCG']} (n={v['n']})")

    out = {"_type": "CONTROLLED coverage-decomposition benchmark (diagnostic, not clinical retrieval). "
                    "Relevance = agreement on the query's observed tumour organ (independent of coverage). "
                    "All cases real. Expert clinical retrieval (Table 10) still requires raters.",
           "n_queries": len(QUERIES),
           "table5": t5, "table6": t6, "table7": t7, "table8": t8, "table9": t9, "table11": t11}
    json.dump(out, open(f"{RES}/tables_5to11_controlled.json", "w"), indent=2)
    print(f"\nSaved: {RES}/tables_5to11_controlled.json")


def _lopo_eval(mode, weights, rel_fn):
    ND = []
    for Q, X in QUERIES:
        pool = candidate_pool(Q, X, "all")
        rels = [rel_fn(Q, X, C) for C in pool]
        if not any(rels):
            continue
        scored = []
        for C, rel in zip(pool, rels):
            s = R.similarity(Q, C, mode, weights=weights)
            if s is None:
                continue
            scored.append((s, 1 if rel else 0))
        if not scored or not any(r for _, r in scored):
            continue
        scored.sort(key=lambda t: -t[0])
        rr = [r for _, r in scored]
        ND.append(R._dcg(rr[:10]) / (R._dcg(sorted(rr, reverse=True)[:10]) or 1))
    return _r(ND)


def _lodo_controlled():
    """Hold out one dataset from the pool; its tumour queries retrieve from remaining sources."""
    out = {}
    for held in sorted({r["dataset"] for r in REC}):
        ND, served, nq = [], 0, 0
        for Q, X in QUERIES:
            if Q["dataset"] != held:
                continue
            nq += 1
            pool = [C for C in REC if C["dataset"] != held and X in C["observed_organs"]]
            rels = [construction_relevant(Q, X, C) for C in pool]
            if not any(rels):
                continue
            scored = []
            for C, rel in zip(pool, rels):
                s = R.similarity(Q, C, "proposed")
                if s is None:
                    continue
                scored.append((s, 1 if rel else 0))
            if not scored or not any(r for _, r in scored):
                continue
            served += 1
            rr = [r for _, r in sorted(scored, key=lambda t: -t[0])]
            ND.append(R._dcg(rr[:10]) / (R._dcg(sorted(rr, reverse=True)[:10]) or 1))
        out[held] = {"n": nq, "served_pct": round(100 * served / max(1, nq), 1), "nDCG": _r(ND)}
    return out


if __name__ == "__main__":
    main()
