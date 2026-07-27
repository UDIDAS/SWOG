#!/usr/bin/env python3
"""
IPKG observability-aware retrieval over the 3-regime corpus (Pancreas + LiTS + FLARE
multi-organ). Fills JBI Tables 5 (aggregate ablation ladder), 6 (stratified: within /
cross / decomposition), 7 (LOPO incl. cross-organ), 8 (coverage controls), 9 (sensitivity),
11 (cross-dataset integration: mixing index vs null, observability gap, leave-one-dataset-out).

Reads corpus_3regime.json (built by jbi_build_corpus.py). All phenotypes are GT-derived
(the retrieval benchmark); segmentation-prediction robustness is a separate arm.

Honesty notes carried into the outputs:
  - FLARE cases are SLICE-level pseudo-cases (content-matched), not per-patient volumes.
  - Automatic relevance is phenotype-agreement (circular by construction); the non-circular
    signals are LOPO (Table 7), the coverage controls (Table 8), and expert judgment (Table 10,
    needs raters). Reported as such.
  - Cited external methods [33],[34] require their own code/data -> left TBD, not fabricated.
"""
import os, json, itertools
import numpy as np
from scipy.stats import wilcoxon

RES = "/home/ud3d4/Desktop/SWOG/kg/data"
CORPUS = json.load(open(f"{RES}/corpus_3regime.json"))
REC = CORPUS["records"]
ORGAN_UNIVERSE = CORPUS["organ_universe"]        # 5 organs
CATS = ["burden_cat", "multiplicity", "containment", "anatomic_location"]
LOC_CATS = {"anatomic_location"}                 # only pancreas sub-site uses graded ontology
NULL_PERMUTATIONS = 200
SEED = 42

# ---- anatomy ontology for graded sub-site similarity (Lin-style) ----
# root -> abdominal_organ -> {liver, spleen, pancreas, kidney}; kidney -> {right/left kidney};
# pancreas -> {head, body, tail}
IC = {"root": 0.0, "abdominal_organ": 0.3,
      "liver": 0.6, "spleen": 0.6, "pancreas": 0.6, "kidney": 0.5,
      "right_kidney": 0.75, "left_kidney": 0.75,
      "head": 0.9, "body": 0.9, "tail": 0.9}
PARENT = {"abdominal_organ": "root",
          "liver": "abdominal_organ", "spleen": "abdominal_organ",
          "pancreas": "abdominal_organ", "kidney": "abdominal_organ",
          "right_kidney": "kidney", "left_kidney": "kidney",
          "head": "pancreas", "body": "pancreas", "tail": "pancreas"}


def ancestors(c):
    out = [c]
    while c in PARENT:
        c = PARENT[c]; out.append(c)
    return out


def onto_sim(a, b, graded=True):
    if a == b:
        return 1.0
    if not graded or a not in IC or b not in IC:
        return 0.0
    Aa, Ab = ancestors(a), ancestors(b)
    lcs = next((x for x in Aa if x in Ab), "root")
    denom = IC.get(a, 0) + IC.get(b, 0)
    return (2 * IC.get(lcs, 0) / denom) if denom > 0 else 0.0


def phen(rec, organ):
    return rec["organs"].get(organ)


# ============================ similarity model ============================
def feat_agree(pa, pb, cat, graded):
    va, vb = pa.get(cat, "none"), pb.get(cat, "none")
    if cat in LOC_CATS:
        if va in ("na", "unknown", "none") or vb in ("na", "unknown", "none"):
            return 1.0 if va == vb else 0.0
        return onto_sim(va, vb, graded)
    return 1.0 if va == vb else 0.0


def organ_cats(o):
    return CATS if o == "pancreas" else CATS[:3]


def similarity(A, B, mode, weights=None, gamma_min=1):
    """Similarity in [0,1], or None if incomparable (proposed on <gamma_min shared organs)."""
    oa, ob = set(A["observed_organs"]), set(B["observed_organs"])
    if mode == "base":                      # organ-agnostic pooled, no observability/typing
        fa, fb = phen(A, sorted(oa)[0]), phen(B, sorted(ob)[0])
        return float(np.mean([1.0 if fa.get(c, "none") == fb.get(c, "none") else 0.0 for c in CATS[:3]]))
    if mode == "proposed":
        shared = oa & ob
        if len(shared) < gamma_min:
            return None                     # incomparable
        organs, graded = shared, True
    elif mode == "coverage_blind":
        organs, graded = set(ORGAN_UNIVERSE), True   # unobserved -> absent
    elif mode == "graded":
        organs, graded = oa | ob, True
    elif mode in ("typed", "flat_tier"):
        organs, graded = oa | ob, False
    else:
        raise ValueError(mode)

    sims, wts = [], []
    for o in organs:
        pa, pb = phen(A, o), phen(B, o)
        if pa is None and pb is None:
            continue
        if pa is None or pb is None:
            if mode == "coverage_blind":
                sims.append(0.0); wts.append(1.0)   # observed-absent penalty
            continue                                # typed/graded/proposed: not jointly meaningful
        cats = organ_cats(o)
        w = weights if weights is not None else {c: 1.0 for c in CATS}
        num = sum(w.get(c, 1.0) * feat_agree(pa, pb, c, graded) for c in cats)
        den = sum(w.get(c, 1.0) for c in cats)
        sims.append(num / den if den else 0.0); wts.append(1.0)
    if not sims:
        return None if mode == "proposed" else 0.0
    return float(np.average(sims, weights=wts))


# ============================ relevance ============================
def _agree_over(A, B, keys):
    shared = set(A["observed_organs"]) & set(B["observed_organs"])
    for o in shared:
        pa, pb = phen(A, o), phen(B, o)
        n = 0
        for k in keys:
            va = pa.get(k)
            if va == pb.get(k) and va not in ("none", "na", "unknown", None):
                n += 1
        if n >= (2 if len(keys) > 1 else 1):
            return True
    return False


def relevant(A, B):
    return _agree_over(A, B, ["burden_cat", "multiplicity", "containment", "anatomic_location"])


# ============================ evaluation ============================
def _query_metrics(A, cands, rel_fn):
    cands.sort(key=lambda x: -x[0])
    rels = [1 if rel_fn(A, c[1]) else 0 for c in cands]
    tot = sum(rels)
    if tot == 0:
        return None
    p5, p10 = np.mean(rels[:5]), np.mean(rels[:10])
    hit = ap = 0.0
    for i, r in enumerate(rels):
        if r:
            hit += 1; ap += hit / (i + 1)
    ap /= tot
    ndcg = _dcg(rels[:10]) / (_dcg(sorted(rels, reverse=True)[:10]) or 1)
    ep = np.mean([1.0 if set(A["observed_organs"]) & set(c[1]["observed_organs"]) else 0.0
                  for c in cands[:10]]) if cands else 0.0
    return p5, p10, ap, ndcg, ep


def _dcg(rels):
    return sum((2 ** r - 1) / np.log2(i + 2) for i, r in enumerate(rels))


def eval_mode(mode, records, rel_fn=relevant, stratum=None, weights=None, gamma_min=1,
              return_per_query=False):
    P5, P10, AP, ND, EP, perq = [], [], [], [], [], {}
    for qi, A in enumerate(records):
        cands = []
        for bi, B in enumerate(records):
            if bi == qi:
                continue
            if stratum == "within" and B["dataset"] != A["dataset"]:
                continue
            if stratum == "cross" and B["dataset"] == A["dataset"]:
                continue
            if stratum == "decomp":
                oa, ob = set(A["observed_organs"]), set(B["observed_organs"])
                if not (oa < ob):            # query strictly narrower-coverage than candidate
                    continue
            s = similarity(A, B, mode, weights=weights, gamma_min=gamma_min)
            if s is None:
                continue
            cands.append((s, B))
        if not cands:
            continue
        m = _query_metrics(A, cands, rel_fn)
        if m is None:
            continue
        P5.append(m[0]); P10.append(m[1]); AP.append(m[2]); ND.append(m[3]); EP.append(m[4])
        perq[A["case_id"]] = m[3]           # nDCG per query for paired tests
    out = {"P@5": _r(P5), "P@10": _r(P10), "mAP": _r(AP), "nDCG": _r(ND),
           "expl_path": _r(EP), "n_queries": len(ND)}
    return (out, perq) if return_per_query else out


def _r(x):
    return round(float(np.mean(x)), 3) if len(x) else None


def paired_p(perq_a, perq_b):
    """Wilcoxon signed-rank on shared queries; returns (median_diff, p) or (diff, None)."""
    keys = [k for k in perq_a if k in perq_b]
    da = np.array([perq_a[k] for k in keys]); db = np.array([perq_b[k] for k in keys])
    diff = float(np.mean(da - db))
    if len(keys) < 8 or np.allclose(da, db):
        return round(diff, 3), None
    try:
        p = float(wilcoxon(da, db, zero_method="wilcox").pvalue)
    except ValueError:
        p = None
    return round(diff, 3), (round(p, 5) if p is not None else None)


def holm(pvals):
    """Holm-Bonferroni over a dict name->p; returns name->adjusted p (None passthrough)."""
    items = [(k, v) for k, v in pvals.items() if v is not None]
    items.sort(key=lambda x: x[1])
    m = len(items); adj = {}
    prev = 0.0
    for i, (k, p) in enumerate(items):
        a = min(1.0, (m - i) * p)
        a = max(a, prev); prev = a
        adj[k] = round(a, 5)
    for k, v in pvals.items():
        adj.setdefault(k, None)
    return adj


# ============================ Table 11 helpers ============================
def build_graph(records, mode="proposed", tau=0.6, gamma_min=1):
    import networkx as nx
    G = nx.Graph()
    for i, r in enumerate(records):
        G.add_node(i, dataset=r["dataset"])
    for i, j in itertools.combinations(range(len(records)), 2):
        s = similarity(records[i], records[j], mode, gamma_min=gamma_min)
        if s is not None and s >= tau:
            G.add_edge(i, j, weight=s)
    return G


def mixing_index(records, G):
    """M = fraction of intra-community edges that are cross-dataset, vs label-permutation null."""
    import networkx as nx
    from networkx.algorithms.community import greedy_modularity_communities
    if G.number_of_edges() == 0:
        return None
    comms = list(greedy_modularity_communities(G))
    node2c = {n: ci for ci, com in enumerate(comms) for n in com}
    labels = np.array([records[n]["dataset"] for n in G.nodes()])
    nodes = list(G.nodes())
    idx = {n: k for k, n in enumerate(nodes)}

    def cross_frac(lab):
        intra = cross = 0
        for u, v in G.edges():
            if node2c[u] == node2c[v]:
                intra += 1
                if lab[idx[u]] != lab[idx[v]]:
                    cross += 1
        return cross / intra if intra else 0.0

    M = cross_frac(labels)
    rng = np.random.RandomState(SEED)
    null = []
    for _ in range(NULL_PERMUTATIONS):
        perm = labels.copy(); rng.shuffle(perm)
        null.append(cross_frac(perm))
    null = np.array(null)
    p = float((np.sum(null >= M) + 1) / (len(null) + 1))
    # community composition (multi-dataset communities = integrated)
    multi = 0
    for com in comms:
        ds = {records[n]["dataset"] for n in com}
        if len(ds) >= 2 and len(com) >= 3:
            multi += 1
    return {"M_observed": round(M, 4), "null_mean": round(float(null.mean()), 4),
            "null_std": round(float(null.std()), 4), "delta_M": round(M - float(null.mean()), 4),
            "p_vs_null": round(p, 5), "n_communities": len(comms),
            "multi_dataset_communities": multi, "n_edges": G.number_of_edges()}


def leave_one_dataset_out(records):
    """Withhold each dataset from the corpus; its cases retrieve from the remaining sources."""
    out = {}
    datasets = sorted({r["dataset"] for r in records})
    for held in datasets:
        held_q = [r for r in records if r["dataset"] == held]
        pool = [r for r in records if r["dataset"] != held]
        # can held-out queries still find relevant cases among OTHER datasets (proposed)?
        ndcgs, served = [], 0
        for A in held_q:
            cands = []
            for B in pool:
                s = similarity(A, B, "proposed")
                if s is not None:
                    cands.append((s, B))
            if not cands:
                continue
            served += 1
            m = _query_metrics(A, cands, relevant)
            if m is not None:
                ndcgs.append(m[3])
        out[held] = {"n_held_queries": len(held_q),
                     "served_by_other_sources": served,
                     "served_pct": round(100 * served / max(1, len(held_q)), 1),
                     "nDCG_from_other_sources": _r(ndcgs),
                     "n_scored": len(ndcgs)}
    return out


# ============================ main ============================
def main():
    from collections import Counter
    print(f"Corpus: {len(REC)} cases  {dict(Counter(r['dataset'] for r in REC))}")

    # ---------- Table 5: aggregate ablation ladder ----------
    LADDER = [("Phenotype-vector (base)", "base", "Limited"),
              ("+ typed relations", "typed", "Partial"),
              ("+ graded ontology", "graded", "yes"),
              ("(flat-tier ontology, abl.)", "flat_tier", "yes"),
              ("(coverage-blind, abl.)", "coverage_blind", "yes"),
              ("Proposed IPKG", "proposed", "yes")]
    t5, perq = {}, {}
    print("\n=== Table 5: Aggregate retrieval (ablation ladder) ===")
    print(f"{'Method':<30s} {'P@5':>6s} {'P@10':>6s} {'mAP':>6s} {'nDCG':>6s} {'Expl':>6s} {'n':>5s}")
    for label, mode, expl in LADDER:
        r, pq = eval_mode(mode, REC, return_per_query=True)
        r["expl_flag"] = expl; t5[label] = r; perq[mode] = pq
        print(f"{label:<30s} {r['P@5']!s:>6} {r['P@10']!s:>6} {r['mAP']!s:>6} {r['nDCG']!s:>6} {r['expl_path']!s:>6} {r['n_queries']:>5}")
    # paired ∆nDCG vs proposed (Holm-corrected)
    praw = {}
    for label, mode, _ in LADDER:
        if mode == "proposed":
            continue
        d, p = paired_p(perq[mode], perq["proposed"])
        t5[label]["dNDCG_vs_prop"] = d; praw[label] = p
    hadj = holm(praw)
    for label in praw:
        t5[label]["p_holm"] = hadj[label]

    # ---------- Table 6: stratified ----------
    print("\n=== Table 6: Retrieval by stratum (proposed vs coverage-blind) ===")
    t6 = {}
    for stratum, tag in [("within", "Within-dataset (control)"),
                         ("cross", "Cross-dataset (target)"),
                         ("decomp", "Decomposition (target)")]:
        pr, ppq = eval_mode("proposed", REC, stratum=stratum, return_per_query=True)
        cb, cpq = eval_mode("coverage_blind", REC, stratum=stratum, return_per_query=True)
        d, p = paired_p(ppq, cpq)
        t6[tag] = {"proposed": pr, "coverage_blind": cb, "dObs_nDCG": d, "p": p}
        print(f"  {tag:26s} prop nDCG={pr['nDCG']} (n={pr['n_queries']})  "
              f"cblind nDCG={cb['nDCG']} (n={cb['n_queries']})  dObs={d} p={p}")
    # Holm across the two targeted strata
    praw6 = {k: v["p"] for k, v in t6.items() if v["p"] is not None}
    h6 = holm(praw6)
    for k in t6:
        t6[k]["p_holm"] = h6.get(k)

    # ---------- Table 7: LOPO (incl. cross-organ) ----------
    print("\n=== Table 7: LOPO (proposed vs coverage-blind) ===")
    t7 = {}
    def lopo_rel(held):
        return lambda A, B: _agree_over(A, B, [held])
    def crossorgan_rel(A, B):
        return A.get("cross_organ", False) and B.get("cross_organ", False) \
            and bool(set(A["observed_organs"]) & set(B["observed_organs"]))
    lopo = [("Tumor burden", lopo_rel("burden_cat")),
            ("Lesion multiplicity", lopo_rel("multiplicity")),
            ("Organ containment", lopo_rel("containment")),
            ("Cross-organ distrib.", crossorgan_rel)]
    for name, fn in lopo:
        pr = eval_mode("proposed", REC, rel_fn=fn)
        cb = eval_mode("coverage_blind", REC, rel_fn=fn)
        t7[name] = {"proposed": pr["nDCG"], "coverage_blind": cb["nDCG"],
                    "n_proposed": pr["n_queries"], "n_cblind": cb["n_queries"]}
        print(f"  {name:22s} proposed={pr['nDCG']}  coverage-blind={cb['nDCG']}")
    vp = [t7[n]["proposed"] for n, _ in lopo if t7[n]["proposed"] is not None]
    vc = [t7[n]["coverage_blind"] for n, _ in lopo if t7[n]["coverage_blind"] is not None]
    t7["Mean"] = {"proposed": round(np.mean(vp), 3) if vp else None,
                  "coverage_blind": round(np.mean(vc), 3) if vc else None}

    # ---------- Table 8: coverage controls ----------
    print("\n=== Table 8: Coverage controls (disjoint-observability pairs) ===")
    dt = de = cb_scored = 0
    for A, B in itertools.combinations(REC, 2):
        if not (set(A["observed_organs"]) & set(B["observed_organs"])):
            dt += 1
            if similarity(A, B, "proposed") is None:
                de += 1
            if similarity(A, B, "coverage_blind") is not None:
                cb_scored += 1
    t8 = {"n_disjoint_pairs": dt,
          "incomparable_excluded_pct": {"proposed": round(100 * de / dt, 1) if dt else None,
                                        "coverage_blind": 0.0},
          "silence_false_penalty_pct": {"proposed": 0.0,
                                        "coverage_blind": round(100 * cb_scored / dt, 1) if dt else None}}
    print(f"  disjoint pairs={dt}  proposed excl={t8['incomparable_excluded_pct']['proposed']}%  "
          f"cblind penalty={t8['silence_false_penalty_pct']['coverage_blind']}%")

    # ---------- Table 9: sensitivity ----------
    print("\n=== Table 9: Sensitivity (nDCG@10 range) ===")
    t9 = {}
    # (a) weights lambda on the simplex around uniform
    weight_sets = [
        {"burden_cat": 1, "multiplicity": 1, "containment": 1, "anatomic_location": 1},   # uniform
        {"burden_cat": 2, "multiplicity": 1, "containment": 1, "anatomic_location": 1},
        {"burden_cat": 1, "multiplicity": 2, "containment": 1, "anatomic_location": 1},
        {"burden_cat": 1, "multiplicity": 1, "containment": 2, "anatomic_location": 1},
        {"burden_cat": 3, "multiplicity": 2, "containment": 2, "anatomic_location": 1},
        {"burden_cat": 1, "multiplicity": 1, "containment": 1, "anatomic_location": 3},
    ]
    wv = [eval_mode("proposed", REC, weights=w)["nDCG"] for w in weight_sets]
    wv = [x for x in wv if x is not None]
    t9["Weights lambda (simplex, around 1/4 each)"] = {"range": "6 vertices around uniform",
        "nDCG_min": round(min(wv), 3), "nDCG_max": round(max(wv), 3)}
    # (b) coverage threshold gamma_min
    gv = {g: eval_mode("proposed", REC, gamma_min=g)["nDCG"] for g in (1, 2, 3)}
    gvv = [v for v in gv.values() if v is not None]
    t9["Coverage threshold gamma_min"] = {"range": "{1,2,3}", "per_value": gv,
        "nDCG_min": round(min(gvv), 3), "nDCG_max": round(max(gvv), 3)}
    # (c) burden threshold theta_B: re-bin burden_cat from stored numeric size, +/- nominal
    t9["Burden threshold theta_B (+/- nominal)"] = burden_sweep()
    for k, v in t9.items():
        print(f"  {k:42s} nDCG {v['nDCG_min']}-{v['nDCG_max']}")

    # ---------- Table 11: integration ----------
    print("\n=== Table 11: Cross-dataset integration ===")
    G = build_graph(REC, mode="proposed", tau=0.6)
    mix = mixing_index(REC, G)
    obs_gap = {"cross": {"dObs_nDCG": t6["Cross-dataset (target)"]["dObs_nDCG"],
                         "p_holm": t6["Cross-dataset (target)"].get("p_holm")}}
    lodo = leave_one_dataset_out(REC)
    t11 = {"mixing_index": mix, "observability_gap_cross": obs_gap, "leave_one_dataset_out": lodo}
    print(f"  Mixing M={mix['M_observed']} vs null {mix['null_mean']}+/-{mix['null_std']} "
          f"(dM={mix['delta_M']}, p={mix['p_vs_null']}); multi-dataset communities={mix['multi_dataset_communities']}")
    print(f"  Obs gap (cross-dataset nDCG): {obs_gap['cross']['dObs_nDCG']} (p_holm={obs_gap['cross']['p_holm']})")
    for held, v in lodo.items():
        print(f"  LODO hold {held:8s}: {v['served_pct']}% served by other sources, "
              f"nDCG={v['nDCG_from_other_sources']} (n={v['n_scored']})")

    out = {"_note": "3-regime corpus (Pancreas+LiTS+FLARE). FLARE cases are SLICE-level content-matched "
                    "pseudo-cases, not per-patient volumes. Automatic relevance is phenotype-agreement "
                    "(circular); non-circular evidence = Tables 7,8 and expert Table 10 (needs raters). "
                    "External methods [33],[34] require their own code -> TBD.",
           "corpus": {"n": len(REC), "by_dataset": dict(Counter(r["dataset"] for r in REC))},
           "table5": t5, "table6": t6, "table7": t7, "table8": t8, "table9": t9, "table11": t11}
    json.dump(out, open(f"{RES}/tables_5to11_retrieval.json", "w"), indent=2)
    print(f"\nSaved: {RES}/tables_5to11_retrieval.json")


def burden_sweep():
    """Re-bin burden_cat from stored numeric tumor size at +/- nominal thresholds, re-eval nDCG."""
    # nominal cutoffs: volume(cm3) for pancreas/lits ~ tumor_volume_cm3; area(px) for flare.
    # burden nominal: low<5, high>20 (cm3-ish) / low<80, high>400 (px). sweep +/-25%.
    import copy
    def rebin(rec, scale):
        r = copy.deepcopy(rec)
        for o in r["organs"].values():
            if not o.get("has_tumor"):
                continue
            if "tumor_volume_cm3" in o:
                v = o["tumor_volume_cm3"]; lo, hi = 5 * scale, 20 * scale
            else:
                v = o.get("tumor_area_px", 0); lo, hi = 80 * scale, 400 * scale
            o["burden_cat"] = "low" if v < lo else ("high" if v > hi else "medium")
        return r
    vals = []
    for scale in (0.75, 1.0, 1.25):
        recs = [rebin(r, scale) for r in REC]
        vals.append(eval_mode("proposed", recs)["nDCG"])
    vals = [v for v in vals if v is not None]
    return {"range": "+/-25% of nominal", "nDCG_min": round(min(vals), 3), "nDCG_max": round(max(vals), 3)}


if __name__ == "__main__":
    main()
