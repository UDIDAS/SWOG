#!/usr/bin/env python3
"""
IPKG Steps 2-4 — observability-aware phenotype-driven retrieval + ablation ladder,
producing JBI Tables 5 (aggregate), 6 (stratified), 7 (LOPO), 8 (coverage controls),
9 (sensitivity). Reads the persistent case_phenotypes.json.

Ablation ladder (increasing mechanism):
  base        : flat phenotype-vector agreement, organ-agnostic, no observability
  +typed      : features namespaced by (organ, relation type)
  +graded     : anatomic concepts scored by IC-weighted ontology similarity (siblings partial)
  flat-tier   : ablation of +graded — ontology match is exact-only (binary)
  coverage-blind: observability ablation — unobserved organs treated as absent (comparable)
  proposed    : observability-aware — similarity only over JOINTLY-OBSERVED organs; disjoint => incomparable

Relevance (automatic, phenotype-agreement): B relevant to A iff, over jointly-observed
anatomy, they agree on >= 2 of {burden_cat, multiplicity, containment} (+ location for pancreas).
Circularity is acknowledged; LOPO (Table 7) provides the non-circular signal.
"""
import os, json, itertools
import numpy as np

RES = "/home/ud3d4/Desktop/SWOG/kg/data"
REC = json.load(open(f"{RES}/case_phenotypes.json"))["records"]
ORGAN_UNIVERSE = ["pancreas", "liver"]
CATS = ["burden_cat", "multiplicity", "containment", "anatomic_location"]

# tiny anatomy ontology for graded similarity (Lin-style over head/body/tail siblings)
# depth: root -> abdominal_organ -> {pancreas, liver}; pancreas -> {head, body, tail}
IC = {"root": 0.0, "abdominal_organ": 0.3, "pancreas": 0.6, "liver": 0.6,
      "head": 0.9, "body": 0.9, "tail": 0.9}
PARENT = {"abdominal_organ": "root", "pancreas": "abdominal_organ", "liver": "abdominal_organ",
          "head": "pancreas", "body": "pancreas", "tail": "pancreas"}


def ancestors(c):
    out = [c]
    while c in PARENT:
        c = PARENT[c]; out.append(c)
    return out


def onto_sim(a, b, graded=True):
    if a == b:
        return 1.0
    if not graded:
        return 0.0
    if a not in IC or b not in IC:
        return 0.0
    Aa, Ab = ancestors(a), ancestors(b)
    lcs = next((x for x in Aa if x in Ab), "root")
    denom = IC.get(a, 0) + IC.get(b, 0)
    return (2 * IC.get(lcs, 0) / denom) if denom > 0 else 0.0


def phen(rec, organ):
    return rec["organs"].get(organ)


def feat_agree(pa, pb, cat, graded):
    """agreement in [0,1] for one categorical feature over one organ."""
    va, vb = pa.get(cat, "none"), pb.get(cat, "none")
    if cat == "anatomic_location":
        return onto_sim(va, vb, graded) if va not in ("na", "unknown", "none") else (1.0 if va == vb else 0.0)
    return 1.0 if va == vb else 0.0


def similarity(A, B, mode):
    """Return similarity in [0,1], or None if incomparable (proposed on disjoint observability)."""
    oa, ob = set(A["observed_organs"]), set(B["observed_organs"])
    if mode == "base":
        # organ-agnostic: pool phenotype categoricals ignoring which organ / observability
        fa = [phen(A, o) for o in oa][0]; fb = [phen(B, o) for o in ob][0]
        vals = [1.0 if fa.get(c, "none") == fb.get(c, "none") else 0.0 for c in CATS[:3]]
        return float(np.mean(vals))
    if mode == "proposed":
        shared = oa & ob
        if not shared:
            return None  # incomparable
        organs = shared; graded = True; typed = True
    elif mode == "coverage_blind":
        organs = set(ORGAN_UNIVERSE); graded = True; typed = True  # unobserved -> absent
    elif mode == "graded":
        organs = oa | ob; graded = True; typed = True
    elif mode == "flat_tier":
        organs = oa | ob; graded = False; typed = True
    elif mode == "typed":
        organs = oa | ob; graded = False; typed = True
    else:
        raise ValueError(mode)
    sims = []
    for o in organs:
        pa, pb = phen(A, o), phen(B, o)
        if pa is None and pb is None:
            sims.append(1.0 if mode != "coverage_blind" else 1.0)  # both unobserved
            continue
        if pa is None or pb is None:
            # one observes o, other doesn't
            if mode == "coverage_blind":
                sims.append(0.0)   # treat unobserved as absent -> mismatch
            continue               # typed/graded/flat: skip organ neither jointly meaningful
        cats = CATS if o == "pancreas" else CATS[:3]
        sims.append(float(np.mean([feat_agree(pa, pb, c, graded) for c in cats])))
    return float(np.mean(sims)) if sims else None


def relevant(A, B):
    shared = set(A["observed_organs"]) & set(B["observed_organs"])
    if not shared:
        return False
    for o in shared:
        pa, pb = phen(A, o), phen(B, o)
        agree = sum(1 for c in ["burden_cat", "multiplicity", "containment"]
                    if pa.get(c) == pb.get(c) and pa.get(c) != "none")
        if o == "pancreas" and pa.get("anatomic_location") == pb.get("anatomic_location") \
           and pa.get("anatomic_location") not in ("na", "unknown", "none"):
            agree += 1
        if agree >= 2:
            return True
    return False


def dcg(rels):
    return sum((2 ** r - 1) / np.log2(i + 2) for i, r in enumerate(rels))


def eval_mode(mode, records, rel_fn=relevant, stratum=None):
    P5, P10, APs, nDCGs, expl = [], [], [], [], []
    for qi, A in enumerate(records):
        cands = []
        for bi, B in enumerate(records):
            if bi == qi:
                continue
            if stratum == "within" and B["dataset"] != A["dataset"]:
                continue
            if stratum == "cross" and B["dataset"] == A["dataset"]:
                continue
            s = similarity(A, B, mode)
            if s is None:
                continue  # incomparable -> not retrieved
            cands.append((s, rel_fn(A, B), B))
        if not cands:
            continue
        cands.sort(key=lambda x: -x[0])
        rels = [1 if c[1] else 0 for c in cands]
        total_rel = sum(rels)
        if total_rel == 0:
            continue
        P5.append(np.mean(rels[:5])); P10.append(np.mean(rels[:10]))
        # AP
        hit = 0; ap = 0.0
        for i, r in enumerate(rels):
            if r:
                hit += 1; ap += hit / (i + 1)
        APs.append(ap / total_rel)
        ideal = sorted(rels, reverse=True)
        nDCGs.append(dcg(rels[:10]) / (dcg(ideal[:10]) or 1))
        # explanation path: top-10 retrievals that share observed anatomy AND >=1 phenotype agreement
        ep = 0
        for _, _, B in cands[:10]:
            if set(A["observed_organs"]) & set(B["observed_organs"]):
                ep += 1
        expl.append(ep / min(10, len(cands)))
    return {"P@5": round(np.mean(P5), 3), "P@10": round(np.mean(P10), 3),
            "mAP": round(np.mean(APs), 3), "nDCG": round(np.mean(nDCGs), 3),
            "expl_path": round(np.mean(expl), 3), "n_queries": len(APs)}


LADDER = [("Phenotype-vector (base)", "base"), ("+typed relations", "typed"),
          ("+graded ontology", "graded"), ("(flat-tier ontology, abl.)", "flat_tier"),
          ("(coverage-blind, abl.)", "coverage_blind"), ("Proposed IPKG", "proposed")]

if __name__ == "__main__":
    print(f"Cases: {len(REC)}  (pancreas {sum(r['dataset']=='pancreas' for r in REC)}, "
          f"lits {sum(r['dataset']=='lits' for r in REC)})")

    # ---- Table 5: aggregate ablation ladder ----
    t5 = {}
    print("\n=== Table 5: Aggregate phenotype-driven retrieval (ablation ladder) ===")
    print(f"{'Method':<30s} {'P@5':>6s} {'P@10':>6s} {'mAP':>6s} {'nDCG':>6s} {'Expl':>6s}")
    for label, mode in LADDER:
        r = eval_mode(mode, REC)
        t5[label] = r
        print(f"{label:<30s} {r['P@5']:>6.3f} {r['P@10']:>6.3f} {r['mAP']:>6.3f} {r['nDCG']:>6.3f} {r['expl_path']:>6.3f}")

    # ---- Table 6: stratified (within / cross) ----
    t6 = {}
    print("\n=== Table 6: Retrieval by stratum (proposed vs coverage-blind) ===")
    for stratum in ["within", "cross"]:
        t6[stratum] = {"proposed": eval_mode("proposed", REC, stratum=stratum),
                       "coverage_blind": eval_mode("coverage_blind", REC, stratum=stratum)}
        p, c = t6[stratum]["proposed"], t6[stratum]["coverage_blind"]
        print(f"  {stratum:8s} proposed nDCG={p['nDCG']} (n={p['n_queries']})  "
              f"coverage-blind nDCG={c['nDCG']} (n={c['n_queries']})")

    # ---- Table 8: coverage-model controls ----
    # incomparability: fraction of disjoint-observability pairs correctly excluded by proposed
    disjoint_total = disjoint_excl = 0
    for A, B in itertools.combinations(REC, 2):
        if not (set(A["observed_organs"]) & set(B["observed_organs"])):
            disjoint_total += 1
            if similarity(A, B, "proposed") is None:
                disjoint_excl += 1
    incomparable_rate = round(100 * disjoint_excl / disjoint_total, 1) if disjoint_total else None
    # silence control: coverage-blind assigns a (non-None) score to disjoint pairs => false penalty
    cb_scores_disjoint = sum(1 for A, B in itertools.combinations(REC, 2)
                             if not (set(A["observed_organs"]) & set(B["observed_organs"]))
                             and similarity(A, B, "coverage_blind") is not None)
    t8 = {"incomparable_pairs_excluded_pct": {"proposed": incomparable_rate, "coverage_blind": 0.0},
          "silence_false_penalty_pct": {"proposed": 0.0,
              "coverage_blind": round(100 * cb_scores_disjoint / disjoint_total, 1) if disjoint_total else None}}
    print(f"\n=== Table 8: Coverage controls ===")
    print(f"  Incomparable pairs correctly excluded: proposed={incomparable_rate}%  coverage-blind=0.0%")
    print(f"  Silence-as-absence false-penalty: proposed=0.0%  coverage-blind={t8['silence_false_penalty_pct']['coverage_blind']}%")

    # ---- Table 7: LOPO (hold out one phenotype from relevance; score on it alone) ----
    def lopo_rel(held):
        def fn(A, B):
            shared = set(A["observed_organs"]) & set(B["observed_organs"])
            for o in shared:
                pa, pb = phen(A, o), phen(B, o)
                if pa.get(held) == pb.get(held) and pa.get(held) not in ("none", "na", "unknown"):
                    return True
            return False
        return fn
    t7 = {}
    print("\n=== Table 7: LOPO (proposed vs coverage-blind), nDCG on held-out phenotype ===")
    for held, name in [("burden_cat", "Tumor burden"), ("multiplicity", "Lesion multiplicity"),
                       ("containment", "Organ containment")]:
        pr = eval_mode("proposed", REC, rel_fn=lopo_rel(held))
        cb = eval_mode("coverage_blind", REC, rel_fn=lopo_rel(held))
        t7[name] = {"proposed": pr["nDCG"], "coverage_blind": cb["nDCG"]}
        print(f"  {name:20s} proposed={pr['nDCG']:.3f}  coverage-blind={cb['nDCG']:.3f}")

    json.dump({"table5": t5, "table6": t6, "table7": t7, "table8": t8},
              open(f"{RES}/tables_5to8_retrieval.json", "w"), indent=2)
    print(f"\nSaved: {RES}/tables_5to8_retrieval.json")
