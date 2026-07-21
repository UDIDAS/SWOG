"""OAKG vs OAKG-Union ablation — isolates the observation-boundary mechanism.

OAKG and OAKG-Union are identical in every respect (graph, phenotypes, component
similarities, feature weights, ranking policy, queries, candidates, relevance,
masks, seeds, gamma) EXCEPT the treatment of one-sided anatomy:

  * OAKG (intersection): compare only anatomy observed in BOTH cases; a pair with
    no jointly observed anatomy is INCOMPARABLE (ranked at the bottom).
  * OAKG-Union: compare over the UNION of observed anatomy; anatomy observed on
    only one side is completed as ABSENT (0) on the other side, and a numerical
    similarity is always returned.

gamma is always computed from the ORIGINAL (masked) observation scopes for both
methods. Component similarity, weights, and the ranking policy are our existing
ones (group-mean of numeric + categorical components). Statistics (query-cluster
+ seed bootstrap, sign-flip p, Holm) follow the provided reference module.

Run:  python -m oakg.union_ablation --policy lexicographic --n-boot 10000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .data import load_experiment_data, Corpus
from .masking import (build_default_realizations, make_uniform_mask, apply_mask,
                      MaskingRealization)
from .benchmark import candidate_pool
from .oakg import observation_overlap

LEX_BINS = (0.25, 0.50, 0.75)


# --------------------------------------------------------------------------
# Pair similarity with support_mode (reuses our group-mean component similarity)
# --------------------------------------------------------------------------
def _pair_similarity(qi, ci, xf, M, numeric_mask, ranges, mode):
    """Group-mean similarity (numeric group avg + categorical group avg, averaged).

    intersection: features observed in both.  union: features observed in either,
    one-sided completed as absent (0). Returns None if no usable component.
    (Valid for single-organ/global feature supports, as in this benchmark.)
    """
    qm, cm = M[qi], M[ci]
    eff = (qm & cm) if mode == "intersection" else (qm | cm)
    if not eff.any():
        return None
    qv = np.where(qm, xf[qi], 0.0)   # absent=0 where unobserved (matters in union)
    cv = np.where(cm, xf[ci], 0.0)
    comps = []
    nj = eff & numeric_mask
    if nj.any():
        comps.append(float(np.mean(np.maximum(0.0, 1.0 - np.abs(qv[nj] - cv[nj]) / ranges[nj]))))
    sj = eff & ~numeric_mask
    if sj.any():
        comps.append(float(np.mean((qv[sj] == cv[sj]).astype(float))))
    return float(np.mean(comps)) if comps else None


def _rank_key(sim, gamma, policy, gamma_min=0.0):
    if sim is None:
        return (-np.inf,)
    if policy == "similarity":
        return (sim,)
    if policy == "product":
        return (gamma * sim,)
    if policy == "threshold":
        return (sim,) if gamma >= gamma_min else (-np.inf,)
    if policy == "lexicographic":
        cat = float(sum(gamma >= cut for cut in LEX_BINS))
        return (cat, sim)
    raise ValueError(policy)


def _ndcg_at_k(ranked_rel, k=10):
    rel = np.asarray(ranked_rel, float)[:k]
    if rel.size == 0:
        return 0.0
    disc = 1.0 / np.log2(np.arange(2, rel.size + 2))
    dcg = float(np.sum((2.0 ** rel - 1.0) * disc))
    ideal = np.sort(np.asarray(ranked_rel, float))[::-1][:k]
    idisc = 1.0 / np.log2(np.arange(2, ideal.size + 2))
    idcg = float(np.sum((2.0 ** ideal - 1.0) * idisc))
    return dcg / idcg if idcg > 0 else float("nan")


def _score_query(qcase, cands, graded, xf, M, real, corpus, numeric_mask, ranges, policy, mode):
    keys = []
    for order, c in enumerate(cands):
        inter, gamma = observation_overlap(qcase, c, real)
        if mode == "intersection" and not inter:
            key = (-np.inf,)
        else:
            union = real.observation_sets[qcase] | real.observation_sets[c]
            if not union:
                key = (-np.inf,)
            else:
                s = _pair_similarity(corpus.case_to_row[qcase], corpus.case_to_row[c],
                                     xf, M, numeric_mask, ranges, mode)
                key = _rank_key(s, gamma, policy)
        keys.append((key, -order, graded[order]))
    keys.sort(key=lambda x: (x[0], x[1]), reverse=True)
    ranked = [r for _, _, r in keys]
    return _ndcg_at_k(ranked) if any(g > 0 for g in graded) else np.nan


# --------------------------------------------------------------------------
# Evaluation across regimes + the hard-distractor stratum
# --------------------------------------------------------------------------
def _query_rows(qcase, graded_map, realization, xf, M, corpus, nm, ranges, policy, regime, seed):
    cands = candidate_pool(qcase, corpus)
    graded = np.array([graded_map.get(c, 0.0) for c in cands], float)
    rows = []
    for mode, name in [("intersection", "OAKG"), ("union", "OAKG-Union")]:
        rows.append({"regime": regime, "seed": seed, "query_id": qcase, "method": name,
                     "nDCG@10": _score_query(qcase, cands, graded, xf, M, real=realization,
                                             corpus=corpus, numeric_mask=nm, ranges=ranges,
                                             policy=policy, mode=mode)})
    return rows


def evaluate(config: Config, policy: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = load_experiment_data(config.data_dir)
    corpus = Corpus.build(data, config)
    nm = np.array([corpus.feature_type[f] == "numeric" for f in corpus.features])
    ranges = corpus.feature_ranges
    grel = {qid: dict(zip(g.candidate_id, g.graded_relevance))
            for qid, g in data.relevance.groupby("query_id")}
    qcase_of = dict(zip(data.queries.query_id, data.queries.query_case_id))
    graded_by_case = {qcase_of[qid]: grel[qid] for qid in data.queries.query_id}

    rows = []
    # Distinct "seed" per realization WITHIN a regime (asymmetric directions and
    # dataset-style variants otherwise share config.seed and would collide in the
    # paired pivot on (regime, seed, query_id)). seed indexes the realization
    # (fraction / style / direction) within its regime.
    regime_seed: dict[str, int] = {}
    for real in build_default_realizations(data.cases, corpus, config):
        sid = regime_seed.get(real.name, 0); regime_seed[real.name] = sid + 1
        xf, M = apply_mask(corpus.x_ref_full, corpus.m_ref_full, real, corpus)
        for q in data.queries.itertuples(index=False):
            rows += _query_rows(q.query_case_id, graded_by_case[q.query_case_id], real,
                                xf, M, corpus, nm, ranges, policy, real.name, sid)
    main = pd.DataFrame(rows)

    hd = pd.DataFrame(_hard_distractor_rows(data, corpus, config, nm, ranges, policy))
    return main, hd


def _hard_distractor_rows(data, corpus, config, nm, ranges, policy):
    """The 41-query patient-level hard-distractor stratum (regime='hard_distractor')."""
    org = {r.case_id: set(str(r.available_organs).split("|")) for r in data.cases.itertuples(index=False)}
    split = {r.case_id: r.split for r in data.cases.itertuples(index=False)}
    real = make_uniform_mask(data.cases, config.organs, config.seed)
    xf, M = apply_mask(corpus.x_ref_full, corpus.m_ref_full, real, corpus)
    val = lambda c, f: corpus.x_ref_full[corpus.case_to_row[c], corpus.feature_index[f]]
    rows = []
    for TARGET in ["pancreas", "liver"]:
        FEAT = f"{TARGET}_volume_cm3"; fi = corpus.feature_index[FEAT]
        fit = [c for c in corpus.case_order if split[c] in ("train", "val") and TARGET in org[c]]
        thr = float(np.median([val(c, FEAT) for c in fit if np.isfinite(val(c, FEAT))]))
        broad = [c for c in corpus.case_order if len(org[c]) >= 4]
        Q = [c for c in broad if split[c] == "test" and TARGET in org[c]
             and np.isfinite(val(c, FEAT)) and val(c, FEAT) >= thr]
        for qc in Q:
            cands = candidate_pool(qc, corpus)
            graded = np.array([1.0 if (TARGET in org[c] and np.isfinite(val(c, FEAT)) and val(c, FEAT) >= thr) else 0.0
                               for c in cands], float)
            for mode, name in [("intersection", "OAKG"), ("union", "OAKG-Union")]:
                rows.append({"regime": "hard_distractor", "seed": 0, "query_id": f"{TARGET}:{qc}",
                             "method": name,
                             "nDCG@10": _score_query(qc, cands, graded, xf, M, real, corpus, nm, ranges, policy, mode)})
    return rows


# --------------------------------------------------------------------------
# Statistics (ported from the reference module)
# --------------------------------------------------------------------------
def summarize_methods(ql: pd.DataFrame, n_boot=10000, seed=2027) -> pd.DataFrame:
    valid = ql.dropna(subset=["nDCG@10"]); rng = np.random.default_rng(seed); out = []
    for (regime, method), g in valid.groupby(["regime", "method"], sort=True):
        pqs = g[["query_id", "seed", "nDCG@10"]]; qids = pqs.query_id.unique()
        sv = {qid: gg["nDCG@10"].to_numpy(float) for qid, gg in pqs.groupby("query_id")}
        boots = np.array([np.mean([rng.choice(sv[q]) for q in rng.choice(qids, len(qids), replace=True)])
                          for _ in range(n_boot)])
        lo, hi = np.quantile(boots, [0.025, 0.975])
        out.append({"regime": regime, "method": method, "nDCG@10": float(pqs["nDCG@10"].mean()),
                    "ci_low": float(lo), "ci_high": float(hi),
                    "n_queries": int(len(qids)), "n_query_seed_rows": int(len(pqs))})
    return pd.DataFrame(out)


def paired_summary(ql: pd.DataFrame, n_boot=10000, seed=2027) -> pd.DataFrame:
    valid = ql.dropna(subset=["nDCG@10"])
    wide = valid.pivot_table(index=["regime", "seed", "query_id"], columns="method",
                             values="nDCG@10", aggfunc="first").reset_index().dropna(subset=["OAKG", "OAKG-Union"])
    wide["delta_obs"] = wide["OAKG"] - wide["OAKG-Union"]
    rng = np.random.default_rng(seed); rows = []
    for regime, g in wide.groupby("regime", sort=True):
        qids = g.query_id.unique(); byq = {q: gg.delta_obs.to_numpy(float) for q, gg in g.groupby("query_id")}
        boots = np.array([np.mean([rng.choice(byq[q]) for q in rng.choice(qids, len(qids), replace=True)])
                          for _ in range(n_boot)])
        lo, hi = np.quantile(boots, [0.025, 0.975])
        qeff = np.array([np.mean(byq[q]) for q in qids])
        null = np.array([np.mean(rng.choice([-1.0, 1.0], len(qeff)) * qeff) for _ in range(n_boot)])
        p = float((np.sum(np.abs(null) >= abs(np.mean(qeff))) + 1) / (n_boot + 1))
        rows.append({"regime": regime, "OAKG": float(g.OAKG.mean()), "OAKG_Union": float(g["OAKG-Union"].mean()),
                     "delta_obs": float(g.delta_obs.mean()), "ci_low": float(lo), "ci_high": float(hi),
                     "p_value": p, "n_queries": int(len(qids)), "n_query_seed_rows": int(len(g))})
    out = pd.DataFrame(rows)
    out["holm_p"] = _holm(out.p_value.to_numpy(float))
    out["significant_0_05"] = out.holm_p < 0.05
    return out


def _holm(p):
    p = np.asarray(p, float); m = len(p); order = np.argsort(p)
    adj_sorted = np.empty(m); running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx]); adj_sorted[rank] = min(running, 1.0)
    adj = np.empty(m)
    for rank, idx in enumerate(order):
        adj[idx] = adj_sorted[rank]
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results/union_ablation")
    ap.add_argument("--policy", default="lexicographic",
                    choices=["similarity", "product", "threshold", "lexicographic"])
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()
    cfg = Config(use_demo_data=False, data_dir=args.data)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    print(f"policy={args.policy}  n_boot={args.n_boot}")
    main_ql, hd_ql = evaluate(cfg, args.policy)
    main_ql.to_csv(out / "union_ablation_query_level.csv", index=False)

    summarize_methods(main_ql, args.n_boot).to_csv(out / "union_ablation_method_summary.csv", index=False)
    paired = paired_summary(main_ql, args.n_boot)
    paired.to_csv(out / "union_ablation_paired_comparison.csv", index=False)
    hd_paired = paired_summary(hd_ql, args.n_boot)
    hd_paired.to_csv(out / "union_ablation_hard_distractor.csv", index=False)

    (out / "union_ablation_config.json").write_text(json.dumps({
        "policy": args.policy, "policy_note": "validation-selected (lexicographic; tied with product)",
        "n_boot": args.n_boot, "n_queries_total": int(main_ql.query_id.nunique()),
        "query_convention": "zero-relevant queries -> nDCG NaN, excluded (106 evaluable of 111)",
        "incomparable_policy": "bottom", "lex_bins": list(LEX_BINS),
    }, indent=2))
    print("\n=== paired OAKG - OAKG-Union (Δ_obs) ===")
    print(paired[["regime", "OAKG", "OAKG_Union", "delta_obs", "ci_low", "ci_high", "holm_p", "significant_0_05"]].round(4).to_string(index=False))
    print("\n=== hard-distractor ===")
    print(hd_paired[["regime", "OAKG", "OAKG_Union", "delta_obs", "ci_low", "ci_high", "holm_p"]].round(4).to_string(index=False))
    print("DONE")


if __name__ == "__main__":
    main()
