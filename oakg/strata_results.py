"""Generate compact CSVs for the supplementary strata findings.

Writes small, review-friendly summary tables (method, nDCG@10, paired delta vs
zero-imputation with 95% CI) for:
  - hard_distractor.csv            : scaled hard-distractor stratum (patient-level)
  - hard_distractor_adversarial.csv: adversarial construction, all OAKG policies
  - flare_tumor_realgt.csv         : real-GT FLARE cross-organ tumor stratum

Run:  python -m oakg.strata_results --out results/strata
Requires the patient-level benchmark in data/ and (for the tumor table) the
tumor stratum in data_tumor/ (see oakg.build_tumor_stratum).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .data import load_experiment_data, Corpus
from .masking import make_uniform_mask, make_random_mask, apply_mask
from .benchmark import candidate_pool
from .baselines import (zero_imputed_similarity, masked_cosine_similarity,
                        missing_indicator_similarity, mean_imputed_similarity)
from .oakg import oakg_scores
from .graph import wl_embeddings
from .neural import embedding_scores
from .metrics import ndcg_at_k
from .stats import paired_bootstrap_difference

N_BOOT = 10000  # final study setting (matches pipeline)


def _summary(df: pd.DataFrame, ref_method: str = "ZeroImp") -> pd.DataFrame:
    rows = []
    means = df.groupby("method")["nDCG@10"].mean()
    served = df.groupby("method")["nDCG@10"].apply(lambda s: float(np.isfinite(s).mean()))
    for m in means.sort_values(ascending=False, na_position="last").index:
        if m == ref_method:
            d = {"difference": 0.0, "ci_low": 0.0, "ci_high": 0.0}
        elif served[m] == 0:               # method abstained on every query
            d = {"difference": np.nan, "ci_low": np.nan, "ci_high": np.nan}
        else:
            d = paired_bootstrap_difference(df, m, ref_method, n_bootstrap=N_BOOT, seed=7)
        sig = "" if m == ref_method else (
            "abstains" if served[m] == 0 else
            ("SIG" if (d["ci_low"] > 0 or d["ci_high"] < 0) else "ns"))
        rows.append({"method": m, "nDCG@10": round(float(means[m]), 4) if np.isfinite(means[m]) else np.nan,
                     f"delta_vs_{ref_method}": round(d["difference"], 4) if np.isfinite(d["difference"]) else np.nan,
                     "ci_low": round(d["ci_low"], 4) if np.isfinite(d["ci_low"]) else np.nan,
                     "ci_high": round(d["ci_high"], 4) if np.isfinite(d["ci_high"]) else np.nan,
                     "served_rate": round(served[m], 3), "sig": sig})
    return pd.DataFrame(rows)


def _load(data_dir):
    cfg = Config(use_demo_data=False, data_dir=data_dir)
    data = load_experiment_data(cfg.data_dir); corpus = Corpus.build(data, cfg)
    real = make_uniform_mask(data.cases, cfg.organs, cfg.seed)
    X, M = apply_mask(corpus.x_ref_full, corpus.m_ref_full, real, corpus)
    wl_emb, _ = wl_embeddings(X, M, corpus)
    wl = {c: wl_emb[i] for i, c in enumerate(corpus.case_order)}
    return cfg, data, corpus, real, X, M, wl


def _score_row(qc, cands, qidx, cidx, g, X, M, real, corpus, wl, policies=None):
    out = {"ZeroImp": zero_imputed_similarity(qidx, cidx, X, M, corpus),
           "MeanImp": mean_imputed_similarity(qidx, cidx, X, M, corpus),
           "MissInd": missing_indicator_similarity(qidx, cidx, X, M, corpus),
           "MaskedCos": masked_cosine_similarity(qidx, cidx, X, M, corpus),
           "WL": embedding_scores(qc, cands, wl)}
    if policies:
        for p in policies:
            out[f"OAKG-{p}"] = oakg_scores(qc, cands, X, M, real, corpus, policy=p)[0]
    else:
        out["OAKG"] = oakg_scores(qc, cands, X, M, real, corpus, policy="product")[0]
    return out


def _ndcg(method: str, g: np.ndarray, s: np.ndarray) -> float:
    """nDCG@10. OAKG methods use incomparable_policy=bottom (keep the full pool,
    push observation-incomparable candidates below every eligible score, so the
    ideal-DCG is over the full pool) — consistent with oakg.benchmark and the
    union ablation. Baselines have no observation-eligibility, so their non-finite
    scores are dropped."""
    v = np.isfinite(s)
    if not v.any():
        # all-query convention: a query with no comparable candidate still counts,
        # scoring 0 where relevant candidates exist (NaN only if none are relevant).
        return 0.0 if g.sum() > 0 else np.nan
    if method.startswith("OAKG"):
        floor = float(s[v].min()) - 1.0
        return ndcg_at_k(g, np.where(v, s, floor))
    return ndcg_at_k(g[v], s[v])


def tumor_stratum(out_dir, data_dir="data_tumor"):
    cfg, data, corpus, real, X, M, wl = _load(data_dir)
    recs = []
    for q in data.queries.itertuples(index=False):
        qc = q.query_case_id; cands = candidate_pool(qc, corpus)
        cidx = np.array([corpus.case_to_row[c] for c in cands]); qidx = corpus.case_to_row[qc]
        rel = data.relevance[data.relevance.query_id == q.query_id].set_index("candidate_id")
        g = rel.reindex(cands)["graded_relevance"].fillna(0).to_numpy(float)
        if g.sum() == 0:
            continue
        for m, s in _score_row(qc, cands, qidx, cidx, g, X, M, real, corpus, wl).items():
            recs.append({"query_id": q.query_id, "method": m, "nDCG@10": _ndcg(m, g, s)})
    _summary(pd.DataFrame(recs)).to_csv(out_dir / "flare_tumor_realgt.csv", index=False)
    print("wrote flare_tumor_realgt.csv")


def hard_distractor(out_dir, adversarial: bool):
    cfg, data, corpus, real, X, M, wl = _load("data")
    org = {r.case_id: set(str(r.available_organs).split("|")) for r in data.cases.itertuples(index=False)}
    split = {r.case_id: r.split for r in data.cases.itertuples(index=False)}
    val = lambda c, fi: corpus.x_ref_full[corpus.case_to_row[c], fi]
    policies = ["similarity", "product", "threshold", "lexicographic"] if adversarial else None
    recs = []
    for TARGET in ["pancreas", "liver"]:
        fi = corpus.feature_index[f"{TARGET}_volume_cm3"]
        fit = [c for c in corpus.case_order if split[c] in ("train", "val") and TARGET in org[c]]
        thr = float(np.median([val(c, fi) for c in fit if np.isfinite(val(c, fi))]))
        broad = [c for c in corpus.case_order if len(org[c]) >= 4]
        narrow = [c for c in corpus.case_order if org[c] == {TARGET}]
        Q = [c for c in broad if split[c] == "test" and TARGET in org[c] and np.isfinite(val(c, fi)) and val(c, fi) >= thr]
        for qc in Q:
            if adversarial:
                relevant = [c for c in narrow if np.isfinite(val(c, fi)) and val(c, fi) >= thr]
                distract = [c for c in broad if np.isfinite(val(c, fi)) and val(c, fi) < thr]
                cands = [c for c in relevant + distract if c != qc]
                g = np.array([1.0 if c in relevant else 0.0 for c in cands])
            else:
                cands = candidate_pool(qc, corpus)
                g = np.array([1.0 if (TARGET in org[c] and np.isfinite(val(c, fi)) and val(c, fi) >= thr) else 0.0 for c in cands])
            if g.sum() == 0:
                continue
            cidx = np.array([corpus.case_to_row[c] for c in cands]); qidx = corpus.case_to_row[qc]
            for m, s in _score_row(qc, cands, qidx, cidx, g, X, M, real, corpus, wl, policies).items():
                recs.append({"query_id": f"{TARGET}:{qc}", "method": m, "nDCG@10": _ndcg(m, g, s)})
    name = "hard_distractor_adversarial.csv" if adversarial else "hard_distractor.csv"
    _summary(pd.DataFrame(recs)).to_csv(out_dir / name, index=False)
    print("wrote", name)


def backbone_comparison(out_dir, frac: float = 0.40):
    """All retrieval backbones incl. CompGCN at one masking level (random, ref).

    CompGCN is trained here (needs torch + torch_geometric); skipped gracefully if
    unavailable. Observed-region CT is a separate 3D pipeline, not included.
    """
    cfg, data, corpus, real, X, M, wl = _load("data")  # uses uniform mask; re-mask below
    real = make_random_mask(data.cases, frac, cfg.seed + int(frac * 100))
    X, M = apply_mask(corpus.x_ref_full, corpus.m_ref_full, real, corpus)
    wl_emb, _ = wl_embeddings(X, M, corpus); wl = {c: wl_emb[i] for i, c in enumerate(corpus.case_order)}
    comp = None
    try:
        from .compgcn import compgcn_embeddings
        split = {r.case_id: r.split for r in data.cases.itertuples(index=False)}
        train_ids = [c for c in corpus.case_order if split[c] in ("train", "val")]
        comp = compgcn_embeddings(X, M, corpus, train_ids, dim=32, epochs=200)
    except Exception as e:
        print("CompGCN skipped:", type(e).__name__, str(e)[:80])
    recs = []
    for q in data.queries.itertuples(index=False):
        qc = q.query_case_id; cands = candidate_pool(qc, corpus)
        cidx = np.array([corpus.case_to_row[c] for c in cands]); qidx = corpus.case_to_row[qc]
        rel = data.relevance[data.relevance.query_id == q.query_id].set_index("candidate_id")
        g = rel.reindex(cands)["graded_relevance"].fillna(0).to_numpy(float)
        if g.sum() == 0:
            continue
        sc = _score_row(qc, cands, qidx, cidx, g, X, M, real, corpus, wl)
        if comp is not None:
            sc["CompGCN"] = embedding_scores(qc, cands, comp)
        for m, s in sc.items():
            recs.append({"query_id": q.query_id, "method": m, "nDCG@10": _ndcg(m, g, s)})
    _summary(pd.DataFrame(recs)).to_csv(out_dir / "backbone_comparison.csv", index=False)
    print("wrote backbone_comparison.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/strata")
    ap.add_argument("--backbones", action="store_true", help="also train+compare CompGCN backbone")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    hard_distractor(out, adversarial=False)
    hard_distractor(out, adversarial=True)
    tumor_stratum(out)
    if args.backbones:
        backbone_comparison(out)
    print("DONE")


if __name__ == "__main__":
    main()
