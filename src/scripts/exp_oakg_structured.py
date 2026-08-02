#!/usr/bin/env python3
"""Experiment A2 (corrected) — OAKG under STRUCTURED partial-observability, measuring FALSE POSITIVES.

Fixes the first attempt: (1) missingness is STRUCTURED — patients belong to "datasets" that observe
fixed, disjoint organ subsets (Pancreas→pancreas, LiTS→liver, KiTS→kidneys, FLARE→all), mirroring the
real merge, so joint-observability (γ) actually varies; (2) the metric is the FALSE-POSITIVE rate among
retrieved top-k (spurious matches), which is OAKG's actual claim — not full-ranking recovery.

Vectorized: similarity is computed in blocks per (source_a, source_b) pair via matrix products — no
O(n^2) Python loop. γ ablation is built in (masked-cosine = OAKG without γ).
"""
import json
import os

import numpy as np

CORPUS = "/home/ud3d4/Desktop/SWOG/kg/data/corpus_flare_train.json"
ORGANS = ["liver", "right_kidney", "spleen", "pancreas", "left_kidney"]   # indices 0..4
SRC = {"liver": {0}, "kidney": {1, 4}, "pancreas": {3}, "all": {0, 1, 2, 3, 4}}  # dataset-like obs sets
DISJOINT = ["liver", "kidney", "pancreas"]           # the single-site "datasets"
K, KTRUE, SEEDS = 10, 20, 5
OUT = "/home/ud3d4/Desktop/SWOG/results"


def load_full():
    recs = json.load(open(CORPUS))["records"]
    V = [[o[x]["organ_volume_cm3"] for x in ORGANS] for r in recs
         for o in [r["organs"]] if all(x in o and o[x].get("organ_volume_cm3") for x in ORGANS)]
    V = np.array(V, float)
    return (V - V.mean(0)) / (V.std(0) + 1e-9)


def cos_block(A, B):
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    return An @ Bn.T


def sim(Z, src_of, method):
    """Vectorized n×n similarity. src_of[i] = source name of patient i."""
    n, d = Z.shape
    S = np.zeros((n, n))
    names = list(SRC)
    rows = {s: np.where(np.array(src_of) == s)[0] for s in names}
    if method in ("zero", "mean"):
        Zf = Z.copy()
        for i in range(n):
            obs = SRC[src_of[i]]
            for f in range(d):
                if f not in obs:
                    Zf[i, f] = (0 - 0) if method == "mean" else (Z[:, f].min() - 1)  # mean=0 (z), zero=below-range
        return cos_block(Zf, Zf)
    for a in names:
        for b in names:
            ra, rb = rows[a], rows[b]
            if len(ra) == 0 or len(rb) == 0:
                continue
            shared = SRC[a] & SRC[b]
            if not shared:
                continue                                  # no shared evidence -> similarity 0
            idx = list(shared)
            if method == "gower":
                A, B = Z[ra][:, idx], Z[rb][:, idx]
                rng = Z[:, idx].max(0) - Z[:, idx].min(0) + 1e-9
                blk = 1 - np.mean(np.abs(A[:, None, :] - B[None, :, :]) / rng, axis=2)
            else:
                blk = cos_block(Z[ra][:, idx], Z[rb][:, idx])
                if method == "oakg":
                    blk = blk * (len(shared) / len(SRC[a] | SRC[b]))   # γ = joint-observability weight
            S[np.ix_(ra, rb)] = blk
    return S


def evaluate(Z, src_of, gt):
    """Returns per-method (precision@k, FP-rate@k, thin-overlap spurious-match rate@k)."""
    n = Z.shape[0]
    shared_n = np.array([[len(SRC[src_of[i]] & SRC[src_of[j]]) for j in range(n)] for i in range(n)])
    res = {}
    for m in ["zero", "mean", "gower", "masked", "oakg"]:
        S = sim(Z, src_of, m)
        np.fill_diagonal(S, -1e9)
        prec, fp, spur = [], [], []
        for i in range(n):
            ret = np.argsort(-S[i])[:K]
            hit = len(set(ret) & gt[i])
            prec.append(hit / K)
            fp.append((K - hit) / K)                       # false discoveries among top-k
            # spurious thin-overlap match: retrieved, shares <=1 organ with query, NOT a true neighbour
            spur.append(np.mean([(shared_n[i, j] <= 1) and (j not in gt[i]) for j in ret]))
        res[m] = (float(np.mean(prec)), float(np.mean(fp)), float(np.mean(spur)))
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    Z = load_full()
    if len(Z) > 600:
        Z = Z[np.random.RandomState(0).choice(len(Z), 600, replace=False)]
    n = len(Z)
    gt = [set(np.argsort(-r)[1:KTRUE + 1]) for r in cos_block(Z, Z)]   # true neighbours on FULL vector
    print(f"patients: {n}  | true-neighbour set K={KTRUE}, retrieve k={K}", flush=True)

    levels = [0.0, 0.25, 0.5, 0.75, 1.0]                 # fraction assigned to single-site datasets
    curve = {m: {"prec": [], "fp": [], "spur": []} for m in ["zero", "mean", "gower", "masked", "oakg"]}
    for t in levels:
        agg = {m: [] for m in curve}
        for seed in range(SEEDS):
            rng = np.random.RandomState(seed)
            src_of = []
            for _ in range(n):
                if rng.rand() < t:
                    src_of.append(DISJOINT[rng.randint(len(DISJOINT))])
                else:
                    src_of.append("all")
            r = evaluate(Z, src_of, gt)
            for m in curve:
                agg[m].append(r[m])
        for m in curve:
            p = np.mean([x[0] for x in agg[m]]); f = np.mean([x[1] for x in agg[m]]); s = np.mean([x[2] for x in agg[m]])
            curve[m]["prec"].append(round(float(p), 3)); curve[m]["fp"].append(round(float(f), 3))
            curve[m]["spur"].append(round(float(s), 3))
        print(f"het={t:.2f}  " + "  ".join(f"{m}:P{curve[m]['prec'][-1]:.2f}/spur{curve[m]['spur'][-1]:.2f}"
                                          for m in curve), flush=True)

    json.dump({"levels": levels, "curve": curve, "k": K}, open(f"{OUT}/oakg_structured.json", "w"), indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = {"zero": "zero-impute", "mean": "mean-impute", "gower": "Gower",
             "masked": "masked-cosine (OAKG w/o γ)", "oakg": "OAKG (masked×γ)"}
    st = {"oakg": dict(lw=3, color="#1e7a3c", marker="o"), "masked": dict(lw=1.8, color="#e08e0b", marker="s"),
          "gower": dict(lw=1.5, color="#8e44ad", marker="^"), "mean": dict(lw=1.5, color="#2980b9", marker="v"),
          "zero": dict(lw=1.5, color="#c0392b", marker="x")}
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for m in curve:
        ax[0].plot(levels, curve[m]["prec"], label=names[m], **st[m])
        ax[1].plot(levels, curve[m]["spur"], label=names[m], **st[m])
    ax[0].set_ylabel(f"Precision@{K} vs. full-observation ranking")
    ax[0].set_title("Retrieval fidelity (higher better)")
    ax[1].set_ylabel(f"Spurious thin-overlap matches @ top-{K}")
    ax[1].set_title("False positives OAKG must suppress (lower better)")
    for a in ax:
        a.set_xlabel("heterogeneity  (fraction from single-site datasets →)")
        a.legend(fontsize=8); a.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{OUT}/oakg_structured_fp.png", dpi=130)
    print(f"-> {OUT}/oakg_structured.json + oakg_structured_fp.png", flush=True)


if __name__ == "__main__":
    main()
