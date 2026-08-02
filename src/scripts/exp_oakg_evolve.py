#!/usr/bin/env python3
"""Experiment B2 — the self-evolving GT-free validation curve.

Claim: the KG validates a NEW, unlabeled patient better as it admits more patients. We hold out a fixed
test set (clean patients + planted realistic segmentation errors) and grow the reference cohort from a
handful to the full corpus. At each size we score every test patient's plausibility — NO ground truth
used — and measure how well it separates clean from corrupted.

Features: the 5 core organs (full coverage) × {log-volume, log-max-diameter} = 10-D. This encodes a
real error signature — an over-segmentation leak inflates VOLUME but not the max-DIAMETER, breaking
their tight physical relationship. Planted error = scale one organ's volume (a leak), diameter intact.

Two plausibility models:
  • MARGINAL (baseline) — max |z| of each feature vs the cohort's per-feature mean/sd. Needs almost no
    data, so it SATURATES immediately (flat curve) and can't see a volume-vs-diameter inconsistency —
    a real, instructive negative.
  • JOINT / Mahalanobis (the KG's model) — uses the cohort COVARIANCE, so it catches errors that break
    the correlation structure (volume up, diameter flat) which the marginal check misses. The 10-D
    covariance genuinely needs data, so this validation IMPROVES as the KG grows.

The contrast is the point: accumulating patients doesn't sharpen a mean (saturates) — it sharpens the
JOINT phenotype model, which is exactly what a growing KG accumulates. Vectorized, CPU, seconds.
"""
import json
import os

import numpy as np

CORPUS = "/home/ud3d4/Desktop/SWOG/kg/data/corpus_flare_train.json"
ORGANS = ["liver", "right_kidney", "spleen", "pancreas", "left_kidney"]
NTEST, SEEDS = 200, 8
# realistic autonomous-segmentation error on ONE organ: moderate leak / truncation. Deliberately kept
# near the marginal range so it OFTEN stays individually plausible but breaks the joint correlation.
OVER = (1.4, 1.9)      # ×volume when a mask leaks into neighbours
UNDER = (0.45, 0.70)   # ×volume when a structure is truncated
OUT = "/home/ud3d4/Desktop/SWOG/results"


VOL_COLS = list(range(0, 2 * len(ORGANS), 2))         # even columns = log-volume (odd = log-diameter)


def load_features():
    recs = json.load(open(CORPUS))["records"]
    rows = []
    for r in recs:
        o = r["organs"]
        if not all(x in o and o[x].get("organ_volume_cm3") and o[x].get("organ_max_diameter_mm")
                   for x in ORGANS):
            continue
        feat = []
        for x in ORGANS:
            feat += [o[x]["organ_volume_cm3"], o[x]["organ_max_diameter_mm"]]
        rows.append(feat)
    return np.log(np.array(rows, float))              # log space: volume/diameter relations near-linear


def corrupt(test, rng):
    """Plant a realistic over-/under-segmentation LEAK: scale one organ's VOLUME, leave its diameter —
    breaks the volume-diameter relation (invisible to marginal checks, caught by the joint model)."""
    C = test.copy()
    org = rng.randint(0, len(ORGANS), size=len(test))
    vol_col = np.array(VOL_COLS)[org]
    over = rng.rand(len(test)) < 0.5
    fac = np.where(over, rng.uniform(*OVER, len(test)), rng.uniform(*UNDER, len(test)))
    C[np.arange(len(test)), vol_col] += np.log(fac)
    return C


def auroc(pos, neg):
    """AUC = P(score_corrupted > score_clean). Rank-based, vectorized."""
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort().astype(float) + 1
    rp = ranks[:len(pos)].sum()
    return (rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def score_marginal(X, cohort):
    mu, sd = cohort.mean(0), cohort.std(0) + 1e-6
    return np.max(np.abs((X - mu) / sd), axis=1)      # worst per-organ z (marginal)


def score_joint(X, cohort):
    """Mahalanobis distance to the cohort — uses the covariance (needs data to estimate)."""
    mu = cohort.mean(0)
    d = X.shape[1]
    cov = np.cov(cohort, rowvar=False) + 1e-3 * np.eye(d)   # minimal ridge for invertibility
    P = np.linalg.inv(cov)
    D = X - mu
    return np.einsum("ij,jk,ik->i", D, P, D)          # squared Mahalanobis


def main():
    os.makedirs(OUT, exist_ok=True)
    V = load_features()
    n = len(V)
    print(f"feature dim: {V.shape[1]} (5 organs × [log-vol, log-diam])", flush=True)
    sizes = [15, 30, 60, 120, 240, 480, 960, n - NTEST]
    sizes = sorted({s for s in sizes if s >= 5 and s <= n - NTEST})
    print(f"patients: {n} | fixed test={NTEST} | cohort sizes: {sizes}", flush=True)

    auc = {"marginal": {s: [] for s in sizes}, "joint": {s: [] for s in sizes}}
    for seed in range(SEEDS):
        rng = np.random.RandomState(seed)
        idx = rng.permutation(n)
        test, pool = V[idx[:NTEST]], V[idx[NTEST:]]
        Ct = corrupt(test, rng)
        for s in sizes:
            cohort = pool[:s]
            auc["marginal"][s].append(auroc(score_marginal(Ct, cohort), score_marginal(test, cohort)))
            auc["joint"][s].append(auroc(score_joint(Ct, cohort), score_joint(test, cohort)))

    curve = {"sizes": sizes}
    for m in ("marginal", "joint"):
        curve[m] = [round(float(np.mean(auc[m][s])), 3) for s in sizes]
        curve[m + "_std"] = [round(float(np.std(auc[m][s])), 3) for s in sizes]
    for i, s in enumerate(sizes):
        print(f"N={s:5d}  marginal AUROC={curve['marginal'][i]:.3f}  "
              f"JOINT(Mahalanobis) AUROC={curve['joint'][i]:.3f}±{curve['joint_std'][i]:.3f}", flush=True)
    lift = curve["joint"][-1] - curve["joint"][0]
    print(f"joint-model lift from N={sizes[0]} to N={sizes[-1]}: +{lift:.3f} AUROC "
          f"(marginal stays flat at ~{np.mean(curve['marginal']):.2f})", flush=True)
    json.dump(curve, open(f"{OUT}/oakg_evolve.json", "w"), indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7.5, 4.8))
    plt.plot(sizes, curve["joint"], "-o", color="#1e7a3c", lw=2.6, label="JOINT plausibility (KG covariance)")
    plt.fill_between(sizes, np.array(curve["joint"]) - np.array(curve["joint_std"]),
                     np.array(curve["joint"]) + np.array(curve["joint_std"]), alpha=0.15, color="#1e7a3c")
    plt.plot(sizes, curve["marginal"], "--s", color="#888", lw=1.8, label="marginal per-organ z (baseline)")
    plt.xscale("log")
    plt.xlabel("patients admitted to the KG (log scale)")
    plt.ylabel("clean-vs-error AUROC  (GT-free validation)")
    plt.title("The KG validates new patients better as it grows")
    plt.legend(fontsize=9); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(f"{OUT}/oakg_evolve.png", dpi=130)
    print(f"-> {OUT}/oakg_evolve.json + oakg_evolve.png", flush=True)


if __name__ == "__main__":
    main()
