"""
Parallel SMBO parameter search for SEMIR on LiTS.
Uses joblib to parallelize volume processing within each trial.
Uses fewer volumes (5) and trials (50) for speed.
"""
import os, sys, re, time
import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from semir.graph_minor import build_graph_minor
from semir.param_search import boundary_dice

DATA_ROOT = "/scratch/ud3d4/acm_data/Data"


def load_case(vid):
    ct = np.load(os.path.join(DATA_ROOT, "ct", f"volume-{vid}.npy")).astype(np.float32)
    seg = np.load(os.path.join(DATA_ROOT, "seg", f"segmentation-{vid}.npy")).astype(np.int32)
    ct = np.clip(ct, 0, 200).astype(np.float64) / 200.0
    return ct, seg


def eval_one_volume(vol, seg, psi, alpha, beta_min, beta_max, m_min, m_max):
    """Evaluate boundary Dice on a single volume."""
    try:
        gm = build_graph_minor(vol, psi=psi, alpha=alpha,
                               beta_min=beta_min, beta_max=beta_max,
                               m_min=m_min, m_max=m_max, method="c")
        bd = boundary_dice(gm["labels"], seg, target_label=2)
        n_sn = gm["n_supernodes"]
        return bd, n_sn
    except Exception:
        return 0.0, 0


def main():
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Discover volumes with tumor
    np.random.seed(42)
    all_ids = []
    for f in sorted(os.listdir(os.path.join(DATA_ROOT, "ct"))):
        m = re.match(r"volume-(\d+)\.npy", f)
        if m:
            vid = int(m.group(1))
            if os.path.exists(os.path.join(DATA_ROOT, "seg", f"segmentation-{vid}.npy")):
                all_ids.append(vid)

    # Pick 5 with decent tumor (speed vs coverage trade-off)
    few_ids = []
    for vid in np.random.permutation(all_ids):
        _, seg = load_case(vid)
        if (seg == 2).sum() > 500:
            few_ids.append(vid)
        if len(few_ids) >= 5:
            break

    print(f"Loading {len(few_ids)} few-shot volumes...", flush=True)
    few_vols, few_segs = [], []
    for vid in few_ids:
        ct, seg = load_case(vid)
        few_vols.append(ct)
        few_segs.append(seg)
        print(f"  vol-{vid}: shape={ct.shape} tumor={(seg==2).sum():,}", flush=True)

    n_cpus = min(5, os.cpu_count() or 4)
    print(f"Using {n_cpus} parallel workers", flush=True)

    best = {"score": -1, "params": {}, "n_sn": 0}

    def objective(trial):
        psi = trial.suggest_float("psi", 0.02, 0.20, log=True)
        alpha = trial.suggest_float("alpha", 0.02, 0.30, log=True)
        beta_min = trial.suggest_int("beta_min", 1, 200, log=True)
        beta_max = trial.suggest_int("beta_max", 2000, 50000, log=True)
        m_min = trial.suggest_float("m_min", 0.0, 0.15)
        m_max = trial.suggest_float("m_max", 0.85, 1.0)

        # Parallel evaluation across volumes
        results = Parallel(n_jobs=n_cpus)(
            delayed(eval_one_volume)(vol, seg, psi, alpha, beta_min, beta_max, m_min, m_max)
            for vol, seg in zip(few_vols, few_segs)
        )

        bds = [r[0] for r in results]
        sns = [r[1] for r in results]
        score = float(np.mean(bds))
        mean_sn = float(np.mean(sns))

        # Penalize extreme supernode counts
        if mean_sn < 100 or mean_sn > 50000:
            score *= 0.5

        if score > best["score"]:
            best["score"] = score
            best["n_sn"] = mean_sn
            best["params"] = {"psi": psi, "alpha": alpha, "beta_min": beta_min,
                              "beta_max": beta_max, "m_min": m_min, "m_max": m_max}
            print(f"  [BEST] trial {trial.number}: bDice={score:.4f} "
                  f"psi={psi:.4f} alpha={alpha:.4f} bmin={beta_min} bmax={beta_max} "
                  f"mean_SN={mean_sn:.0f}", flush=True)
        return score

    n_trials = int(os.environ.get("SEMIR_N_TRIALS", "50"))
    t0 = time.time()
    print(f"\nRunning SMBO ({n_trials} trials, {len(few_ids)} vols, {n_cpus} workers)...", flush=True)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    dt = time.time() - t0
    print(f"\n{'='*60}", flush=True)
    print(f"SMBO Complete: {n_trials} trials in {dt:.0f}s ({dt/60:.1f} min)", flush=True)
    print(f"Best boundary Dice: {best['score']:.4f}", flush=True)
    print(f"Best mean supernodes: {best['n_sn']:.0f}", flush=True)
    print(f"Best params: {best['params']}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
