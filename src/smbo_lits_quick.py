"""Quick SMBO parameter search for SEMIR on LiTS with C flood-fill + size cap."""
import os, sys, re, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from semir.graph_minor import build_graph_minor
from semir.param_search import boundary_dice

DATA_ROOT = "/scratch/ud3d4/acm_data/Data"

def load_case(vid):
    ct = np.load(os.path.join(DATA_ROOT, "ct", f"volume-{vid}.npy")).astype(np.float32)
    seg = np.load(os.path.join(DATA_ROOT, "seg", f"segmentation-{vid}.npy")).astype(np.int32)
    # HU window [0, 200]
    ct = np.clip(ct, 0, 200).astype(np.float64) / 200.0
    return ct, seg

def main():
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    # Use 10 few-shot volumes (paper uses 20 but we want speed)
    np.random.seed(42)
    all_ids = []
    for f in sorted(os.listdir(os.path.join(DATA_ROOT, "ct"))):
        m = re.match(r"volume-(\d+)\.npy", f)
        if m:
            vid = int(m.group(1))
            if os.path.exists(os.path.join(DATA_ROOT, "seg", f"segmentation-{vid}.npy")):
                all_ids.append(vid)
    
    # Pick 10 with tumor
    few_ids = []
    for vid in np.random.permutation(all_ids):
        _, seg = load_case(vid)
        if (seg == 2).sum() > 100:
            few_ids.append(vid)
        if len(few_ids) >= 10:
            break
    
    print(f"Loading {len(few_ids)} few-shot volumes...", flush=True)
    few_vols, few_segs = [], []
    for vid in few_ids:
        ct, seg = load_case(vid)
        few_vols.append(ct)
        few_segs.append(seg)
        print(f"  vol-{vid}: shape={ct.shape} tumor={(seg==2).sum():,}", flush=True)
    
    best = {"score": -1, "params": {}}
    
    def objective(trial):
        psi = trial.suggest_float("psi", 0.02, 0.20, log=True)
        alpha = trial.suggest_float("alpha", 0.02, 0.30, log=True)
        beta_min = trial.suggest_int("beta_min", 1, 200, log=True)
        # Size cap: beta_max controls max supernode growth
        beta_max = trial.suggest_int("beta_max", 2000, 50000, log=True)
        m_min = trial.suggest_float("m_min", 0.0, 0.15)
        m_max = trial.suggest_float("m_max", 0.85, 1.0)
        
        scores = []
        for vol, seg in zip(few_vols, few_segs):
            try:
                HWD = int(np.prod(vol.shape))
                gm = build_graph_minor(vol, psi=psi, alpha=alpha,
                    beta_min=beta_min, beta_max=beta_max,
                    m_min=m_min, m_max=m_max, method="c")
                
                bd = boundary_dice(gm["labels"], seg, target_label=2)
                n_sn = gm["n_supernodes"]
                
                # Penalize extreme supernode counts
                if n_sn < 100 or n_sn > 50000:
                    bd *= 0.5
                
                scores.append(bd)
            except Exception as e:
                scores.append(0.0)
        
        score = float(np.mean(scores))
        if score > best["score"]:
            best["score"] = score
            best["params"] = {"psi": psi, "alpha": alpha, "beta_min": beta_min,
                              "beta_max": beta_max, "m_min": m_min, "m_max": m_max}
            print(f"  [NEW BEST] trial {trial.number}: bDice={score:.4f} "
                  f"psi={psi:.4f} alpha={alpha:.4f} bmin={beta_min} bmax={beta_max}", flush=True)
        return score
    
    n_trials = int(os.environ.get("SEMIR_N_TRIALS", "100"))
    print(f"\nRunning SMBO with {n_trials} trials...", flush=True)
    
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    print(f"\n{'='*60}")
    print(f"SMBO Complete: {n_trials} trials")
    print(f"Best boundary Dice: {best['score']:.4f}")
    print(f"Best params: {best['params']}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
