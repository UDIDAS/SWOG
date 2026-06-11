"""Run boundary Dice parameter search with liver window + recursive contraction."""
import os, sys, json, numpy as np, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from semir.graph_minor import build_graph_minor
from semir.param_search import boundary_dice

DATA_ROOT = "/scratch/ud3d4/acm_data/Data"
RESULTS_DIR = "/home/ud3d4/Desktop/SWOG/results/semir_lits"

def load(vid):
    ct = np.load(os.path.join(DATA_ROOT, "ct", f"volume-{vid}.npy")).astype(np.float32)
    seg = np.load(os.path.join(DATA_ROOT, "seg", f"segmentation-{vid}.npy")).astype(np.int32)
    ct = np.clip(ct, 0, 200) / 200.0  # liver window
    return ct, seg

def pr(msg=""):
    print(msg, flush=True)

# Use 5 few-shot volumes with decent tumor content
few_ids = [74, 4, 16, 28, 56]
pr(f"Loading {len(few_ids)} few-shot volumes (liver window [0,200])...")
vols, segs = [], []
for vid in few_ids:
    ct, seg = load(vid)
    tu = (seg == 2).sum()
    pr(f"  vol-{vid}: {ct.shape}, tumor={tu:,}")
    vols.append(ct)
    segs.append(seg)

# Parameter grid matching paper: 64 per parameter would be ~260K evals
# Use 20 x 10 x 8 = 1600 for practical search
k_psi, k_alpha, k_beta = 20, 10, 8
psi_vals = np.linspace(0.005, 0.15, k_psi)
alpha_vals = np.linspace(0.01, 0.25, k_alpha)
beta_min_vals = np.logspace(0.5, 3, k_beta).astype(int)  # 3 to 1000
beta_max = 500000
m_min, m_max = 0.0, 1.0

total = k_psi * k_alpha * k_beta
pr(f"\nBoundary Dice search: {k_psi}x{k_alpha}x{k_beta}={total} configs")
pr(f"ψ range: [{psi_vals[0]:.3f}, {psi_vals[-1]:.3f}]")

log = []
best_score = -1
best_params = {}
t0 = time.time()
evaluated = 0

for pi, psi in enumerate(psi_vals):
    psi_best = -1
    for alpha in alpha_vals:
        if alpha < psi:  # paper: infeasible if α ≤ ψ... actually paper says exclude α ≤ ψ
            continue
        for beta_min in beta_min_vals:
            dices = []
            n_sn_list = []
            for vol, seg in zip(vols, segs):
                gm = build_graph_minor(vol, psi=psi, alpha=alpha,
                                       beta_min=int(beta_min), beta_max=beta_max,
                                       m_min=m_min, m_max=m_max, fast=True)
                d = boundary_dice(gm["labels"], seg, target_label=2)
                dices.append(d)
                n_sn_list.append(gm["n_supernodes"])

            mean_bd = float(np.mean(dices))
            mean_sn = float(np.mean(n_sn_list))

            entry = {
                "psi": round(float(psi), 4),
                "alpha": round(float(alpha), 4),
                "beta_min": int(beta_min),
                "boundary_dice": round(mean_bd, 4),
                "mean_supernodes": round(mean_sn, 0),
            }
            log.append(entry)

            if mean_bd > best_score:
                best_score = mean_bd
                best_params = {
                    "psi": round(float(psi), 4),
                    "alpha": round(float(alpha), 4),
                    "beta_min": int(beta_min),
                    "beta_max": beta_max,
                    "m_min": m_min, "m_max": m_max,
                }

            evaluated += 1

    if psi_best < best_score:
        psi_best = best_score
    elapsed = time.time() - t0
    pr(f"  ψ={psi:.3f} done ({evaluated}/{total}, {elapsed:.0f}s) "
       f"best_bdice={best_score:.4f} best_sn={best_params.get('psi','?')}")

elapsed = time.time() - t0
pr(f"\nSearch complete: {evaluated} configs in {elapsed:.1f}s")
pr(f"Best: boundary_dice={best_score:.4f}")
pr(f"  ψ={best_params['psi']}  α={best_params['alpha']}  β_min={best_params['beta_min']}")

# Show top 10
sorted_log = sorted(log, key=lambda x: -x["boundary_dice"])
pr(f"\nTop-10 configs:")
pr(f"  {'ψ':>8} {'α':>8} {'β_min':>6} {'BDice':>8} {'#SN':>10}")
for e in sorted_log[:10]:
    pr(f"  {e['psi']:8.4f} {e['alpha']:8.4f} {e['beta_min']:6d} "
       f"{e['boundary_dice']:8.4f} {e['mean_supernodes']:10.0f}")

# Save
os.makedirs(RESULTS_DIR, exist_ok=True)
with open(os.path.join(RESULTS_DIR, "boundary_search.json"), "w") as f:
    json.dump({"best_params": best_params, "best_score": best_score, "log": sorted_log[:50]},
              f, indent=2)
pr(f"\nSaved to {RESULTS_DIR}/boundary_search.json")
