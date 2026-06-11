"""
Few-shot parameter optimisation for SEMIR graph minor construction.

Searches over (ψ, α, β) to maximise boundary Dice on 5-20 labelled examples.
Each parameter is discretised to K values; total search is O(K³) with K~16-32
(practical: <5 min on CPU for |D_few|=5).
"""

import numpy as np
from .graph_minor import build_graph_minor


def boundary_dice(labels: np.ndarray, gt: np.ndarray, target_label: int = 2):
    """
    SEMIR's actual objective (Eq. 1 in the paper):
    Dice between supernode partition boundaries and GT target boundaries.

    This measures how well supernode edges align with semantic edges,
    independent of classification. Vectorised — no Python loops.
    """
    from scipy.ndimage import binary_erosion, generate_binary_structure
    struct = generate_binary_structure(3, 1)

    gt_mask = (gt == target_label)
    if gt_mask.sum() == 0:
        return 1.0

    # Predicted tumor mask via LUT (vectorised)
    flat_labels = labels.ravel()
    flat_gt = gt_mask.ravel().astype(np.float64)
    max_id = int(flat_labels.max())
    if max_id == 0:
        return 0.0

    counts = np.bincount(flat_labels, minlength=max_id + 1)
    tumor_counts = np.bincount(flat_labels, weights=flat_gt, minlength=max_id + 1)
    safe = np.where(counts > 0, counts, 1)
    overlap_frac = tumor_counts / safe

    # LUT: supernode → is_tumor
    tumor_lut = np.zeros(max_id + 1, dtype=bool)
    tumor_lut[1:] = overlap_frac[1:] > 0.1
    pred_mask = tumor_lut[flat_labels].reshape(labels.shape)

    # Boundary Dice
    gt_boundary = gt_mask & ~binary_erosion(gt_mask, structure=struct)
    pred_boundary = pred_mask & ~binary_erosion(pred_mask, structure=struct)

    if gt_boundary.sum() == 0 and pred_boundary.sum() == 0:
        return 1.0

    intersection = (gt_boundary & pred_boundary).sum()
    return float(2 * intersection / (gt_boundary.sum() + pred_boundary.sum() + 1e-8))


def volume_dice(labels: np.ndarray, gt: np.ndarray, target_label: int = 2):
    """Standard volumetric Dice between predicted tumor mask and GT. Vectorised."""
    gt_mask = (gt == target_label)
    if gt_mask.sum() == 0:
        return 1.0

    flat_labels = labels.ravel()
    flat_gt = gt_mask.ravel().astype(np.float64)
    max_id = int(flat_labels.max())
    if max_id == 0:
        return 0.0

    # Per-supernode: count total voxels and tumor-overlapping voxels
    counts = np.bincount(flat_labels, minlength=max_id + 1)
    tumor_counts = np.bincount(flat_labels, weights=flat_gt, minlength=max_id + 1)
    safe_counts = np.where(counts > 0, counts, 1)
    overlap_frac = tumor_counts / safe_counts

    # Supernodes with >10% tumor overlap are predicted as tumor
    tumor_sids = np.where((overlap_frac > 0.1) & (np.arange(max_id + 1) > 0))[0]
    pred_mask = np.isin(flat_labels, tumor_sids)

    intersection = (pred_mask & gt_mask.ravel()).sum()
    return float(2 * intersection / (pred_mask.sum() + gt_mask.ravel().sum() + 1e-8))


def few_shot_search(volumes: list,
                    gt_segs: list,
                    target_label: int = 2,
                    k_psi: int = 16,
                    k_alpha: int = 8,
                    k_beta: int = 8,
                    verbose: bool = True):
    """
    Grid search for optimal (ψ, α, β_min) on a few-shot set.

    Parameters
    ----------
    volumes   : list of 3D float arrays (HU-windowed)
    gt_segs   : list of 3D int arrays (0=bg, 1=organ, 2=tumor)
    k_psi     : number of ψ values to try
    k_alpha   : number of α values to try
    k_beta    : number of β_min values to try

    Returns
    -------
    best_params : dict {psi, alpha, beta_min, beta_max}
    search_log  : list of dicts with all evaluations
    """
    import time

    # Determine intensity range from the data
    all_vals = np.concatenate([v.ravel() for v in volumes])
    v_min, v_max = float(np.percentile(all_vals, 1)), float(np.percentile(all_vals, 99))
    v_range = v_max - v_min

    # Parameter grids — wider ψ range to target ~1000 supernodes (paper's range)
    # With proper recursive contraction, larger ψ merges more aggressively
    psi_vals = np.linspace(v_range * 0.02, v_range * 0.35, k_psi)
    alpha_vals = np.linspace(v_range * 0.05, v_range * 0.5, k_alpha)
    beta_min_vals = np.logspace(1, 3.5, k_beta).astype(int)  # 10 to ~3000
    beta_max = 500000

    # Intensity bounds for node deletion (keep clinically relevant range)
    m_min = v_min
    m_max = v_max

    total = k_psi * k_alpha * k_beta
    if verbose:
        print(f"  Few-shot search: {len(volumes)} volumes, "
              f"{k_psi}×{k_alpha}×{k_beta}={total} configs")
        print(f"  Intensity range: [{v_min:.0f}, {v_max:.0f}]")
        print(f"  ψ range: [{psi_vals[0]:.1f}, {psi_vals[-1]:.1f}]")

    search_log = []
    best_dice = -1
    best_params = {}
    t0 = time.time()
    evaluated = 0

    for psi in psi_vals:
        for alpha in alpha_vals:
            for beta_min in beta_min_vals:
                dices = []
                n_supernodes_list = []
                for vol, seg in zip(volumes, gt_segs):
                    result = build_graph_minor(
                        vol, psi=psi, alpha=alpha,
                        beta_min=int(beta_min), beta_max=beta_max,
                        m_min=m_min, m_max=m_max, fast=True
                    )
                    d = boundary_dice(result["labels"], seg, target_label)
                    dices.append(d)
                    n_supernodes_list.append(result["n_supernodes"])

                mean_dice = float(np.mean(dices))
                mean_sn = float(np.mean(n_supernodes_list))
                mean_voxels = float(np.mean([np.prod(v.shape) for v in volumes]))
                compression = mean_voxels / max(mean_sn, 1)

                # Paper Eq. 5: pure boundary Dice — no compression bonus
                score = mean_dice

                entry = {
                    "psi": round(float(psi), 2),
                    "alpha": round(float(alpha), 2),
                    "beta_min": int(beta_min),
                    "mean_dice": round(mean_dice, 4),
                    "score": round(float(score), 4),
                    "mean_supernodes": round(mean_sn, 0),
                    "compression": round(compression, 1),
                    "per_volume_dice": [round(d, 4) for d in dices],
                }
                search_log.append(entry)

                if score > best_dice:
                    best_dice = score
                    best_params = {
                        "psi": round(float(psi), 2),
                        "alpha": round(float(alpha), 2),
                        "beta_min": int(beta_min),
                        "beta_max": beta_max,
                        "m_min": round(m_min, 1),
                        "m_max": round(m_max, 1),
                    }

                evaluated += 1

    elapsed = time.time() - t0
    if verbose:
        print(f"  Search complete: {evaluated} configs in {elapsed:.1f}s", flush=True)
        print(f"  Best: Score={best_dice:.4f}  ψ={best_params['psi']}"
              f"  α={best_params['alpha']}  β_min={best_params['beta_min']}", flush=True)

    return best_params, search_log
